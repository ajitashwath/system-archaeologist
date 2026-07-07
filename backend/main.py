from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx

import dspy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from debate import stream_debate
from pipeline import stream_analysis

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CORS_ORIGINS    = os.getenv("CORS_ORIGINS", "http://localhost:4321").split(",")


# ── LM initialisation ─────────────────────────────────────────────────────────

async def _is_ollama_running() -> bool:
    """Async HTTP ping to Ollama's /api/tags endpoint (3 s timeout)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
            return resp.status_code == 200
    except Exception:
        return False


async def init_lm() -> dspy.LM:
    if await _is_ollama_running():
        try:
            lm = dspy.LM(
                f"ollama/{OLLAMA_MODEL}",
                api_base=OLLAMA_BASE_URL,
                temperature=0.7,
                max_tokens=4096,
            )
            logger.info("✅ Ollama LLM active: %s @ %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
            return lm
        except Exception as e:
            logger.warning("⚠️  Ollama reachable but failed to load model (%s) — falling back to Gemini.", e)
    else:
        logger.warning("⚠️  Ollama not running at %s — falling back to Gemini.", OLLAMA_BASE_URL)

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Ollama is not running and GOOGLE_API_KEY is not set. "
            "Start Ollama or set GOOGLE_API_KEY in .env"
        )

    lm = dspy.LM(
        f"gemini/{GEMINI_MODEL}",
        api_key=GOOGLE_API_KEY,
        temperature=0.7,
        max_tokens=4096,
    )
    logger.info("✅ Gemini LLM active: %s", GEMINI_MODEL)
    return lm


# ── App state ─────────────────────────────────────────────────────────────────

_active_lm: dspy.LM | None = None
_lm_provider: str = "unknown"

# Tracks system names currently being analysed to prevent duplicate parallel runs
_inflight: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _active_lm, _lm_provider

    _active_lm = await init_lm()
    _lm_provider = "ollama" if "ollama" in _active_lm.model else "gemini"
    dspy.configure(lm=_active_lm, async_max_workers=8)

    await db.init_pool()

    logger.info("🚀 System Archaeologist API ready.")
    yield

    await db.close_pool()
    logger.info("👋 Shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="System Archaeologist API",
    description="Reverse-engineer how any product works using evidence-based DSPy reasoning.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    system_name: str
    include_debate: bool = True


class AnalysisResponse(BaseModel):
    id: str
    system_name: str
    evidence: list[str]
    architecture: str
    contradictions: list[str]
    final_design: str
    mermaid_code: str
    debate: dict[str, Any]
    confidence: int
    alternative_hypotheses: list[str]
    created_at: str | None = None


# ── SSE helpers ───────────────────────────────────────────────────────────────

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def pipeline_sse_generator(
    system_name: str,
    include_debate: bool = True,
    save_to_db: bool = True,
) -> AsyncGenerator[str, None]:
    """Drive the full pipeline and yield SSE-formatted events.

    * Tracks in-flight requests so duplicate concurrent analyses are blocked.
    * Catches errors from stream_analysis / stream_debate and forwards them
      to the client as error-stage events instead of crashing the stream.
    """
    key = system_name.lower()
    full_result: dict[str, Any] = {}
    debate_result: dict[str, Any] = {}

    _inflight.add(key)
    try:
        # ── Analysis pipeline ─────────────────────────────────────────────────
        try:
            async for chunk in stream_analysis(system_name):
                yield sse_event(chunk)
                if chunk["stage"] == "error":
                    return      # pipeline already emitted the error event
                if chunk["stage"] == "done":
                    full_result = chunk["data"]
        except Exception as exc:
            logger.exception("Unexpected error in stream_analysis for '%s'", system_name)
            yield sse_event({
                "stage": "error",
                "message": f"Pipeline error: {exc}",
                "data": None,
                "progress": 0,
            })
            return

        # ── Debate ────────────────────────────────────────────────────────────
        if include_debate and full_result:
            yield sse_event({
                "stage": "debate_start",
                "message": "Starting multi-agent debate…",
                "data": None,
                "progress": 0,
            })
            try:
                async for chunk in stream_debate(
                    system_name=system_name,
                    architecture=full_result.get("architecture", ""),
                    evidence=full_result.get("evidence", []),
                ):
                    yield sse_event(chunk)
                    if chunk["stage"] == "debate_error":
                        logger.warning("Debate failed for '%s': %s", system_name, chunk["message"])
                        break   # non-fatal — continue to save
                    if chunk["stage"] == "verdict":
                        debate_result = chunk["data"]
            except Exception as exc:
                logger.exception("Unexpected error in stream_debate for '%s'", system_name)
                yield sse_event({
                    "stage": "debate_error",
                    "message": f"Debate error: {exc}",
                    "data": None,
                    "progress": 0,
                })
                # non-fatal — fall through to save

        # ── Persist ───────────────────────────────────────────────────────────
        analysis_id: str | None = None
        if save_to_db and full_result:
            try:
                analysis_id = await db.save_analysis(
                    system_name=system_name,
                    evidence=full_result.get("evidence", []),
                    architecture=full_result.get("architecture", ""),
                    contradictions=full_result.get("contradictions", []),
                    final_design=full_result.get("final_design", ""),
                    mermaid_code=full_result.get("mermaid_code", ""),
                    debate=debate_result,
                    confidence=full_result.get("confidence", 0),
                    alternative_hypotheses=full_result.get("alternative_hypotheses", []),
                )
            except Exception as e:
                logger.error("Failed to save analysis to DB: %s", e)

        yield sse_event({
            "stage": "saved",
            "message": "Analysis saved to database" if analysis_id else "Analysis complete (not saved)",
            "data": {"id": analysis_id},
            "progress": 100,
        })

    finally:
        _inflight.discard(key)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    db_ok = False
    try:
        await db.list_analyses(limit=1)
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "lm_provider": _lm_provider,
        "lm_model": _active_lm.model if _active_lm else None,
        "database": "connected" if db_ok else "error",
    }


@app.get("/api/analyze/stream")
async def analyze_stream(
    system: str = Query(..., description="Product or system name to reverse-engineer"),
    debate: bool = Query(True, description="Include multi-agent debate"),
):
    system = system.strip()
    if not system:
        raise HTTPException(status_code=400, detail="system query param cannot be empty")

    if system.lower() in _inflight:
        raise HTTPException(
            status_code=409,
            detail=f"Analysis of '{system}' is already in progress. Please wait.",
        )

    return StreamingResponse(
        pipeline_sse_generator(system, include_debate=debate),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_blocking(body: AnalyzeRequest):
    system_name = body.system_name.strip()
    if not system_name:
        raise HTTPException(status_code=400, detail="system_name cannot be empty")

    if system_name.lower() in _inflight:
        raise HTTPException(
            status_code=409,
            detail=f"Analysis of '{system_name}' is already in progress. Please wait.",
        )

    full_result: dict[str, Any] = {}
    debate_result: dict[str, Any] = {}
    analysis_id: str | None = None

    async for chunk in pipeline_sse_generator(
        system_name,
        include_debate=body.include_debate,
        save_to_db=True,
    ):
        if chunk.startswith("data: "):
            data = json.loads(chunk[6:])
            if data["stage"] == "error":
                raise HTTPException(status_code=500, detail=data["message"])
            elif data["stage"] == "done":
                full_result = data["data"]
            elif data["stage"] == "verdict":
                debate_result = data["data"]
            elif data["stage"] == "saved":
                analysis_id = (data.get("data") or {}).get("id")

    if not full_result:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")

    return {
        **full_result,
        "id": analysis_id or "",
        "debate": debate_result,
    }


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    row = await db.get_analysis(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found")
    return row


@app.get("/api/analyses")
async def list_analyses(
    limit: int = Query(20, ge=1, le=100),
    search: str = Query("", description="Filter by system name"),
    all_runs: bool = Query(False, description="If true, return all runs; otherwise return latest per system"),
):
    if search.strip():
        return await db.search_analyses(search.strip(), limit=limit)
    return await db.list_analyses(limit=limit, deduplicate=not all_runs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
    )
