from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import dspy

from signatures import (
    CollectEvidence,
    BuildHypothesis,
    CheckContradictions,
    DesignSystem,
    GenerateMermaid,
)
from search import gather_evidence

logger = logging.getLogger(__name__)

# Timeout (seconds) for each individual LLM call
_LLM_TIMEOUT = 120.0
# Timeout for the web-search step
_SEARCH_TIMEOUT = 30.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_confidence(raw) -> int:
    """Robustly convert whatever the LLM returns for confidence → int 0-100.

    Handles: 75, 75.0, "75", "75%", "~80", "high (85)", etc.
    Falls back to 50 on parse failure.
    """
    try:
        cleaned = str(raw).strip().rstrip('%').strip()
        # Strip any leading non-numeric characters (e.g. "~", "≈")
        cleaned = cleaned.lstrip('~≈≤≥<> ')
        # Take the first numeric token in case the LLM added prose
        numeric = ''.join(ch for ch in cleaned.split()[0] if ch.isdigit() or ch == '.')
        return max(0, min(100, int(float(numeric))))
    except (ValueError, TypeError, IndexError):
        logger.warning("Could not parse confidence value %r — defaulting to 50.", raw)
        return 50


def _error_event(message: str, progress: int) -> dict:
    return {
        "stage": "error",
        "message": message,
        "data": None,
        "progress": progress,
    }


def clean_mermaid(raw: str) -> str:
    """Strip code fences (``` or ~~~, with optional language tag) from LLM output."""
    raw = raw.strip()
    for fence in ("```", "~~~"):
        if raw.startswith(fence):
            lines = raw.splitlines()
            # Drop the opening fence line (may include a language tag e.g. ```mermaid)
            lines = lines[1:]
            # Drop the closing fence line if present (allow trailing whitespace)
            if lines and lines[-1].strip() in ("```", "~~~", f"{fence}"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            break
    return raw


# ── DSPy Module ───────────────────────────────────────────────────────────────

class ReverseEngineer(dspy.Module):
    """Full pipeline used for batch evaluation and DSPy optimization."""

    def __init__(self) -> None:
        super().__init__()
        self.collect    = dspy.ChainOfThought(CollectEvidence)
        self.hypothesis = dspy.ChainOfThought(BuildHypothesis)
        self.check      = dspy.ChainOfThought(CheckContradictions)
        self.design     = dspy.ChainOfThought(DesignSystem)
        self.mermaid    = dspy.ChainOfThought(GenerateMermaid)

    def forward(self, system_name: str, web_evidence: list[str] | None = None) -> dspy.Prediction:
        ev_pred = self.collect(system_name=system_name)
        combined_evidence: list[str] = (web_evidence or []) + ev_pred.evidence

        hypo_pred = self.hypothesis(
            system_name=system_name,
            evidence=combined_evidence,
        )

        contra_pred = self.check(
            evidence=combined_evidence,
            architecture=hypo_pred.architecture,
        )

        design_pred = self.design(
            system_name=system_name,
            evidence=combined_evidence,
            architecture=hypo_pred.architecture,
            contradictions=contra_pred.contradictions,
        )

        mermaid_pred = self.mermaid(
            system_name=system_name,
            final_design=design_pred.final_design,
        )

        return dspy.Prediction(
            evidence=combined_evidence,
            architecture=hypo_pred.architecture,
            contradictions=contra_pred.contradictions,
            final_design=design_pred.final_design,
            confidence=_parse_confidence(design_pred.confidence),
            alternative_hypotheses=design_pred.alternative_hypotheses,
            mermaid_code=clean_mermaid(mermaid_pred.mermaid_code),
        )


# ── Streaming pipeline ────────────────────────────────────────────────────────

async def stream_analysis(
    system_name: str,
) -> AsyncGenerator[dict, None]:
    """Yield SSE-compatible dicts for each pipeline stage.

    Emits an ``error`` stage event and returns early on any failure so the
    client always receives a terminal event rather than a dead stream.
    """
    loop = asyncio.get_running_loop()

    # ── 1. Web evidence (non-fatal — degrade gracefully) ─────────────────────
    try:
        web_evidence: list[str] = await asyncio.wait_for(
            loop.run_in_executor(None, gather_evidence, system_name),
            timeout=_SEARCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        web_evidence = []
        logger.warning("Web search timed out for '%s' — continuing without it.", system_name)
    except Exception as exc:
        web_evidence = []
        logger.warning("Web search failed for '%s': %s — continuing.", system_name, exc)

    yield {
        "stage": "evidence_web",
        "message": f"Found {len(web_evidence)} web sources via Tavily",
        "data": web_evidence[:5],
        "progress": 10,
    }

    # Instantiate the module once so sub-predictors are reused across steps
    module = ReverseEngineer()

    # ── 2. LLM evidence collection ────────────────────────────────────────────
    try:
        ev_pred = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: module.collect(system_name=system_name)),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield _error_event("Evidence collection timed out — try again or reduce scope.", 20)
        return
    except Exception as exc:
        yield _error_event(f"Evidence collection failed: {exc}", 20)
        return

    combined_evidence: list[str] = web_evidence + ev_pred.evidence

    yield {
        "stage": "evidence",
        "message": f"Evidence collected — {len(combined_evidence)} observations total",
        "data": combined_evidence,
        "progress": 20,
    }

    # ── 3. Architecture hypothesis ────────────────────────────────────────────
    try:
        hypo_pred = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.hypothesis(system_name=system_name, evidence=combined_evidence),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield _error_event("Hypothesis building timed out.", 40)
        return
    except Exception as exc:
        yield _error_event(f"Hypothesis building failed: {exc}", 40)
        return

    yield {
        "stage": "hypothesis",
        "message": "Initial architecture hypothesis built",
        "data": hypo_pred.architecture,
        "progress": 40,
    }

    # ── 4. Contradiction check ────────────────────────────────────────────────
    try:
        contra_pred = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.check(
                    evidence=combined_evidence,
                    architecture=hypo_pred.architecture,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield _error_event("Contradiction check timed out.", 60)
        return
    except Exception as exc:
        yield _error_event(f"Contradiction check failed: {exc}", 60)
        return

    yield {
        "stage": "contradictions",
        "message": f"Found {len(contra_pred.contradictions)} contradictions",
        "data": contra_pred.contradictions,
        "progress": 60,
    }

    # ── 5. Final design synthesis ─────────────────────────────────────────────
    try:
        design_pred = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.design(
                    system_name=system_name,
                    evidence=combined_evidence,
                    architecture=hypo_pred.architecture,
                    contradictions=contra_pred.contradictions,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield _error_event("Design synthesis timed out.", 80)
        return
    except Exception as exc:
        yield _error_event(f"Design synthesis failed: {exc}", 80)
        return

    confidence = _parse_confidence(design_pred.confidence)

    yield {
        "stage": "design",
        "message": f"Final design synthesized — confidence {confidence}%",
        "data": {
            "final_design": design_pred.final_design,
            "confidence": confidence,
            "alternative_hypotheses": design_pred.alternative_hypotheses,
        },
        "progress": 80,
    }

    # ── 6. Mermaid diagram (non-fatal — empty string on failure) ──────────────
    mermaid_code = ""
    try:
        mermaid_pred = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.mermaid(
                    system_name=system_name,
                    final_design=design_pred.final_design,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
        mermaid_code = clean_mermaid(mermaid_pred.mermaid_code)
    except asyncio.TimeoutError:
        logger.warning("Mermaid generation timed out for '%s' — skipping diagram.", system_name)
    except Exception as exc:
        logger.warning("Mermaid generation failed for '%s': %s — skipping.", system_name, exc)

    yield {
        "stage": "mermaid",
        "message": "Architecture diagram generated" if mermaid_code else "Diagram unavailable",
        "data": mermaid_code,
        "progress": 95,
    }

    yield {
        "stage": "done",
        "message": "Analysis complete",
        "data": {
            "system_name": system_name,
            "evidence": combined_evidence,
            "architecture": hypo_pred.architecture,
            "contradictions": contra_pred.contradictions,
            "final_design": design_pred.final_design,
            "confidence": confidence,
            "alternative_hypotheses": design_pred.alternative_hypotheses,
            "mermaid_code": mermaid_code,
        },
        "progress": 100,
    }
