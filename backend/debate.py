from __future__ import annotations

import asyncio
import logging

import dspy

from signatures import DebatePosition, JudgeDebate

logger = logging.getLogger(__name__)

_LLM_TIMEOUT = 90.0


class ExtractDebateTopics(dspy.Signature):

    system_name: str = dspy.InputField()
    architecture: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()

    debate_topic: str = dspy.OutputField(
        desc="The contested architectural question in one sentence (e.g., 'Does X use local or cloud-based indexing?')"
    )
    position_a: str = dspy.OutputField(
        desc="First position to argue (e.g., 'X uses local, on-device indexing')"
    )
    position_b: str = dspy.OutputField(
        desc="Second position, opposing position_a (e.g., 'X uses cloud-based indexing')"
    )


class MultiAgentDebate(dspy.Module):
    """Used for batch / evaluation runs. Streaming uses stream_debate()."""

    def __init__(self) -> None:
        super().__init__()
        self.topic_extractor = dspy.ChainOfThought(ExtractDebateTopics)
        self.agent_a         = dspy.ChainOfThought(DebatePosition)
        self.agent_b         = dspy.ChainOfThought(DebatePosition)
        self.judge           = dspy.ChainOfThought(JudgeDebate)

    def forward(
        self,
        system_name: str,
        architecture: str,
        evidence: list[str],
    ) -> dspy.Prediction:
        topics = self.topic_extractor(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
        )

        arg_a = self.agent_a(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
            position=topics.position_a,
        )
        arg_b = self.agent_b(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
            position=topics.position_b,
        )
        verdict = self.judge(
            system_name=system_name,
            evidence=evidence,
            argument_a=arg_a.argument,
            argument_b=arg_b.argument,
        )

        return dspy.Prediction(
            debate_topic=topics.debate_topic,
            position_a=topics.position_a,
            position_b=topics.position_b,
            argument_a=arg_a.argument,
            argument_b=arg_b.argument,
            verdict=verdict.verdict,
            winning_position=verdict.winning_position,
            synthesis=verdict.synthesis,
        )


def _truncate(text: str, max_len: int = 60) -> str:
    return text[:max_len] + ("..." if len(text) > max_len else "")


async def stream_debate(
    system_name: str,
    architecture: str,
    evidence: list[str],
):
    """Yield debate stages one by one.

    Emits a ``debate_error`` stage on any failure so the client always
    receives a terminal event rather than a dead stream.
    """
    module = MultiAgentDebate()
    loop = asyncio.get_running_loop()

    # ── Extract debate topic ──────────────────────────────────────────────────
    try:
        topics = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.topic_extractor(
                    system_name=system_name,
                    architecture=architecture,
                    evidence=evidence,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield {"stage": "debate_error", "message": "Debate topic extraction timed out.", "data": None, "progress": 0}
        return
    except Exception as exc:
        yield {"stage": "debate_error", "message": f"Debate topic extraction failed: {exc}", "data": None, "progress": 0}
        return

    yield {
        "stage": "debate_topic",
        "message": "Debate topic identified",
        "data": {
            "topic": topics.debate_topic,
            "position_a": topics.position_a,
            "position_b": topics.position_b,
        },
        "progress": 0,
    }

    # ── Agent A ───────────────────────────────────────────────────────────────
    try:
        arg_a = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.agent_a(
                    system_name=system_name,
                    architecture=architecture,
                    evidence=evidence,
                    position=topics.position_a,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield {"stage": "debate_error", "message": "Agent A timed out.", "data": None, "progress": 33}
        return
    except Exception as exc:
        yield {"stage": "debate_error", "message": f"Agent A failed: {exc}", "data": None, "progress": 33}
        return

    yield {
        "stage": "agent_a",
        "message": f"Agent A argues: {_truncate(topics.position_a)}",
        "data": {
            "position": topics.position_a,
            "argument": arg_a.argument,
        },
        "progress": 33,
    }

    # ── Agent B ───────────────────────────────────────────────────────────────
    try:
        arg_b = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.agent_b(
                    system_name=system_name,
                    architecture=architecture,
                    evidence=evidence,
                    position=topics.position_b,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield {"stage": "debate_error", "message": "Agent B timed out.", "data": None, "progress": 66}
        return
    except Exception as exc:
        yield {"stage": "debate_error", "message": f"Agent B failed: {exc}", "data": None, "progress": 66}
        return

    yield {
        "stage": "agent_b",
        "message": f"Agent B argues: {_truncate(topics.position_b)}",
        "data": {
            "position": topics.position_b,
            "argument": arg_b.argument,
        },
        "progress": 66,
    }

    # ── Judge ─────────────────────────────────────────────────────────────────
    try:
        verdict = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: module.judge(
                    system_name=system_name,
                    evidence=evidence,
                    argument_a=arg_a.argument,
                    argument_b=arg_b.argument,
                ),
            ),
            timeout=_LLM_TIMEOUT,
        )
    except asyncio.TimeoutError:
        yield {"stage": "debate_error", "message": "Judge timed out.", "data": None, "progress": 90}
        return
    except Exception as exc:
        yield {"stage": "debate_error", "message": f"Judge failed: {exc}", "data": None, "progress": 90}
        return

    yield {
        "stage": "verdict",
        "message": f"Judge rules: {verdict.winning_position}",
        "data": {
            "verdict": verdict.verdict,
            "winning_position": verdict.winning_position,
            "synthesis": verdict.synthesis,
        },
        "progress": 100,
    }
