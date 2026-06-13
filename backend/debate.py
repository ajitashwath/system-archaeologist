from __future__ import annotations

import asyncio
import logging

import dspy

from signatures import DebatePosition, JudgeDebate

logger = logging.getLogger(__name__)


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


async def stream_debate(
    system_name: str,
    architecture: str,
    evidence: list[str],
):
    debate_module = MultiAgentDebate()
    loop = asyncio.get_event_loop()
    topics = await loop.run_in_executor(
        None,
        lambda: debate_module.topic_extractor(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
        ),
    )

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

    arg_a = await loop.run_in_executor(
        None,
        lambda: debate_module.agent_a(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
            position=topics.position_a,
        ),
    )

    yield {
        "stage": "agent_a",
        "message": f"Agent A argues: {topics.position_a[:60]}...",
        "data": {
            "position": topics.position_a,
            "argument": arg_a.argument,
        },
        "progress": 33,
    }
    arg_b = await loop.run_in_executor(
        None,
        lambda: debate_module.agent_b(
            system_name=system_name,
            architecture=architecture,
            evidence=evidence,
            position=topics.position_b,
        ),
    )

    yield {
        "stage": "agent_b",
        "message": f"Agent B argues: {topics.position_b[:60]}...",
        "data": {
            "position": topics.position_b,
            "argument": arg_b.argument,
        },
        "progress": 66,
    }
    verdict = await loop.run_in_executor(
        None,
        lambda: debate_module.judge(
            system_name=system_name,
            evidence=evidence,
            argument_a=arg_a.argument,
            argument_b=arg_b.argument,
        ),
    )

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
