from __future__ import annotations

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


class ReverseEngineer(dspy.Module):

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
            confidence=design_pred.confidence,
            alternative_hypotheses=design_pred.alternative_hypotheses,
            mermaid_code=mermaid_pred.mermaid_code,
        )


async def stream_analysis(
    system_name: str,
) -> AsyncGenerator[dict, None]:

    import asyncio
    web_evidence = await asyncio.get_event_loop().run_in_executor(
        None, gather_evidence, system_name
    )

    yield {
        "stage": "evidence_web",
        "message": f"Found {len(web_evidence)} web sources via Tavily",
        "data": web_evidence[:5],
        "progress": 10,
    }

    module = ReverseEngineer()

    ev_pred = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: module.collect(system_name=system_name),
    )
    combined_evidence: list[str] = web_evidence + ev_pred.evidence

    yield {
        "stage": "evidence",
        "message": f"Evidence collected — {len(combined_evidence)} observations total",
        "data": combined_evidence,
        "progress": 20,
    }

    hypo_pred = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: module.hypothesis(system_name=system_name, evidence=combined_evidence),
    )

    yield {
        "stage": "hypothesis",
        "message": "Initial architecture hypothesis built",
        "data": hypo_pred.architecture,
        "progress": 40,
    }

    contra_pred = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: module.check(
            evidence=combined_evidence,
            architecture=hypo_pred.architecture,
        ),
    )

    yield {
        "stage": "contradictions",
        "message": f"Found {len(contra_pred.contradictions)} contradictions",
        "data": contra_pred.contradictions,
        "progress": 60,
    }

    design_pred = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: module.design(
            system_name=system_name,
            evidence=combined_evidence,
            architecture=hypo_pred.architecture,
            contradictions=contra_pred.contradictions,
        ),
    )

    yield {
        "stage": "design",
        "message": f"Final design synthesized — confidence {design_pred.confidence}%",
        "data": {
            "final_design": design_pred.final_design,
            "confidence": design_pred.confidence,
            "alternative_hypotheses": design_pred.alternative_hypotheses,
        },
        "progress": 80,
    }

    mermaid_pred = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: module.mermaid(
            system_name=system_name,
            final_design=design_pred.final_design,
        ),
    )

    mermaid_code = clean_mermaid(mermaid_pred.mermaid_code)

    yield {
        "stage": "mermaid",
        "message": "Architecture diagram generated",
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
            "confidence": design_pred.confidence,
            "alternative_hypotheses": design_pred.alternative_hypotheses,
            "mermaid_code": mermaid_code,
        },
        "progress": 100,
    }

def clean_mermaid(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).strip()
    return raw
