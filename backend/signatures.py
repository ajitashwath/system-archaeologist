from __future__ import annotations

import dspy


class CollectEvidence(dspy.Signature):

    system_name: str = dspy.InputField(
        desc="The name of the software product or system to investigate"
    )
    evidence: list[str] = dspy.OutputField(
        desc=(
            "A list of 8-15 distinct factual observations about the system. "
            "Each item is one sentence. Be specific (e.g. 'Cursor is built on top of VS Code, "
            "confirmed by their open-source fork at github.com/getcursor/cursor')."
        )
    )


class BuildHypothesis(dspy.Signature):

    system_name: str = dspy.InputField()
    evidence: list[str] = dspy.InputField(
        desc="Factual observations about the system gathered from public sources"
    )
    architecture: str = dspy.OutputField(
        desc=(
            "A structured architecture description in tree/outline format. "
            "Group by layer (Frontend, Backend, Data, Infrastructure, External Services). "
            "Include technology names where inferrable."
        )
    )


class CheckContradictions(dspy.Signature):

    evidence: list[str] = dspy.InputField()
    architecture: str = dspy.InputField(
        desc="The proposed architecture hypothesis to challenge"
    )
    contradictions: list[str] = dspy.OutputField(
        desc=(
            "A list of 3-6 specific contradictions or tensions. "
            "Format: 'Hypothesis assumes X, but evidence suggests Y because Z.'"
        )
    )


class DesignSystem(dspy.Signature):

    system_name: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    architecture: str = dspy.InputField(
        desc="Initial architecture hypothesis"
    )
    contradictions: list[str] = dspy.InputField(
        desc="Identified contradictions in the hypothesis"
    )
    final_design: str = dspy.OutputField(
        desc=(
            "The refined, most-probable architecture. Use clear section headers. "
            "Include: Most Likely Architecture, Key Design Decisions, Alternative Hypotheses, "
            "and Unresolved Questions."
        )
    )
    confidence: int = dspy.OutputField(
        desc=(
            "Overall confidence score from 0 to 100. "
            "80+ means strong evidence, 60-79 means moderate, below 60 means speculative."
        )
    )
    alternative_hypotheses: list[str] = dspy.OutputField(
        desc="2-3 plausible alternative architectures if the main hypothesis is wrong"
    )


class GenerateMermaid(dspy.Signature):

    system_name: str = dspy.InputField()
    final_design: str = dspy.InputField(
        desc="The refined architecture description to visualize"
    )
    mermaid_code: str = dspy.OutputField(
        desc=(
            "Valid Mermaid diagram syntax starting with 'graph TD' or 'graph LR'. "
            "Do NOT wrap in ``` fences. Nodes should use alphanumeric IDs. "
            "Example: graph TD\\n  A[Frontend] --> B[API Gateway]\\n  B --> C[Service]"
        )
    )


class DebatePosition(dspy.Signature):

    system_name: str = dspy.InputField()
    architecture: str = dspy.InputField(
        desc="The proposed main architecture for context"
    )
    evidence: list[str] = dspy.InputField()
    position: str = dspy.InputField(
        desc="The specific architectural stance you must argue for"
    )
    argument: str = dspy.OutputField(
        desc=(
            "A structured argument (3-5 sentences) defending your position. "
            "Reference specific evidence. End with your strongest supporting point."
        )
    )


class JudgeDebate(dspy.Signature):

    system_name: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    argument_a: str = dspy.InputField(desc="First agent's argument")
    argument_b: str = dspy.InputField(desc="Second agent's argument")
    verdict: str = dspy.OutputField(
        desc=(
            "A balanced evaluation of both arguments (3-4 sentences). "
            "Explain which evidence is more compelling and why."
        )
    )
    winning_position: str = dspy.OutputField(
        desc="One of: 'Agent A', 'Agent B', or 'Neither (hybrid)'"
    )
    synthesis: str = dspy.OutputField(
        desc="A synthesized conclusion that may draw from both arguments"
    )
