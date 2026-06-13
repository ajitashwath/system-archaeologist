from __future__ import annotations

import json
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Any

import dspy
from dotenv import load_dotenv

load_dotenv()


sys.path.insert(0, os.path.dirname(__file__))

from pipeline import ReverseEngineer
from search import gather_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class KnownSystem:
    name: str
    known_components: list[str]       # must appear in output
    known_technologies: list[str]     # expected tech stack terms
    known_patterns: list[str]         # architectural patterns (e.g., "event sourcing")
    notes: str = ""


EVAL_DATASET: list[KnownSystem] = [
    KnownSystem(
        name="Redis",
        known_components=["in-memory store", "key-value", "pub/sub", "persistence", "replication", "cluster"],
        known_technologies=["C", "ANSI C", "TCP", "RDB", "AOF"],
        known_patterns=["single-threaded event loop", "master-replica", "sentinel"],
        notes="Well-documented open-source system — should score high.",
    ),
    KnownSystem(
        name="Git",
        known_components=["object store", "DAG", "blob", "tree", "commit", "ref", "index", "working tree"],
        known_technologies=["SHA-1", "zlib compression", "pack files"],
        known_patterns=["content-addressed storage", "directed acyclic graph", "copy-on-write"],
        notes="Fully open-source — near-perfect recall expected.",
    ),
    KnownSystem(
        name="Docker",
        known_components=["container runtime", "image layers", "overlay filesystem", "namespace", "cgroup", "registry", "daemon"],
        known_technologies=["containerd", "runc", "OCI", "UnionFS", "Linux namespaces"],
        known_patterns=["copy-on-write layers", "client-server daemon", "image registry"],
        notes="Well-documented open-source — should score high.",
    ),
    KnownSystem(
        name="Nginx",
        known_components=["event loop", "worker process", "master process", "reverse proxy", "load balancer", "upstream"],
        known_technologies=["epoll", "kqueue", "C", "non-blocking I/O"],
        known_patterns=["async event-driven", "master-worker", "upstream pooling"],
        notes="Classic system — well-studied.",
    ),
    KnownSystem(
        name="Kafka",
        known_components=["broker", "topic", "partition", "consumer group", "producer", "offset", "zookeeper", "kraft"],
        known_technologies=["JVM", "log segments", "page cache", "zero-copy"],
        known_patterns=["distributed log", "pub-sub", "consumer group rebalancing"],
        notes="Partially open-source — high recall expected.",
    ),
]


def component_recall(known: KnownSystem, result: dict[str, Any]) -> float:
    output_text = (
        result.get("final_design", "") + " " +
        result.get("architecture", "") + " " +
        " ".join(result.get("evidence", []))
    ).lower()

    found = sum(1 for c in known.known_components if c.lower() in output_text)
    return found / len(known.known_components) if known.known_components else 0.0


def technology_recall(known: KnownSystem, result: dict[str, Any]) -> float:
    output_text = (
        result.get("final_design", "") + " " + result.get("architecture", "")
    ).lower()
    found = sum(1 for t in known.known_technologies if t.lower() in output_text)
    return found / len(known.known_technologies) if known.known_technologies else 0.0


def pattern_recall(known: KnownSystem, result: dict[str, Any]) -> float:
    output_text = (result.get("final_design", "") + " " + result.get("architecture", "")).lower()
    found = sum(1 for p in known.known_patterns if p.lower() in output_text)
    return found / len(known.known_patterns) if known.known_patterns else 0.0


def confidence_calibration_error(accuracy: float, confidence: int) -> float:
    return abs(accuracy - (confidence / 100.0))


def dspy_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    known_name = example.system_name
    known = next((k for k in EVAL_DATASET if k.name == known_name), None)
    if known is None:
        return 0.0

    result = {
        "final_design": getattr(pred, "final_design", ""),
        "architecture": getattr(pred, "architecture", ""),
        "evidence": getattr(pred, "evidence", []),
    }

    comp = component_recall(known, result)
    tech = technology_recall(known, result)
    return 0.6 * comp + 0.4 * tech


@dataclass
class EvalResult:
    system_name: str
    component_recall: float
    technology_recall: float
    pattern_recall: float
    confidence: int
    calibration_error: float
    overall_accuracy: float = field(init=False)

    def __post_init__(self):
        self.overall_accuracy = (
            0.5 * self.component_recall +
            0.3 * self.technology_recall +
            0.2 * self.pattern_recall
        )


def run_evaluation() -> list[EvalResult]:
    results: list[EvalResult] = []
    module = ReverseEngineer()

    for known in EVAL_DATASET:
        logger.info("Evaluating: %s …", known.name)
        try:
            web_evidence = gather_evidence(known.name, max_results_per_query=3)
            pred = module(system_name=known.name, web_evidence=web_evidence)

            result_dict = {
                "final_design": pred.final_design,
                "architecture": pred.architecture,
                "evidence": pred.evidence,
            }

            comp  = component_recall(known, result_dict)
            tech  = technology_recall(known, result_dict)
            pat   = pattern_recall(known, result_dict)
            conf  = int(pred.confidence) if str(pred.confidence).isdigit() else 0
            cal   = confidence_calibration_error(0.6 * comp + 0.4 * tech, conf)

            er = EvalResult(
                system_name=known.name,
                component_recall=comp,
                technology_recall=tech,
                pattern_recall=pat,
                confidence=conf,
                calibration_error=cal,
            )
            results.append(er)

            logger.info(
                "  %s → comp=%.2f tech=%.2f pat=%.2f conf=%d%% calib_err=%.2f",
                known.name, comp, tech, pat, conf, cal,
            )

        except Exception as e:
            logger.error("Failed to evaluate %s: %s", known.name, e)

    return results


def print_report(results: list[EvalResult]) -> None:
    print("\n" + "=" * 70)
    print("  SYSTEM ARCHAEOLOGIST — EVALUATION REPORT")
    print("=" * 70)
    print(f"{'System':<15} {'CompRecall':>10} {'TechRecall':>10} {'PatRecall':>10} {'Conf':>6} {'CalErr':>8} {'Score':>8}")
    print("-" * 70)

    total_score = 0.0
    for r in results:
        print(
            f"{r.system_name:<15} {r.component_recall:>10.2f} {r.technology_recall:>10.2f} "
            f"{r.pattern_recall:>10.2f} {r.confidence:>5d}% {r.calibration_error:>8.2f} {r.overall_accuracy:>8.2f}"
        )
        total_score += r.overall_accuracy

    avg = total_score / len(results) if results else 0.0
    print("-" * 70)
    print(f"{'AVERAGE':<15} {'':>10} {'':>10} {'':>10} {'':>6} {'':>8} {avg:>8.2f}")
    print("=" * 70 + "\n")


def run_optimization():
    from dspy.teleprompt import BootstrapFewShot

    trainset = [
        dspy.Example(system_name=k.name).with_inputs("system_name")
        for k in EVAL_DATASET
    ]

    optimizer = BootstrapFewShot(
        metric=dspy_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        max_rounds=2,
    )

    compiled = optimizer.compile(
        ReverseEngineer(),
        trainset=trainset,
    )

    compiled.save("optimized_pipeline.json")
    logger.info("Optimized pipeline saved to optimized_pipeline.json")
    return compiled


if __name__ == "__main__":
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
    GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        lm = dspy.LM(f"ollama/{OLLAMA_MODEL}", api_base=OLLAMA_BASE_URL)
        lm("Say OK", cache=False)
        logger.info("Using Ollama: %s", OLLAMA_MODEL)
    except Exception:
        if not GOOGLE_API_KEY:
            raise RuntimeError("Neither Ollama nor GOOGLE_API_KEY available.")
        lm = dspy.LM(f"google/{GEMINI_MODEL}", api_key=GOOGLE_API_KEY)
        logger.info("Using Gemini: %s", GEMINI_MODEL)

    dspy.configure(lm=lm)

    results = run_evaluation()
    print_report(results)

