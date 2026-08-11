from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    assumption_change: str
    rationale: str
    validation_mse: float
    complexity_note: str


def run_experiment(name: str, assumption_change: str, rationale: str,
                   complexity_note: str, evaluate: Callable[[], float]):
    """Small harness enforcing the Dave Rule around an experiment."""
    return ExperimentResult(
        name=name,
        assumption_change=assumption_change,
        rationale=rationale,
        validation_mse=float(evaluate()),
        complexity_note=complexity_note,
    )
