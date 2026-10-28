"""Perception and navigation backends: the seams where the real robot plugs
in later, without changes to the planner or the domain."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class Observation:
    """What one detect step saw. skipped = no image (beliefs unchanged);
    aborted = the user ended the session."""
    space: str
    found: bool = False
    confidence: float = None
    description: str = ""
    skipped: bool = False
    error: str = None
    aborted: bool = False

    def to_dict(self):
        return asdict(self)


class PerceptionBackend(ABC):
    model_calls = 0

    @abstractmethod
    def observe(self, space, label, task_text):
        """-> Observation for the image taken in `space` (`label` is for display)."""


class ManualImageBackend(PerceptionBackend):
    """Asks for an image path standing in for the camera, then runs the
    VisualDetector from interactive_search.py ('skip' = no image, 'bye' =
    end the session)."""

    def __init__(self, detector, input_fn=input, log=print):
        self.detector, self.input_fn, self.log = detector, input_fn, log
        self.model_calls = 0

    def observe(self, space, label, task_text):
        try:
            answer = self.input_fn(f"  Image at {label} (path, 'skip' or 'bye'): ").strip()
        except EOFError:
            answer = "bye"
        if answer.lower() == "bye":
            return Observation(space, aborted=True)
        if answer.lower() == "skip" or not answer:
            return Observation(space, skipped=True, description="no image (skipped)")
        self.model_calls += 1
        result = self.detector.detect(answer, task_text, label)
        if "error" in result:
            self.log(f"  DETECTION ERROR: {result['error']}")
            return Observation(space, skipped=True, error=result["error"])
        try:
            confidence = float(result.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        return Observation(space, bool(result.get("found")), confidence, str(result.get("description") or ""))


class StubBackend(PerceptionBackend):
    """For tests: the target is "seen" in the given spaces, missed elsewhere."""

    def __init__(self, found_in=(), confidence=1.0, skip=()):
        self.found_in, self.confidence, self.skip = set(found_in), confidence, set(skip)
        self.calls = []

    def observe(self, space, label, task_text):
        self.calls.append(space)
        if space in self.skip:
            return Observation(space, skipped=True)
        found = space in self.found_in
        return Observation(space, found, self.confidence if found else 0.0, "stub")


class NavigationBackend(ABC):
    @abstractmethod
    def execute(self, action):
        """Carry out one PrimitiveAction; False if it failed."""


class SimulatedNavigation(NavigationBackend):
    """Every action succeeds at once; the executed actions are kept."""

    def __init__(self):
        self.executed = []

    def execute(self, action):
        self.executed.append(action)
        return True
