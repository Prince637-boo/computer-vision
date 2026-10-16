from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class TrainingPhase(str, Enum):
    REMOVING_PARTS = "removing_parts"
    QUIZ = "quiz"
    COMPLETE = "complete"


@dataclass(frozen=True)
class TrainingStep:
    match: str
    label: str


class TrainingSession:
    """Panda3D-independent state for the disassembly lesson and its quiz."""

    def __init__(
        self, steps: Sequence[TrainingStep], answer_options: Sequence[str]
    ):
        if not steps:
            raise ValueError("A training session needs at least one step.")

        self.steps = tuple(steps)
        self.answer_options = tuple(dict.fromkeys(answer_options))
        if any(step.label not in self.answer_options for step in self.steps):
            raise ValueError("Every training label must be a quiz answer option.")

        self.phase = TrainingPhase.REMOVING_PARTS
        self.step_index = 0
        self.quiz_index = 0
        self.score = 0

    @property
    def current_step(self) -> Optional[TrainingStep]:
        if self.phase != TrainingPhase.REMOVING_PARTS:
            return None
        return self.steps[self.step_index]

    @property
    def current_question(self) -> Optional[str]:
        if self.phase != TrainingPhase.QUIZ:
            return None
        return self.steps[self.quiz_index].label

    @property
    def question_count(self) -> int:
        return len(self.steps)

    @property
    def quiz_choices(self) -> Tuple[str, ...]:
        if self.current_question is None:
            return ()
        answer = self.current_question
        choices = [answer]
        choices.extend(
            option
            for option in self.answer_options
            if option != answer and option not in choices
        )
        choices = choices[:4]
        offset = self.quiz_index % len(choices)
        return tuple(choices[offset:] + choices[:offset])

    def select_part(self, node_name: str) -> bool:
        """Accept a pinched mesh only if it matches the current lesson step."""
        step = self.current_step
        if step is None or step.match.casefold() not in node_name.casefold():
            return False

        self.step_index += 1
        if self.step_index == len(self.steps):
            self.phase = TrainingPhase.QUIZ
            self.quiz_index = 0
        return True

    def answer_quiz(self, answer: str) -> bool:
        """Score one answer, advance the quiz, and return whether it was correct."""
        if self.phase != TrainingPhase.QUIZ:
            raise RuntimeError("There is no active quiz question.")
        if answer not in self.quiz_choices:
            raise ValueError(f"{answer!r} is not an available quiz answer.")

        correct = answer == self.current_question
        self.score += int(correct)
        self.quiz_index += 1
        if self.quiz_index == len(self.steps):
            self.phase = TrainingPhase.COMPLETE
        return correct
