"""Data classes shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

DIFFICULTY_LEVELS = ("Easy", "Medium", "Hard")

# How many "effort points" each difficulty level is worth.
# Harder topics get proportionally more study time.
DIFFICULTY_WEIGHT = {"Easy": 1.0, "Medium": 2.0, "Hard": 3.0}


@dataclass
class Topic:
    name: str
    unit_no: int
    unit_title: str
    difficulty: str = "Medium"
    score: float = 0.5          # 0 (easy) .. 1 (hard)
    group_id: int = -1

    @property
    def weight(self) -> float:
        return DIFFICULTY_WEIGHT.get(self.difficulty, 2.0)


@dataclass
class Unit:
    number: int
    title: str
    topics: List[Topic] = field(default_factory=list)
    hours_hint: Optional[float] = None   # lecture hours written in the syllabus, if any


@dataclass
class StudyGroup:
    """A cluster of related topics that are studied together."""
    unit_no: int
    unit_title: str
    label: str
    topics: List[Topic]

    @property
    def weight(self) -> float:
        return sum(t.weight for t in self.topics)


@dataclass
class PlanItem:
    unit_no: int
    unit_title: str
    group_label: str
    topic: str
    difficulty: str
    hours: float


@dataclass
class WeekPlan:
    week_no: int
    start: date
    end: date
    kind: str                    # "study" | "buffer" | "revision"
    title: str
    items: List[PlanItem] = field(default_factory=list)
    notes: str = ""

    @property
    def total_hours(self) -> float:
        return round(sum(i.hours for i in self.items), 2)


@dataclass
class StudyPlan:
    weeks: List[WeekPlan]
    start_date: date
    exam_date: date
    hours_per_week: float
    total_weeks: int
    required_hours: float        # recommended hours to cover the whole syllabus once
    available_hours: float       # hours available in the study weeks
    warnings: List[str] = field(default_factory=list)
