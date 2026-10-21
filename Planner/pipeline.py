"""Glue: text -> units/topics -> groups -> difficulty -> weekly plan."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from .difficulty import estimate_difficulty
from .grouping import build_groups, embed_topics
from .models import StudyGroup, StudyPlan, Topic, Unit
from .parser import parse_syllabus
from .scheduler import build_plan


@dataclass
class Analysis:
    units: List[Unit]
    groups: List[StudyGroup]
    embedding_backend: str = ""

    def all_topics(self) -> List[Topic]:
        return [t for u in self.units for t in u.topics]

    def topics_dataframe(self) -> pd.DataFrame:
        group_label = {g_id: g.label for g_id, g in enumerate(self.groups)}
        return pd.DataFrame([{
            "Unit": f"Unit {t.unit_no}: {t.unit_title}",
            "Study block": group_label.get(t.group_id, ""),
            "Topic": t.name,
            "Difficulty": t.difficulty,
            "Score": t.score,
        } for t in self.all_topics()])


def analyze_syllabus(text: str, group_size: int = 4, embedder: str = "auto",
                     difficulty_method: str = "hybrid") -> Analysis:
    units = parse_syllabus(text)
    topics = [t for u in units for t in u.topics]
    if not topics:
        raise ValueError("Could not find any topics. Check that the syllabus has 'Unit I / Module 1' style headings.")
    estimate_difficulty(topics, difficulty_method)
    emb, backend = embed_topics([t.name for t in topics], embedder)
    groups = build_groups(units, emb, group_size=group_size)
    return Analysis(units, groups, backend)


def make_plan(analysis: Analysis, exam_date: date, hours_per_week: float,
              start_date: Optional[date] = None, revision_weeks: Optional[int] = None) -> StudyPlan:
    return build_plan(analysis.groups, exam_date, hours_per_week, start_date, revision_weeks)
