"""Step 5 - spread study groups over the weeks before the exam (harder = more time)."""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import List, Optional, Tuple

from .models import PlanItem, StudyGroup, StudyPlan, WeekPlan, Topic

HOURS_PER_WEIGHT = 1.0       # Easy = 1 h, Medium = 2 h, Hard = 3 h recommended for a first pass


def _round_quarter(x: float) -> float:
    return max(0.25, round(x * 4) / 4)


def partition_groups(weights: List[float], k: int) -> List[Tuple[int, int]]:
    """Split `weights` into k contiguous, non-empty ranges with balanced totals (DP, least squares)."""
    n = len(weights)
    k = max(1, min(k, n))
    prefix = [0.0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    mean = prefix[-1] / k
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for m in range(j - 1, i):
                if dp[j - 1][m] == inf:
                    continue
                cost = dp[j - 1][m] + (prefix[i] - prefix[m] - mean) ** 2
                if cost < dp[j][i]:
                    dp[j][i], cut[j][i] = cost, m
    ranges, i = [], n
    for j in range(k, 0, -1):
        m = cut[j][i]
        ranges.append((m, i))
        i = m
    return ranges[::-1]


def _allocate(items_src, budget: float) -> List[float]:
    """Share `budget` hours among topics in proportion to their weight."""
    total = sum(w for _, w in items_src) or 1.0
    return [_round_quarter(budget * w / total) for _, w in items_src]


def _fmt(d: date) -> str:
    return d.strftime("%d %b")


def build_plan(
    groups: List[StudyGroup],
    exam_date: date,
    hours_per_week: float,
    start_date: Optional[date] = None,
    revision_weeks: Optional[int] = None,
) -> StudyPlan:
    start = start_date or date.today()
    if exam_date <= start:
        raise ValueError("Exam date must be after the start date.")
    if hours_per_week <= 0:
        raise ValueError("Hours per week must be positive.")
    if not groups:
        raise ValueError("No topics found in the syllabus.")

    total_weeks = max(1, math.ceil((exam_date - start).days / 7))
    if revision_weeks is None:
        revision_weeks = 0 if total_weeks < 3 else 1 if total_weeks < 8 else 2
    revision_weeks = max(0, min(revision_weeks, total_weeks - 1))
    study_weeks = total_weeks - revision_weeks
    used_study = min(study_weeks, len(groups))
    buffer_weeks = study_weeks - used_study

    ranges = partition_groups([g.weight for g in groups], used_study)
    all_topics: List[Topic] = [t for g in groups for t in g.topics]
    required = sum(t.weight for t in all_topics) * HOURS_PER_WEIGHT
    available = used_study * hours_per_week
    warnings: List[str] = []
    if required > available:
        warnings.append(
            f"The syllabus needs about {required:.0f} h for a first pass, but only {available:.0f} h are "
            f"available in the study weeks. Increase hours/week, start earlier, or accept lighter coverage of Easy topics."
        )

    def week_bounds(i: int) -> Tuple[date, date]:
        s = start + timedelta(days=7 * i)
        e = min(s + timedelta(days=6), exam_date - timedelta(days=1))
        return s, max(s, e)

    weeks: List[WeekPlan] = []
    # ---- study weeks
    for idx, (a, b) in enumerate(ranges):
        chunk = groups[a:b]
        pairs = [(g, t) for g in chunk for t in g.topics]
        hours = _allocate([(t, t.weight) for _, t in pairs], hours_per_week)
        items = [PlanItem(g.unit_no, g.unit_title, g.label, t.name, t.difficulty, h) for (g, t), h in zip(pairs, hours)]
        s, e = week_bounds(idx)
        units_in = sorted({g.unit_no for g in chunk})
        title = "Unit " + ", ".join(map(str, units_in)) + " - " + "; ".join(dict.fromkeys(g.unit_title for g in chunk))
        need = sum(t.weight for _, t in pairs) * HOURS_PER_WEIGHT
        note = f"Heavy week: ~{need:.0f} h recommended vs {hours_per_week:.0f} h available." if need > hours_per_week * 1.15 else ""
        weeks.append(WeekPlan(idx + 1, s, e, "study", title, items, note))

    # ---- buffer weeks (when the syllabus is small compared with the time available)
    for j in range(buffer_weeks):
        i = used_study + j
        s, e = week_bounds(i)
        item = PlanItem(0, "", "Practice", "Solve problems and previous-year question papers", "-", hours_per_week)
        weeks.append(WeekPlan(i + 1, s, e, "buffer", "Buffer & practice", [item],
                              "Extra time: catch up on weak topics or start early revision."))

    # ---- revision weeks (one row per study block, so the list stays short)
    rank = {"Easy": 1, "Medium": 2, "Hard": 3}
    for r in range(revision_weeks):
        i = study_weeks + r
        s, e = week_bounds(i)
        last = r == revision_weeks - 1
        wanted = ("Hard",) if last else ("Hard", "Medium")
        title = "Final revision & mock tests" if last else "Revision (Hard and Medium topics)"
        note = ("Attempt full mock papers under exam conditions, then revise the mistakes." if last
                else "Revise notes, formulas and solved examples.")
        rows = []                                   # (group, topics to revise, weight)
        for g in groups:
            picked = [t for t in g.topics if t.difficulty in wanted]
            if picked:
                rows.append((g, picked, sum(t.weight for t in picked)))
        if not rows:                                # nothing Hard/Medium -> revise everything
            rows = [(g, g.topics, g.weight) for g in groups]
        rows.sort(key=lambda x: -x[2])
        weights = [w for _, _, w in rows]
        if last:
            weights.append(max(2.0, 0.4 * sum(weights)))     # mock tests get ~30% of the final week
        hrs = _allocate([(None, w) for w in weights], hours_per_week)
        items = []
        for (g, picked, _), h in zip(rows, hrs):
            names = ", ".join(t.name for t in picked)
            names = names if len(names) <= 150 else names[:147].rstrip() + "..."
            level = max((t.difficulty for t in picked), key=lambda d: rank.get(d, 2))
            items.append(PlanItem(g.unit_no, g.unit_title, g.label, names, level, h))
        if last:
            items.append(PlanItem(0, "", "Mock test", "Full-length mock test(s) under timed conditions", "-", hrs[-1]))
        weeks.append(WeekPlan(i + 1, s, e, "revision", title, items, note))

    return StudyPlan(weeks, start, exam_date, hours_per_week, total_weeks, round(required, 1), round(available, 1), warnings)
