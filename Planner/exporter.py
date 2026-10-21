"""Turn a StudyPlan into a table, CSV, Excel or Markdown."""
from __future__ import annotations

import io

import pandas as pd

from .models import StudyPlan

ICON = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}


def plan_to_dataframe(plan: StudyPlan) -> pd.DataFrame:
    rows = []
    for w in plan.weeks:
        for it in w.items:
            rows.append({
                "Week": w.week_no,
                "Dates": f"{w.start:%d %b} - {w.end:%d %b}",
                "Phase": w.kind.title(),
                "Unit": f"Unit {it.unit_no}: {it.unit_title}" if it.unit_no else "",
                "Study block": it.group_label,
                "Topic": it.topic,
                "Difficulty": it.difficulty,
                "Hours": it.hours,
            })
    return pd.DataFrame(rows)


def plan_to_csv(plan: StudyPlan) -> str:
    return plan_to_dataframe(plan).to_csv(index=False)


def plan_to_excel(plan: StudyPlan) -> bytes:
    buf = io.BytesIO()
    df = plan_to_dataframe(plan)
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Study plan", index=False)
        summary = df.groupby("Week", as_index=False)["Hours"].sum()
        summary.to_excel(xl, sheet_name="Hours per week", index=False)
        ws = xl.sheets["Study plan"]
        for col, width in zip("ABCDEFGH", (7, 16, 11, 34, 30, 52, 12, 8)):
            ws.column_dimensions[col].width = width
    return buf.getvalue()


def plan_to_markdown(plan: StudyPlan, course_name: str = "") -> str:
    out = [f"# Study plan{': ' + course_name if course_name else ''}", ""]
    out.append(f"- Start: **{plan.start_date:%d %b %Y}**  |  Exam: **{plan.exam_date:%d %b %Y}**  |  "
               f"Weeks: **{plan.total_weeks}**  |  Hours/week: **{plan.hours_per_week:g}**")
    out.append(f"- Recommended effort for one full pass: **{plan.required_hours:g} h** "
               f"(available in study weeks: {plan.available_hours:g} h)")
    for w in plan.warnings:
        out.append(f"- ⚠️ {w}")
    out.append("")
    for w in plan.weeks:
        out.append(f"## Week {w.week_no} ({w.start:%d %b} - {w.end:%d %b}): {w.title}")
        if w.notes:
            out.append(f"_{w.notes}_")
        out.append("")
        out.append("| Study block | Topic | Difficulty | Hours |")
        out.append("|---|---|---|---|")
        for it in w.items:
            out.append(f"| {it.group_label} | {it.topic} | {ICON.get(it.difficulty, '')} {it.difficulty} | {it.hours:g} |")
        out.append(f"\n**Total: {w.total_hours:g} h**\n")
    return "\n".join(out)
