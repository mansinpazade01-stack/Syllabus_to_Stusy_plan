import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from planner.extractor import extract_text
from planner.parser import parse_syllabus, split_topics
from planner.pipeline import analyze_syllabus, make_plan
from planner.scheduler import partition_groups


def _text(name):
    return (ROOT / "samples" / name).read_text(encoding="utf-8")


def test_split_topics_keeps_brackets_together():
    assert split_topics("Types of learning (supervised, unsupervised), SVM; kNN") == [
        "Types of learning (supervised, unsupervised)", "SVM", "kNN"]


def test_parse_units_and_stop_sections():
    units = parse_syllabus(_text("ai_ml.txt"))
    assert [u.number for u in units] == [1, 2, 3, 4, 5, 6]
    assert units[0].hours_hint == 8
    all_names = " ".join(t.name for u in units for t in u.topics).lower()
    assert "tom mitchell" not in all_names and "co1" not in all_names


def test_bullets_and_roman_numerals():
    units = parse_syllabus(_text("dbms.txt"))
    assert len(units) == 6
    assert len(units[2].topics) >= 6


def test_partition_is_contiguous_and_balanced():
    ranges = partition_groups([3, 3, 3, 3, 3, 3], 3)
    assert ranges == [(0, 2), (2, 4), (4, 6)]


@pytest.mark.parametrize("weeks", [1, 2, 4, 10, 25])
def test_plan_has_one_entry_per_week(weeks):
    a = analyze_syllabus(_text("ai_ml.txt"), embedder="tfidf")
    plan = make_plan(a, date.today() + timedelta(weeks=weeks), 8)
    assert len(plan.weeks) == plan.total_weeks == weeks
    assert all(w.items for w in plan.weeks)


def test_harder_topics_get_more_hours_within_a_week():
    a = analyze_syllabus(_text("ai_ml.txt"), embedder="tfidf")
    plan = make_plan(a, date.today() + timedelta(weeks=8), 8)
    week = plan.weeks[1]
    hard = [i.hours for i in week.items if i.difficulty == "Hard"]
    easy = [i.hours for i in week.items if i.difficulty == "Easy"]
    if hard and easy:
        assert min(hard) > max(easy)


def test_bad_exam_date():
    a = analyze_syllabus(_text("dbms.txt"), embedder="tfidf")
    with pytest.raises(ValueError):
        make_plan(a, date.today(), 8)


def test_pdf_roundtrip(tmp_path):
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "s.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    y = 800
    for line in ["Unit I: Basics (6 Hrs)", "Variables, loops, functions", "Unit II: Data Structures", "Stacks, queues, trees"]:
        c.drawString(60, y, line)
        y -= 20
    c.save()
    units = parse_syllabus(extract_text(pdf))
    assert len(units) == 2 and units[1].topics[0].name == "Stacks"
