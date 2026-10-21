"""Command-line version (no Streamlit needed).

Example:
    python cli.py samples/ai_ml.txt --exam 2026-12-15 --hours 8 --out my_plan.md
"""
import argparse
from datetime import date, datetime, timedelta

from planner.exporter import plan_to_csv, plan_to_markdown
from planner.extractor import extract_text
from planner.pipeline import analyze_syllabus, make_plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Syllabus -> week-by-week study plan")
    ap.add_argument("syllabus", help="path to a .pdf or .txt syllabus")
    ap.add_argument("--exam", required=True, help="exam date, YYYY-MM-DD")
    ap.add_argument("--hours", type=float, default=8, help="study hours available per week")
    ap.add_argument("--start", default=None, help="start date YYYY-MM-DD (default: today)")
    ap.add_argument("--group-size", type=int, default=4, help="approx. topics per study block")
    ap.add_argument("--revision-weeks", type=int, default=None)
    ap.add_argument("--embedder", choices=["auto", "sbert", "tfidf"], default="auto")
    ap.add_argument("--difficulty", choices=["heuristic", "classifier", "hybrid"], default="hybrid")
    ap.add_argument("--out", default=None, help="output file (.md or .csv); prints Markdown if omitted")
    args = ap.parse_args()

    exam = datetime.strptime(args.exam, "%Y-%m-%d").date()
    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else date.today()

    text = extract_text(args.syllabus)
    analysis = analyze_syllabus(text, args.group_size, args.embedder, args.difficulty)
    plan = make_plan(analysis, exam, args.hours, start, args.revision_weeks)

    output = plan_to_csv(plan) if (args.out or "").endswith(".csv") else plan_to_markdown(plan)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {args.out}  ({len(analysis.all_topics())} topics, {plan.total_weeks} weeks, "
              f"embeddings: {analysis.embedding_backend})")
    else:
        print(output)


if __name__ == "__main__":
    main()
