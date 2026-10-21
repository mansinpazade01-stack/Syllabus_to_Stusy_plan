# 📚 Syllabus → Study Plan Generator

Upload a syllabus (PDF or text), enter the exam date and hours available per week, and get a
week-by-week study plan. Topics are grouped, tagged **Easy / Medium / Hard**, and harder topics get more time.

## Folder structure
```
syllabus_planner/
├── app.py                  # Streamlit UI (upload, edit difficulty, download)
├── cli.py                  # same pipeline from the command line
├── requirements.txt
├── planner/
│   ├── extractor.py        # 1. PDF/text extraction (pdfplumber, PyPDF2 fallback)
│   ├── parser.py           # 2. units + topics (regex headings, delimiter splitting)
│   ├── grouping.py         # 3. sentence embeddings + KMeans + TF-IDF keyword labels
│   ├── difficulty.py       # 4. heuristic / classifier / hybrid difficulty
│   ├── scheduler.py        # 5. weekly scheduling (difficulty-weighted)
│   ├── exporter.py         # CSV / Excel / Markdown
│   ├── pipeline.py         # glue code
│   └── models.py           # dataclasses
├── data/labeled_topics.csv # training examples for the difficulty classifier (add your own!)
├── samples/                # ai_ml.txt, dbms.txt
└── tests/test_pipeline.py
```

## Run
```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Command line: `python cli.py samples/ai_ml.txt --exam 2026-12-15 --hours 8 --out plan.md`
Tests: `pytest -q`

The first run with `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB).
Without it (or offline) the app automatically falls back to TF-IDF character n-grams.

## How it works
| Step | Technique |
|---|---|
| Extract | `pdfplumber` (fallback `PyPDF2`); page numbers and repeated headers removed |
| Parse | regex for `Unit I / Module 2 / UNIT-3`; hours like `(08 Hrs)` captured; "Course Outcomes", "Text Books" ignored; topics split on `, ; : •` and full stops, not inside brackets |
| Group | tokenization → sentence embeddings → KMeans inside each unit; each block named with its top TF-IDF keywords |
| Difficulty | topic length + technical-keyword lexicon + term rarity (IDF); optional logistic-regression classifier on `labeled_topics.csv`; `hybrid` averages both |
| Schedule | weights Easy=1, Medium=2, Hard=3; dynamic programming splits blocks into contiguous, balanced weeks; hours inside a week are proportional to weight; last 1-2 weeks are revision/mock tests |

## Tuning
* Wrong difficulty? Edit it in the app table, or add rows to `data/labeled_topics.csv`, or edit `HARD_KEYS` / `EASY_KEYS` in `difficulty.py`.
* Syllabus not parsed? Check the "Extracted text" expander. Scanned PDFs need OCR (`pytesseract`).
* Change effort per level in `models.py` (`DIFFICULTY_WEIGHT`) and `scheduler.py` (`HOURS_PER_WEIGHT`).

## Ideas to extend
KeyBERT for keywords, per-unit marks weightage, OCR support, `.ics` calendar export, a day-by-day view.
