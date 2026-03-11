# CO-PO Attainment System

Streamlit app for calculating Course Outcome (CO) and Program Outcome (PO) attainment from student assessment data.

## Features

- Upload Internal Assessment (IA) and End Semester Exam (ESE) CSV files
- Automatic CO attainment calculation (% of students ≥ 50% marks)
- Direct attainment = weighted average of IA and ESE
- CO-PO mapping with customizable mappings
- Final PO and PSO attainment calculations
- Downloadable reports

## Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=ShivamJoshii/copo-attainment)

## Local Development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage

1. Enter course details and CO/PO statements
2. Upload Internal Assessment scores (CSV)
3. Upload ESE scores (CSV)
4. Configure CO-PO mapping
5. View attainment calculations and download reports
