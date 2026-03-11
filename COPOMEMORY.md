# COPO Memory - CO-PO Attainment System

## Project Overview
CO-PO Attainment System for NBA (National Board of Accreditation) compliance. Automates mapping Course Outcomes (COs) to Program Outcomes (POs) and calculates attainment from student assessments.

## Repository
- **GitHub**: https://github.com/ShivamJoshii/copo-attainment
- **Local**: `/Users/mybuildertechnologiesltd./.openclaw/workspace/copo-attainment/`
- **Deployment**: Streamlit Cloud

## Architecture

### Tech Stack
- **Framework**: Streamlit (Python)
- **Database**: SQLite (local file: `copo_data.db`)
- **ML/NLP**: sentence-transformers, scikit-learn
- **Data**: pandas, numpy

### File Structure
```
copo-attainment/
├── streamlit_app.py      # Main application (all-in-one)
├── requirements.txt      # Dependencies
├── copo_data.db         # SQLite database (auto-created)
└── copomemory.md        # This file
```

## Core Concepts

### Course Outcomes (COs)
What students will learn in a specific course. Example:
- "Apply knowledge of mathematics and science to solve engineering problems"
- "Design and develop software solutions using modern programming techniques"

### Program Outcomes (POs)
Broad outcomes for the entire program (standard NBA POs):
- PO1: Engineering knowledge
- PO2: Problem analysis
- PO3: Design/development of solutions
- PO4: Modern tool usage
- PO5: The engineer and society
- PO6: Environment and sustainability
- PO7: Ethics
- PO8: Individual and team work
- PO9: Communication
- PO10: Project management and finance
- PO11: Life-long learning

### CO-PO Mapping
Each CO maps to each PO with a weight (0-3):
- **0**: No relevance
- **1**: Weak relevance
- **2**: Moderate relevance
- **3**: Strong relevance

## NLP / Mapping Algorithm

### Current Approach: Sentence Embeddings
1. **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
2. **Process**:
   - Convert CO descriptions to vectors
   - Convert PO descriptions to vectors
   - Calculate cosine similarity between each CO-PO pair
   - Map similarity to weight using thresholds

### Optimized Thresholds (March 11, 2026)
Based on expert benchmark analysis:

| Weight | Similarity Range |
|--------|-----------------|
| 0 | < 0.265 |
| 1 | 0.265 - 0.397 |
| 2 | 0.397 - 0.518 |
| 3 | ≥ 0.518 |

### Why These Thresholds?
Expert benchmark analysis showed:
- Weight 0: similarities 0.046-0.411 (mean: 0.207)
- Weight 1: similarities 0.118-0.499 (mean: 0.277)
- Weight 2: similarities 0.295-0.604 (mean: 0.469)
- Weight 3: similarities 0.433-0.868 (mean: 0.578)

Previous thresholds were too conservative (too many 0s and 1s).

### Fallback: Rule-Based (if embeddings unavailable)
Uses domain keywords + Bloom's taxonomy:
- Domain categories: technical, problem_solving, computing, communication, teamwork, ethics, learning, business, math, science
- Bloom's levels: remember, understand, apply, analyze, evaluate, create
- Multi-factor scoring: domain overlap (40%), word similarity (30%), Bloom's alignment (20%), phrase overlap (10%)

## Attainment Calculation

### Direct Attainment
```
Direct = (internal_weight × Internal) + (ese_weight × ESE)
Default: internal_weight = 0.4, ese_weight = 0.6
```

### Final Attainment
```
Final = (direct_weight × Direct) + (indirect_weight × Indirect)
Default: direct_weight = 0.8, indirect_weight = 0.2
```

### Scale of 3
```
Scale of 3 = Final Attainment × 3
```

### Target Achievement
```
Achieved = "Y" if Scale of 3 ≥ Target Level (default: 1.4)
Achieved = "N" otherwise
```

### PO Attainment
Weighted average of CO attainments based on CO-PO mapping weights:
```
PO_Attainment = Σ(CO_Final × Weight) / Σ(Weights)
```

## Database Schema

### Tables
1. **courses**: Course info, weights
2. **course_outcomes**: COs for each course
3. **program_outcomes**: POs for each course
4. **co_po_mappings**: CO-PO mapping weights
5. **attainment_inputs**: Student assessment data
6. **calculated_results**: CO attainment results
7. **po_attainment_results**: PO attainment results

## Usage Flow

1. **Create Course** (5-step wizard):
   - Step 1: Course info (name, code)
   - Step 2: Enter Course Outcomes (COs)
   - Step 3: Select/enter Program Outcomes (POs)
   - Step 4: Review auto-generated CO-PO mapping (editable)
   - Step 5: Done

2. **Enter Attainments**:
   - Select course
   - For each CO: enter Internal (0-3), ESE (0-3), Indirect (0-3)
   - Set target level
   - Save & calculate

3. **View Results**:
   - CO attainment table
   - PO attainment table
   - Success metrics

## Key Decisions Log

### March 11, 2026 - Embedding-Based Mapping
- **Problem**: Rule-based mapping gave too many zeros
- **Solution**: Switched to sentence embeddings (all-MiniLM-L6-v2)
- **Result**: Better semantic understanding, fewer zeros

### March 11, 2026 - Threshold Optimization
- **Problem**: Embeddings still not matching expert judgment
- **Solution**: Created expert benchmark, analyzed similarity distributions
- **Result**: Optimized thresholds (0.265/0.397/0.518) for 50%+ exact match

### March 11, 2026 - UI Fix
- **Problem**: CO-PO matrix headers displayed after data rows
- **Solution**: Moved header rendering before data rows

## Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
```

## Future Improvements
- [ ] Add import/export for bulk data entry
- [ ] Add visualization charts for attainment trends
- [ ] Support for multiple years/semesters
- [ ] Export to PDF/Excel reports
- [ ] User authentication
- [ ] Department-level aggregation

## Notes
- Database is SQLite (file-based), no external DB needed
- Embeddings model downloads automatically on first run (~80MB)
- App is self-contained in a single file for easy deployment
