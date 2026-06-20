# Job Recommendation System

A job recommendation system that suggests relevant jobs based on:
- Job content (TF-IDF similarity)
- User profile (education + experience)
- User behavior (views, applications, hires)
- Job popularity

---

## Goal

Recommend the most relevant jobs to users automatically instead of manual searching.

---

## How it works

### 1. Job & User Representation
- Jobs → title + description + requirements + location  
- Users → degree + major + job history  

### 2. Content-Based Filtering
- TF-IDF converts text into vectors  
- Cosine similarity finds matching jobs  

### 3. Feedback Signals
User interactions are weighted:

- viewed = 1  
- applied = 3  
- hired = 5  

### 4. Popularity Signal
Jobs with more engagement get higher ranking.

### 5. Final Score

```bash
Final Score = α × content similarity + β × user behavior + γ × popularity
```

---

```bash
JOB-RECOMMENDER/
│
├── data/
│   ├── clean/                  # Cleaned datasets
│   └── raw/                    # Original datasets
│
├── notebooks/                  # Data exploration & cleaning
│   ├── 1_jobs_dataset.ipynb
│   ├── 2_users_dataset.ipynb
│   ├── 3_history_dataset.ipynb
│   ├── 4_feedbacks_dataset.ipynb
│   └── 5_test_users_dataset.ipynb
│
├── src/
│   ├── features.py             # Feature engineering
│   ├── signal_builder.py       # Feedback & popularity signals (NEW)
│   ├── recommender.py          # Hybrid recommender model
│   ├── train.py                # Training pipeline
│   └── predict.py              # Recommendation inference
│
├── demo_run.py                 # End-to-end demo
├── requirements.txt            # Dependencies
└── README.md
```

---

## How to run

```bash
pip install -r requirements.txt
python demo_run.py
```
---

## Limitations

* TF-IDF cannot fully understand meaning of words
* No deep learning model yet
* Cold start problem for new users

---

## Future Improvements

* Sentence-BERT embeddings
* Learning-to-rank model
* FastAPI deployment
* Streamlit UI
* Real-time feedback learning
