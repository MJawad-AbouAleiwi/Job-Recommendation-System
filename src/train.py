import pandas as pd
from pathlib import Path

from src.features import build_job_text, build_user_text
from src.recommender import JobRecommender
from src.signal_builder import build_user_job_matrix, build_job_popularity


def train_model(data_dir="data/clean"):

    data_dir = Path(data_dir)

    jobs = pd.read_csv(data_dir / "jobs_cleaned.csv")
    users = pd.read_csv(data_dir / "users_cleaned.csv")
    history = pd.read_csv(data_dir / "history_cleaned.csv")
    feedbacks = pd.read_csv(data_dir / "feedbacks_cleaned.csv")
    test_users = pd.read_csv(data_dir / "test_users_cleaned.csv")

    # features
    jobs["text"] = build_job_text(jobs)
    users = build_user_text(users, history)

    test_users["text"] = (
        test_users["DegreeType"].fillna("") + " " +
        test_users["Major"].fillna("")
    ).str.lower()

    # signals
    user_job = build_user_job_matrix(feedbacks)
    popularity = build_job_popularity(feedbacks)

    # model
    model = JobRecommender()
    model.fit(jobs, user_job=user_job, popularity=popularity)

    return model, jobs, users, test_users