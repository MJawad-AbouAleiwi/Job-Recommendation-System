import pandas as pd

EVENT_WEIGHTS = {
    "viewed": 1,
    "applied": 3,
    "hired": 5
}


def build_user_job_matrix(feedbacks: pd.DataFrame):
    """
    Create weighted user-job interactions.
    """

    feedbacks = feedbacks.copy()
    feedbacks["weight"] = feedbacks["Event"].map(EVENT_WEIGHTS).fillna(0)

    return feedbacks.groupby(["UserID", "JobID"])["weight"].sum().reset_index()


def build_job_popularity(feedbacks: pd.DataFrame):
    """
    Global job popularity score.
    """

    feedbacks = feedbacks.copy()
    feedbacks["weight"] = feedbacks["Event"].map(EVENT_WEIGHTS).fillna(0)

    return feedbacks.groupby("JobID")["weight"].sum().reset_index(
        name="popularity_score"
    )