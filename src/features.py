import pandas as pd


def build_job_text(jobs: pd.DataFrame) -> pd.Series:
    """
    Build a unified text representation for each job.
    """

    text = (
        jobs["Title"].fillna("") + " " +
        jobs["Description"].fillna("") + " " +
        jobs["Requirements"].fillna("") + " " +
        jobs["City"].fillna("") + " " +
        jobs["Country"].fillna("")
    )

    return text.str.lower()


def build_user_text(users: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Build user text profile using:
    - education
    - job history
    """

    hist = history.sort_values(["UserID", "Sequence"])

    hist_text = hist.groupby("UserID")["JobTitle"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index(name="history")

    users = users.merge(hist_text, on="UserID", how="left")
    users["history"] = users["history"].fillna("")

    users["text"] = (
        users["DegreeType"].fillna("") + " " +
        users["Major"].fillna("") + " " +
        users["history"]
    ).str.lower()

    return users