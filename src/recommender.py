from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class JobRecommender:

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        """
        Hybrid recommender:
        - alpha: content-based
        - beta: behavioral
        - gamma: popularity
        """

        self.vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

        self.jobs = None
        self.job_matrix = None

        self.user_job = None
        self.popularity = None

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def fit(self, jobs, user_job=None, popularity=None):

        self.jobs = jobs.copy()
        self.job_matrix = self.vectorizer.fit_transform(jobs["text"])

        self.user_job = user_job
        self.popularity = popularity

        return self


    def _behavior_score(self, job_ids, user_id):

        if self.user_job is None or user_id is None:
            return pd.Series([0] * len(job_ids))

        user_data = self.user_job[self.user_job["UserID"] == user_id]

        score_map = dict(zip(user_data["JobID"], user_data["weight"]))

        return pd.Series([score_map.get(j, 0) for j in job_ids])


    def _popularity_score(self, job_ids):

        if self.popularity is None:
            return pd.Series([0] * len(job_ids))

        pop_map = dict(zip(
            self.popularity["JobID"],
            self.popularity["popularity_score"]
        ))

        return pd.Series([pop_map.get(j, 0) for j in job_ids])


    def recommend(self, user_text, user_id=None, top_k=10, exclude_ids=None):

        exclude_ids = set(exclude_ids or [])

        # Content score
        user_vec = self.vectorizer.transform([user_text])
        content_scores = cosine_similarity(user_vec, self.job_matrix).ravel()

        results = self.jobs.copy()
        results["content_score"] = content_scores

        # Behavioral score
        results["behavior_score"] = self._behavior_score(
            results["JobID"].values,
            user_id
        )

        # Popularity score
        results["popularity_score"] = self._popularity_score(
            results["JobID"].values
        )

        # FINAL HYBRID SCORE
        results["final_score"] = (
            self.alpha * results["content_score"] +
            self.beta * results["behavior_score"] +
            self.gamma * results["popularity_score"]
        )

        # filter seen jobs
        if "JobID" in results.columns:
            results = results[~results["JobID"].isin(exclude_ids)]

        return results.sort_values("final_score", ascending=False).head(top_k)