def get_user_recommendations(model, users, user_id, jobs, top_k=10):

    user = users[users["UserID"] == user_id]

    if user.empty:
        # fallback: popular jobs
        return jobs.head(top_k)

    return model.recommend(
        user_text=user.iloc[0]["text"],
        user_id=user_id,
        top_k=top_k
    )