from src.train import train_model


def main():

    model, jobs, users, test_users = train_model()

    user = test_users.iloc[100]

    recs = model.recommend(
        user_text=user["text"],
        user_id=user.get("UserID", None),
        top_k=10
    )

    print("\n===== TEST USER =====\n")
    print(user)

    print("\n===== TOP RECOMMENDATIONS =====\n")

    print(
        recs[[
            "JobID",
            "Title",
            "final_score",
            "content_score",
            "behavior_score",
            "popularity_score",
            "City"
        ]]
    )


if __name__ == "__main__":
    main()