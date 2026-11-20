from collections import namedtuple

import pandas as pd

from train import add_weighted_ratings, precision_recall_at_k

Prediction = namedtuple("Prediction", "uid iid r_ui est details")


def test_bayes_rating_pulls_rare_books_towards_the_mean():
    data = pd.DataFrame({
        "Book-Title": ["Rare"] + ["Classic"] * 20 + ["Average"] * 30,
        "Book-Rating": [10] + [9] * 20 + [6] * 30,
    })
    out = add_weighted_ratings(data, min_votes=5)
    stats = out.drop_duplicates("Book-Title").set_index("Book-Title")
    assert stats.loc["Rare", "Avg-Rating"] == 10
    assert stats.loc["Classic", "Num-Ratings"] == 20
    # A single 10 no longer beats twenty 9s.
    assert stats.loc["Classic", "Bayes-Rating"] > stats.loc["Rare", "Bayes-Rating"]


def test_precision_recall_at_k():
    # One user, 10 held-out ratings, 4 relevant (>= 8), the model ranks 3 of them in its top 5.
    true = [9, 9, 9, 2, 2, 8, 1, 1, 1, 1]
    est = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    preds = [Prediction(1, i, t, e, {}) for i, (t, e) in enumerate(zip(true, est))]
    precision, recall, n_users = precision_recall_at_k(preds, k=5, threshold=8, min_items=10)
    assert n_users == 1
    assert precision == 3 / 5
    assert recall == 3 / 4


def test_precision_recall_skips_users_with_few_ratings():
    preds = [Prediction(1, i, 9, 9, {}) for i in range(3)]
    assert precision_recall_at_k(preds, min_items=10)[2] == 0
