import pandas as pd

from train import add_weighted_ratings, ranking_metrics


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


def test_ranking_metrics():
    held_out = {1: {"A", "B"}, 2: {"C"}}
    tops = {1: ["A", "X", "Y"], 2: ["X", "Y", "Z"]}
    metrics = ranking_metrics(tops, held_out, k=3)
    assert metrics["Hit@3"] == 0.5          # reader 1 finds A, reader 2 finds nothing
    assert metrics["Recall@3"] == (1 / 2 + 0) / 2
    assert metrics["Distinct books"] == 4   # A, X, Y, Z
    assert metrics["Overlap"] == 2 / 3      # X and Y in both lists
