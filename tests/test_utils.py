import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

import utils


@pytest.fixture
def books_df():
    return pd.DataFrame({
        "User-ID": [1, 2, 1, 3],
        "Book-Title": ["Dune", "Dune", "Emma", "Ubik"],
        "Image-URL-L": ["http://img/dune.jpg", "http://img/dune.jpg", None, "not-a-url"],
        "Description": ["Spice.", "Spice.", "No description found", np.nan],
        "Bayes-Rating": [8.5, 8.5, 9.0, 7.0],
    })


def test_clean_description_hides_placeholders():
    assert utils.clean_description("A fine book.") == "A fine book."
    assert utils.clean_description("No description found") == ""
    assert utils.clean_description("Error: Failed after multiple retries") == ""
    assert utils.clean_description(np.nan) == ""


def test_clean_image_url_falls_back_to_default():
    assert utils.clean_image_url("http://img/a.jpg") == "http://img/a.jpg"
    assert utils.clean_image_url(None) == utils.DEFAULT_IMAGE
    assert utils.clean_image_url("relative.jpg") == utils.DEFAULT_IMAGE
    assert utils.DEFAULT_IMAGE.startswith("data:image/svg+xml")


def test_fetch_poster(books_df):
    names, posters, descriptions = utils.fetch_poster(books_df, ["Dune", "Emma", "Ubik", "Unknown"])
    assert names == ["Dune", "Emma", "Ubik", "Unknown"]
    assert posters == ["http://img/dune.jpg"] + [utils.DEFAULT_IMAGE] * 3
    assert descriptions == ["Spice.", "", "", ""]


def test_recommend_book_knn_excludes_the_query():
    # 12 books on a circle: the app asks for 11 neighbours.
    titles = [f"Book {i}" for i in range(12)]
    angles = np.linspace(0, np.pi / 2, 12)
    X = np.column_stack([np.cos(angles), np.sin(angles)])
    model = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    knn_df = pd.DataFrame({"Book-Title": titles})
    books_df = pd.DataFrame({
        "Book-Title": titles,
        "Image-URL-L": [f"http://img/{i}" for i in range(12)],
        "Description": ["text"] * 12,
    })

    names, posters, descriptions = utils.recommend_book_knn(books_df, "Book 0", knn_df, model, X)
    assert names[0] == "Book 1"
    assert "Book 0" not in names
    assert len(names) == len(posters) == len(descriptions) == 10


def test_recommend_book_knn_unknown_book_returns_three_lists(books_df):
    result = utils.recommend_book_knn(books_df, "Unknown", pd.DataFrame({"Book-Title": ["Dune"]}), None, None)
    assert result == ([], [], [])


def test_recommend_book_svd_guest_uses_bayes_rating(books_df):
    recommendations = utils.recommend_book_svd("Guest user", books_df, svd_model=None, n_recommendations=2)
    assert [title for title, *_ in recommendations] == ["Emma", "Dune"]


def test_render_aligned_image_escapes_title():
    out = utils.render_aligned_image("http://img/a.jpg", "Angels &amp; <Demons>")
    assert "Angels &amp; &lt;Demons&gt;" in out
    assert "<Demons>" not in out
