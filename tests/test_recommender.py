import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recommender import EditionIndex, ReaderRecommender, drop_editions, similar_books


def test_editions_of_the_same_book():
    titles = ["Alias Grace", "Alias Grace : A Novel", "The Accidental Tourist", "Accidental Tourist",
              "Dilbert: A Book of Postcards", "Dilbert: Seven Years Of Highly Defective People",
              "Griffin &amp; Sabine", "Griffin & Sabine: An Extraordinary Correspondence"]
    authors = ["Margaret Atwood", "MARGARET ATWOOD", "Anne Tyler", "Anne Tyler",
               "Scott Adams", "Scott Adams", "Nick Bantock", "Nick Bantock"]
    editions = EditionIndex(titles, authors)
    assert editions.same_book("Alias Grace", "Alias Grace : A Novel")
    assert editions.same_book("The Accidental Tourist", "Accidental Tourist")
    assert editions.same_book("Griffin &amp; Sabine", "Griffin & Sabine: An Extraordinary Correspondence")
    # Same author and main title, but two different books.
    assert not editions.same_book("Dilbert: A Book of Postcards",
                                  "Dilbert: Seven Years Of Highly Defective People")
    assert not editions.same_book("Alias Grace", "Accidental Tourist")


def test_same_title_by_another_author_is_another_book():
    titles = ["Silent Night : A Christmas Suspense Story", "SILENT NIGHT"]
    editions = EditionIndex(titles, ["Mary Higgins Clark", "Stanley Weintraub"])
    assert not editions.same_book(*titles)


def test_drop_editions_keeps_the_first_of_each_book():
    titles = ["Emma", "Emma: A Novel", "Dune", "Dune (Book 1)", "Ubik"]
    editions = EditionIndex(titles, ["Austen", "Austen", "Herbert", "Herbert", "Dick"])
    assert drop_editions(["Emma: A Novel", "Dune", "Dune (Book 1)", "Ubik"], editions,
                         exclude=["Emma"]) == ["Dune", "Ubik"]
    assert drop_editions(titles, editions, n=2) == ["Emma", "Dune"]


def test_similar_books_skips_other_editions():
    titles = ["Dune", "Dune: A Novel", "Ubik", "Emma", "Solaris"]
    X = np.array([[1, 0], [1, 0.01], [1, 0.2], [0.1, 1], [1, 0.4]])
    model = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    editions = EditionIndex(titles, ["Herbert", "Herbert", "Dick", "Austen", "Lem"])
    assert similar_books(0, model, X, titles, editions, n=3) == ["Ubik", "Solaris", "Emma"]


def _toy_recommender():
    # Two readers love the same two fantasy books; a third reader likes cooking.
    titles = ["Hobbit", "Narnia", "Earthsea", "Cookbook", "Baking"]
    X = csr_matrix(np.array([[1, 0], [1, 0], [1, 0.1], [0, 1], [0, 1]], dtype=float))
    ratings = pd.DataFrame({
        "User-ID": [1, 1, 2, 2, 2, 3, 3],
        "Book-Title": ["Hobbit", "Cookbook", "Hobbit", "Narnia", "Earthsea", "Cookbook", "Baking"],
        "Book-Rating": [9, 3, 10, 9, 8, 9, 9],
    })
    return ReaderRecommender(ratings, X, titles)


def test_reader_recommender_follows_what_the_reader_liked():
    model = _toy_recommender()
    picks = model.recommend(1, n=2)
    # Reader 1 loved the Hobbit and disliked the Cookbook: fantasy comes first.
    assert [title for title, _ in picks] == ["Narnia", "Earthsea"]
    assert all(source == "Hobbit" for _, source in picks)


def test_reader_recommender_never_suggests_rated_books():
    model = _toy_recommender()
    assert "Hobbit" not in [title for title, _ in model.recommend(2, n=5)]
    assert model.knows(3) and not model.knows(42)
