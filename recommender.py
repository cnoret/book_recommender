"""Recommendation logic shared by the app, train.py and evaluate_knn.py.

- `EditionIndex` tells when two titles are two editions of the same book.
- `similar_books` picks the KNN neighbours of a book, without its other editions.
- `ReaderRecommender` suggests books to a reader of the dataset.
"""
import html
import re
from difflib import SequenceMatcher

import numpy as np
from scipy.sparse import csr_matrix

# A rating of 8 or more means the reader liked the book.
LIKED_RATING = 8

# Subtitles that say nothing about the content: "A Novel", "A Memoir"...
_GENERIC_SUBTITLE = re.compile(r"^(a |an )?(novel|memoir|mystery|thriller|story|stories|new fiction)\b")


def _split_title(title):
    """Returns the main title and the subtitle, normalized for comparison."""
    text = html.unescape(str(title)).lower()
    parts = re.split(r"\s*[:(]\s*", text, maxsplit=1)
    main = re.sub(r"^(the|a|an)\s+", "", parts[0].strip())
    main = re.sub(r"[^a-z0-9]", "", main)
    subtitle = re.sub(r"[^a-z0-9 ]", "", parts[1]).strip() if len(parts) > 1 else ""
    subtitle = re.sub(r"\s+", " ", subtitle)
    if _GENERIC_SUBTITLE.match(subtitle):
        subtitle = ""
    return main, subtitle


class EditionIndex:
    """Finds the editions of the same book among titles that differ in their spelling.

    Two titles are editions of the same book when they share the author and the
    main title (before ":" or "("), and their subtitles do not contradict each
    other: one is missing or generic, one contains the other, or they are
    almost identical. "Alias Grace" and "Alias Grace : A Novel" match;
    "Dilbert: A Book of Postcards" and "Dilbert: Seven Years Of Highly
    Defective People" do not.
    """

    def __init__(self, titles, authors):
        self._entries = {}
        for title, author in zip(titles, authors):
            main, subtitle = _split_title(title)
            author_key = re.sub(r"[^a-z]", "", str(author).lower())
            self._entries[title] = (f"{main}|{author_key}", subtitle)

    def key(self, title):
        """Author and main title: two editions of the same book share it."""
        return self._entries[title][0] if title in self._entries else None

    def same_book(self, title_a, title_b):
        if title_a == title_b:
            return True
        if title_a not in self._entries or title_b not in self._entries:
            return False
        key_a, sub_a = self._entries[title_a]
        key_b, sub_b = self._entries[title_b]
        if key_a != key_b:
            return False
        if not sub_a or not sub_b or sub_a in sub_b or sub_b in sub_a:
            return True
        return SequenceMatcher(None, sub_a, sub_b).ratio() >= 0.8


def drop_editions(candidates, editions, exclude=(), n=10):
    """Keeps the first `n` candidates that are neither an edition of a title in
    `exclude` nor an edition of a candidate already kept."""
    kept = []
    by_key = {}
    for title in exclude:
        by_key.setdefault(editions.key(title), []).append(title)
    for title in candidates:
        same_key = by_key.get(editions.key(title), [])
        if any(editions.same_book(title, other) for other in same_key):
            continue
        kept.append(title)
        by_key.setdefault(editions.key(title), []).append(title)
        if len(kept) == n:
            break
    return kept


def similar_books(book_idx, knn_model, X, titles, editions, n=10, pool=40):
    """Titles of the `n` nearest neighbours of a book, other editions left out.

    `pool` neighbours are fetched, so that enough remain once the editions
    are removed.
    """
    pool = min(pool, X.shape[0])
    _, neighbours = knn_model.kneighbors(X[book_idx:book_idx + 1], n_neighbors=pool)
    candidates = [titles[i] for i in neighbours[0] if i != book_idx]
    return drop_editions(candidates, editions, exclude=[titles[book_idx]], n=n)


def _zscore(values, mask):
    """Standardizes `values` over the entries where `mask` is True."""
    selected = values[mask]
    std = selected.std() if selected.size else 0.0
    if not std:
        return np.zeros_like(values)
    return (values - selected.mean()) / std


class ReaderRecommender:
    """Suggests books to a reader from the books they liked.

    A candidate book gets two scores, each standardized over the catalog and
    then added up:

    - content: its cosine similarity to the liked books, on the KNN features
      (tags, description, author, cluster);
    - co-liked: how often readers who liked the same books also liked it
      (item-to-item collaborative filtering, cosine on "liked" ratings).

    When a reader liked nothing (no rating of 8 or more), all their rated
    books are used instead.
    """

    def __init__(self, ratings, X, titles, editions=None):
        self.X = csr_matrix(X)
        self.titles = list(titles)
        self.position = {title: i for i, title in enumerate(self.titles)}
        self.editions = editions

        ratings = ratings[ratings["Book-Title"].isin(self.position)]
        liked = ratings[ratings["Book-Rating"] >= LIKED_RATING]
        readers = {user: i for i, user in enumerate(liked["User-ID"].unique())}
        self.M = csr_matrix(
            (np.ones(len(liked)),
             (liked["User-ID"].map(readers), liked["Book-Title"].map(self.position))),
            shape=(len(readers), len(self.titles)),
        )
        self.M.data[:] = 1.0  # a reader who rated two editions counts once
        self.norm = np.sqrt(np.asarray(self.M.sum(axis=0)).ravel())

        self.rated = ratings.groupby("User-ID")["Book-Title"].apply(list).to_dict()
        self.liked = liked.groupby("User-ID")["Book-Title"].apply(list).to_dict()

    def knows(self, user):
        return user in self.rated

    def _profile(self, user):
        books = self.liked.get(user) or self.rated[user]
        return [self.position[b] for b in books]

    def _co_liked(self, idx):
        """Sum over the profile books i of cos(i, j), for every book j."""
        weights = np.zeros(len(self.titles))
        weights[idx] = np.divide(1.0, self.norm[idx], out=np.zeros(len(idx)), where=self.norm[idx] > 0)
        scores = self.M.T @ (self.M @ weights)
        return np.divide(scores, self.norm, out=np.zeros_like(scores), where=self.norm > 0)

    def scores(self, user):
        """Score of every book of the catalog for `user`; rated books get -inf."""
        idx = self._profile(user)
        content = np.asarray(self.X @ np.asarray(self.X[idx].sum(axis=0)).ravel()).ravel()
        co_liked = self._co_liked(idx)
        candidates = np.ones(len(self.titles), dtype=bool)
        candidates[[self.position[b] for b in self.rated[user]]] = False
        total = _zscore(content, candidates) + _zscore(co_liked, candidates)
        total[~candidates] = -np.inf
        return total

    def recommend(self, user, n=10):
        """The `n` best books for `user`, as (title, liked title it comes from) pairs."""
        scores = self.scores(user)
        order = np.argsort(-scores, kind="stable")
        order = [i for i in order if np.isfinite(scores[i])]
        if self.editions is not None:
            # Leaves out other editions of the books they rated, and repeats.
            chosen = drop_editions((self.titles[i] for i in order), self.editions,
                                   exclude=self.rated[user], n=n)
        else:
            chosen = [self.titles[i] for i in order[:n]]
        return [(title, self.because(user, title)) for title in chosen]

    def because(self, user, title):
        """The liked book that contributes the most to the score of `title`."""
        idx = self._profile(user)
        j = self.position[title]
        content = np.asarray((self.X[idx] @ self.X[j].T).todense()).ravel()
        co = np.asarray((self.M[:, idx].T @ self.M[:, j]).todense()).ravel()
        co = np.divide(co, self.norm[idx] * self.norm[j], out=np.zeros_like(co),
                       where=self.norm[idx] * self.norm[j] > 0)
        return self.titles[idx[int(np.argmax(content + co))]]
