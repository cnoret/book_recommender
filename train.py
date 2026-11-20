""" Trains the book recommender: a content-based KNN model and an SVD model. """

import itertools
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, normalize
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

from recommender import LIKED_RATING, EditionIndex, ReaderRecommender


def load_data(file_path):
    """Loads the dataset from a CSV file."""
    print("Step 1: Loading data...")
    data = pd.read_csv(file_path)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
    return data


def data_overview(data):
    """Prints an overview of the data."""
    print("\nStep 2: Data overview...")
    print(data.head())
    print("\nData information:")
    print(data.info())


def create_user_item_matrix(data):
    """Creates a sparse user-item matrix."""
    print("\nStep 3: Creating user-item matrix...")
    users = data["User-ID"].astype("category").cat.codes
    books = data["Book-Title"].astype("category").cat.codes
    ratings = data["Book-Rating"]
    sparse_matrix = coo_matrix((ratings, (users, books)))
    print(f"User-item matrix created with dimensions: {sparse_matrix.shape}.")
    return sparse_matrix, data["Book-Title"].astype("category").cat.categories

def preprocess_data_for_knn(data):
    """Groups the dataset by ISBN: one row per book for the KNN model."""
    print("\nStep 1.1: Preprocessing data for KNN...")
    data_grouped = data.groupby("ISBN").agg({
        "Book-Title": "first",
        "Book-Author": "first",
        "Final_Tags": "first",
        "Tags": "first",
        "Description": "first",
        "Book-Rating": "mean",
        "Cluster_hdbscan": "first",
    }).reset_index()
    return data_grouped


def add_weighted_ratings(data, min_votes=None):
    """Adds per-book rating statistics computed from the ratings themselves.

    - Avg-Rating : mean of the ratings of the book
    - Num-Ratings : number of ratings of the book in the dataset
    - Bayes-Rating : Bayesian average, which pulls books with few ratings
      towards the global mean so that a single 10 does not top the charts.
    """
    print("\nStep 1.2: Computing weighted ratings...")
    stats = data.groupby("Book-Title")["Book-Rating"].agg(["mean", "count"])
    global_mean = data["Book-Rating"].mean()
    if min_votes is None:
        min_votes = stats["count"].median()
    bayes = (stats["count"] * stats["mean"] + min_votes * global_mean) / (stats["count"] + min_votes)
    data = data.drop(columns=["Avg-Rating", "Num-Ratings", "Bayes-Rating"], errors="ignore")
    data["Avg-Rating"] = data["Book-Title"].map(stats["mean"]).round(2)
    data["Num-Ratings"] = data["Book-Title"].map(stats["count"])
    data["Bayes-Rating"] = data["Book-Title"].map(bayes).round(3)
    print(f"Bayesian average computed (prior: {min_votes:.0f} votes at {global_mean:.2f}).")
    return data


def _clean_text(series, placeholders=()):
    """Replaces missing values and scraping placeholders by an empty string."""
    series = series.fillna("").astype(str)
    return series.where(~series.isin(placeholders), "")


# Relative weight of each block of features in the final similarity.
FEATURE_WEIGHTS = {
    "tags": 1.0,         # Open Library subjects + final tags
    "description": 0.7,  # scraped description
    "author": 0.8,       # exact author
    "cluster": 0.3,      # HDBSCAN cluster, one-hot encoded
}


def train_knn_model_with_metadata(data):
    """Trains a content-based KNN model on tags, descriptions, author and cluster.

    Each block is vectorized and L2-normalized separately, then weighted, so
    that no block dominates because of its scale. The cluster is one-hot
    encoded: its number carries no order. The rating is left out on purpose:
    similar books are books that talk about the same things, not books that
    got the same mark.
    """
    print("\nStep 4: Training the content-based KNN model...")

    tags_text = (
        _clean_text(data["Final_Tags"]) + ", "
        + _clean_text(data["Tags"], placeholders=("No themes available",))
    )
    tfidf_tags = TfidfVectorizer(
        tokenizer=lambda s: [t.strip().lower() for t in s.split(",") if t.strip()],
        token_pattern=None, lowercase=False, min_df=2, sublinear_tf=True,
    )
    X_tags = tfidf_tags.fit_transform(tags_text)

    descriptions = _clean_text(
        data["Description"],
        placeholders=("No description found", "No description available",
                      "Error: Failed after multiple retries"),
    )
    tfidf_desc = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                                 max_features=20000, sublinear_tf=True)
    X_desc = tfidf_desc.fit_transform(descriptions)

    authors = data["Book-Author"].fillna("").str.lower().str.strip()
    X_author = OneHotEncoder(handle_unknown="ignore").fit_transform(authors.to_frame())

    X_cluster = OneHotEncoder(handle_unknown="ignore").fit_transform(
        data[["Cluster_hdbscan"]].astype(str)
    )

    blocks = {"tags": X_tags, "description": X_desc,
              "author": X_author, "cluster": X_cluster}
    X_final = hstack([
        normalize(block, norm="l2") * FEATURE_WEIGHTS[name]
        for name, block in blocks.items()
    ]).tocsr()
    X_final = normalize(X_final, norm="l2")
    print(f"Feature matrix created with shape: {X_final.shape} "
          f"(tags {X_tags.shape[1]}, description {X_desc.shape[1]}, "
          f"authors {X_author.shape[1]}, clusters {X_cluster.shape[1]}).")

    knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
    knn_model.fit(X_final)
    print("Content-based KNN model trained successfully.")
    return knn_model, X_final


# Columns of book_df_knn kept in the artifact (the long texts stay in book_df).
KNN_COLUMNS = ["ISBN", "Book-Title", "Book-Author", "Final_Tags", "Book-Rating", "Cluster_hdbscan"]


def ranking_metrics(top_lists, held_out, k=10):
    """Hit rate, recall and diversity of top-k lists.

    - hit@k: share of readers with at least one held-out liked book in their list
    - recall@k: mean share of their held-out liked books found in their list
    - distinct books: number of different books over all the lists
    - overlap: mean share of books two readers' lists have in common
    """
    hits = [len(set(top[:k]) & held_out[u]) for u, top in top_lists.items()]
    recalls = [h / len(held_out[u]) for h, u in zip(hits, top_lists)]
    tops = [set(top[:k]) for top in top_lists.values()]
    pairs = list(itertools.combinations(tops[:150], 2))
    return {
        f"Hit@{k}": float(np.mean([h > 0 for h in hits])),
        f"Recall@{k}": float(np.mean(recalls)),
        "Distinct books": len(set().union(*tops)),
        "Overlap": float(np.mean([len(a & b) / k for a, b in pairs])) if pairs else np.nan,
    }


def evaluate_reader_models(data, X, knn_titles, editions, n_readers=400, k=10):
    """Compares the "books for a reader" models on a 20 % held-out split.

    Each model ranks every book the reader has not rated in the training part,
    as the app does, and is scored on the held-out books the reader liked
    (rating of 8 or more).
    """
    print("\nStep 5: Evaluating the books for a reader...")
    reader = Reader(rating_scale=(data["Book-Rating"].min(), data["Book-Rating"].max()))
    dataset = Dataset.load_from_df(data[["User-ID", "Book-Title", "Book-Rating"]], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    train = pd.DataFrame(
        [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), r) for u, i, r in trainset.all_ratings()],
        columns=["User-ID", "Book-Title", "Book-Rating"],
    )

    held_out = {}
    for user, title, rating in testset:
        if rating >= LIKED_RATING:
            held_out.setdefault(user, set()).add(title)
    rated = train.groupby("User-ID")["Book-Title"].apply(set)
    readers = sorted(u for u in held_out if u in rated.index)
    rng = np.random.default_rng(0)
    readers = list(rng.choice(readers, size=min(n_readers, len(readers)), replace=False))

    stats = train.groupby("Book-Title")["Book-Rating"].agg(["mean", "count"])
    prior = stats["count"].median()
    bayes = (stats["count"] * stats["mean"] + prior * train["Book-Rating"].mean()) / (stats["count"] + prior)

    def top_of(scores):
        def top(user):
            ranked = scores(user).sort_values(ascending=False, kind="stable").index
            return [t for t in ranked if t not in rated[user]][:k]
        return top

    svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    items = stats.index
    inner = [trainset.to_inner_iid(t) for t in items]
    def svd_scores(user):
        u = trainset.to_inner_uid(user)
        est = trainset.global_mean + svd.bu[u] + svd.bi[inner] + svd.qi[inner] @ svd.pu[u]
        return pd.Series(est, index=items)

    hybrid = ReaderRecommender(train, X, knn_titles, editions)
    models = {
        "Most read": top_of(lambda u: stats["count"]),
        "Bayesian average": top_of(lambda u: bayes),
        "SVD (predicted rating)": top_of(svd_scores),
        "Hybrid (content + co-liked)": lambda u: [t for t, _ in hybrid.recommend(u, n=k)],
    }
    rows = []
    for name, top in models.items():
        metrics = ranking_metrics({u: top(u) for u in readers}, held_out, k=k)
        rows.append({"Model": name, **metrics})
    table = pd.DataFrame(rows)
    print(f"{len(readers)} readers, top {k} among all the books they have not rated:")
    print(table.round(3).to_string(index=False))
    return table


def save_artifacts(artifacts_path, **artifacts):
    """Saves artifacts to the specified directory."""
    print("\nStep 6: Saving artifacts...")
    os.makedirs(artifacts_path, exist_ok=True)
    for name, artifact in artifacts.items():
        file_path = os.path.join(artifacts_path, f"{name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
            print(f"{name} saved to '{file_path}'.")


def main():
    """Main function to train the book recommender system."""
    # File paths
    data_file_path = "./data/dataset_final3.csv"
    artifacts_path = "artifacts/"

    # Load and inspect data
    book_df = load_data(data_file_path)
    data_overview(book_df)
    book_df = add_weighted_ratings(book_df)
    book_df_knn = preprocess_data_for_knn(book_df)


    _ , book_titles = create_user_item_matrix(book_df)

    # Train the KNN model, then evaluate the books for a reader
    knn_model, X_final_normalized = train_knn_model_with_metadata(book_df_knn)
    editions = EditionIndex(book_df_knn["Book-Title"], book_df_knn["Book-Author"])
    evaluate_reader_models(book_df, X_final_normalized, book_df_knn["Book-Title"], editions)

    # Save artifacts
    save_artifacts(
        artifacts_path,
        knn_model=knn_model,
        book_titles=book_titles,
        X_final=X_final_normalized,
        book_df=book_df,
        book_df_knn=book_df_knn[KNN_COLUMNS],
    )

    print("\nScript completed successfully.")


if __name__ == "__main__":
    main()
