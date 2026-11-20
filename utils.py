import html
import pickle

import streamlit as st

from recommender import EditionIndex, ReaderRecommender, similar_books


def _load(name):
    with open(f"artifacts/{name}.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model_and_data():
    """Loads the trained artifacts and builds the reader recommender."""
    knn_model = _load("knn_model")
    book_titles = _load("book_titles")
    books_df = _load("book_df")
    X_final = _load("X_final")
    books_df_knn = _load("book_df_knn")
    editions = EditionIndex(books_df_knn["Book-Title"], books_df_knn["Book-Author"])
    reader_model = ReaderRecommender(books_df, X_final, books_df_knn["Book-Title"], editions)
    return knn_model, book_titles, books_df, X_final, books_df_knn, editions, reader_model

# Neutral book cover drawn inline, so it never depends on an external service.
DEFAULT_IMAGE = (
    "data:image/svg+xml;charset=utf-8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='220' viewBox='0 0 150 220'%3E"
    "%3Crect width='150' height='220' rx='6' fill='%23e9e6e1'/%3E"
    "%3Crect x='18' y='30' width='114' height='8' rx='4' fill='%23c9c3ba'/%3E"
    "%3Crect x='30' y='48' width='90' height='8' rx='4' fill='%23c9c3ba'/%3E"
    "%3Ctext x='75' y='130' font-size='48' text-anchor='middle'%3E%F0%9F%93%96%3C/text%3E"
    "%3C/svg%3E"
)

# Values left by the scraper when Open Library had nothing for a book.
DESCRIPTION_PLACEHOLDERS = {
    "No description found",
    "No description available",
    "Error: Failed after multiple retries",
}


def clean_description(description):
    """Returns the description, or an empty string when it is missing or a placeholder."""
    if not isinstance(description, str):
        return ""
    description = description.strip()
    return "" if description in DESCRIPTION_PLACEHOLDERS else description


def clean_image_url(img_url):
    """Returns the image URL, or the default cover when it is missing or invalid."""
    if isinstance(img_url, str) and img_url.startswith("http"):
        return img_url
    return DEFAULT_IMAGE


def fetch_poster(books_df, book_list):
    """Returns the titles, cover URLs and descriptions of the given books.

    A missing description is returned as an empty string, so that the tabs
    display nothing rather than a scraping placeholder.
    """
    first_rows = books_df.drop_duplicates(subset="Book-Title").set_index("Book-Title")
    book_names, poster_urls, book_descriptions = [], [], []

    for book in book_list:
        book_names.append(book)
        if book in first_rows.index:
            row = first_rows.loc[book]
            poster_urls.append(clean_image_url(row["Image-URL-L"]))
            book_descriptions.append(clean_description(row["Description"]))
        else:
            poster_urls.append(DEFAULT_IMAGE)
            book_descriptions.append("")

    return book_names, poster_urls, book_descriptions


def recommend_book_knn(books_df, book_name, books_df_knn, knn_model, X_final, editions=None):
    """Recommends books similar to `book_name` with the content-based KNN model.

    Other editions of the book, and repeated editions among the neighbours,
    are left out.
    """
    if book_name not in books_df_knn["Book-Title"].values:
        st.error("Selected book doesn't exist.")
        return [], [], []
    titles = books_df_knn["Book-Title"].tolist()
    if editions is None:
        editions = EditionIndex(titles, books_df_knn["Book-Author"])
    book_idx = titles.index(book_name)
    books_list = similar_books(book_idx, knn_model, X_final, titles, editions, n=10)
    return fetch_poster(books_df, books_list)


def recommend_for_reader(user_id, books_df, reader_model, n_recommendations=10):
    """Books for a reader of the dataset, as (title, liked title it comes from) pairs.

    Unknown readers and guests get the best books by Bayesian average, which
    a single 10/10 cannot top; their second element is None.
    """
    if reader_model is None or not reader_model.knows(user_id):
        popular_books = (
            books_df.drop_duplicates(subset="Book-Title")
            .sort_values("Bayes-Rating", ascending=False)["Book-Title"]
            .head(n_recommendations)
        )
        return [(title, None) for title in popular_books]
    return reader_model.recommend(user_id, n=n_recommendations)


def render_aligned_image(image_url, title, height=500):
    """Returns the HTML of a book cover with its title underneath.

    The title is escaped: some titles in the dataset hold HTML entities or
    characters such as < and &.
    """
    safe_title = html.escape(html.unescape(str(title)))
    safe_url = html.escape(str(image_url), quote=True)
    return f"""
    <div style="text-align: center;">
        <img src="{safe_url}" alt="{safe_title}" style="height: {height}px; object-fit: contain; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 8px;">
        <p style="margin-top: 5px; font-size: 14px;"><b>{safe_title}</b></p>
    </div>
    """


def show_book_grid(books, per_row=5):
    """Displays the covers of a DataFrame of books, `per_row` books per row."""
    for start in range(0, len(books), per_row):
        row = books.iloc[start:start + per_row]
        cols = st.columns(per_row)
        for col, (_, book) in zip(cols, row.iterrows()):
            col.markdown(
                render_aligned_image(clean_image_url(book["Image-URL-L"]), book["Book-Title"]),
                unsafe_allow_html=True,
            )
