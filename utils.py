import html
import pickle
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model_and_data():
    """Load the pre-trained model and data."""
    knn_model = pickle.load(open("artifacts/knn_model.pkl", "rb"))
    book_titles = pickle.load(open("artifacts/book_titles.pkl", "rb"))
    books_df = pickle.load(open("artifacts/book_df.pkl", "rb"))
    svd_model = pickle.load(open('artifacts/svd_model.pkl', 'rb'))
    X_final = pickle.load(open('artifacts/X_final.pkl', 'rb'))
    books_df_knn = pickle.load(open('artifacts/book_df_knn.pkl', 'rb'))

    return knn_model, book_titles, books_df, svd_model, X_final, books_df_knn

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


def recommend_book_knn(books_df, book_name, books_df_knn, knn_model, X_final):
    """Recommends books similar to `book_name` with the content-based KNN model."""
    try:
        if book_name not in books_df_knn["Book-Title"].values:
            st.error("Selected book doesn't exist.")
            return [], [], []

        book_idx = books_df_knn[books_df_knn["Book-Title"] == book_name].index[0]
        # Slicing keeps a 2D row and works for both dense and sparse matrices.
        _, suggestions = knn_model.kneighbors(X_final[book_idx:book_idx + 1], n_neighbors=11)
        books_list = [books_df_knn.iloc[suggestion_id]["Book-Title"] for suggestion_id in suggestions[0]]
        books_list = [book for book in books_list if book != book_name]
        
        book_names, poster_urls, book_descriptions = fetch_poster(books_df, books_list[:10])
        return book_names, poster_urls, book_descriptions
    except Exception as e:
        st.error("Recommendation display error (KNN): " + str(e))
        return [], [], []

def recommend_book_svd(user_id, books_df, svd_model, n_recommendations=10):
    """Recommend books for a given user using the SVD model."""
    if user_id not in books_df["User-ID"].unique():
        # Bayesian average: a book rated 10 by a single reader does not win.
        popular_books = (
            books_df.drop_duplicates(subset="Book-Title")
            .set_index("Book-Title")["Bayes-Rating"]
            .sort_values(ascending=False)
            .head(n_recommendations)
        )
        book_name, poster_url, book_descriptions = fetch_poster(books_df, popular_books.index.tolist())
        return list(zip(book_name, popular_books.values, poster_url, book_descriptions))

    all_books = books_df["Book-Title"].unique()
    user_books = books_df[books_df["User-ID"] == user_id]["Book-Title"].unique()
    books_to_predict = [book for book in all_books if book not in user_books]
    
    if not books_to_predict:
        st.warning("No books to find for this user.")
        return []

    predictions = [
        (book, svd_model.predict(user_id, book).est) for book in books_to_predict
    ]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]

    book_name, poster_url, book_descriptions = fetch_poster(books_df, [book for book, _ in predictions])
    return list(zip(book_name, [rating for _, rating in predictions], poster_url, book_descriptions))

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
