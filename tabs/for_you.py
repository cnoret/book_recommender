import random

import streamlit as st

from ui import book_grid
from utils import recommend_book_svd


@st.cache_data(show_spinner=False)
def reader_stats(_books_df):
    return _books_df.groupby("User-ID")["Book-Rating"].count().sort_values(ascending=False)


def show(catalog, books_df, svd_model):
    st.title("Picks for a reader")
    st.markdown(
        "This demo has no accounts: choose one of the 1,914 readers of the dataset and see "
        "the books the model predicts they would rate highest."
    )
    stats = reader_stats(books_df)
    readers = stats.index.tolist()

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    if c2.button("Random reader", icon=":material/casino:", use_container_width=True):
        st.session_state["reader"] = random.choice(readers)
    reader = c1.selectbox(
        "Reader",
        readers,
        index=None,
        key="reader",
        format_func=lambda u: f"Reader #{u} · {stats[u]} books rated",
        placeholder="Choose a reader, or leave empty to see general favourites",
    )

    if reader is None:
        st.header("General favourites")
        st.caption("Without a reader, the best books by Bayesian average.")
        recs = recommend_book_svd("guest", books_df, svd_model)
        book_grid(catalog, [r[0] for r in recs], key="guest")
        return

    liked = (
        books_df[books_df["User-ID"] == reader]
        .sort_values("Book-Rating", ascending=False)
        .drop_duplicates("Book-Title")
        .head(5)
    )
    st.header("What they loved")
    book_grid(catalog, liked["Book-Title"].tolist(), key="liked",
              badges=[f"Their rating: {r}/10" for r in liked["Book-Rating"]])

    st.header("What the model suggests")
    st.caption("Books they have not rated, ranked by the rating the model predicts for them.")
    with st.spinner("Computing predictions…"):
        recs = recommend_book_svd(reader, books_df, svd_model)
    book_grid(catalog, [r[0] for r in recs], key="svd",
              badges=[f"Predicted: {r[1]:.1f}/10" for r in recs])
