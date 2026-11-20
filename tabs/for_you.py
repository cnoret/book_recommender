import random

import streamlit as st

from ui import book_grid, display_title
from utils import recommend_for_reader


@st.cache_data(show_spinner=False)
def reader_stats(_books_df):
    return _books_df.groupby("User-ID")["Book-Rating"].count().sort_values(ascending=False)


def show(catalog, books_df, reader_model):
    st.title("Picks for a reader")
    st.markdown(
        "This demo has no accounts: choose one of the 1,914 readers of the dataset and see "
        "the books the app would suggest to them next."
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
        recs = recommend_for_reader(None, books_df, reader_model)
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

    st.header("What they might like next")
    st.caption("Books they have not rated, close in content to the ones they loved and liked by "
               "the readers who loved the same books.")
    with st.spinner("Finding books…"):
        recs = recommend_for_reader(reader, books_df, reader_model)
    book_grid(catalog, [title for title, _ in recs], key="reader-picks",
              badges=[f"Because they liked {short_title(catalog, source)}" for _, source in recs])


def short_title(catalog, raw, limit=40):
    """Display title, cut at `limit` characters on a word boundary."""
    title = catalog.at[raw, "Display-Title"] if raw in catalog.index else display_title(raw)
    title = title.split(":")[0].split(" (")[0].strip()
    if len(title) > limit:
        title = title[:limit].rsplit(" ", 1)[0] + "…"
    return title
