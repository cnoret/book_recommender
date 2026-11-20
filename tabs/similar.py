import streamlit as st

from ui import book_grid, book_hero
from utils import recommend_book_knn


def show(catalog, books_df, books_df_knn, knn_model, X_final, editions):
    st.title("Books like this one")
    options = sorted(catalog.index, key=lambda t: catalog.at[t, "Display-Title"].lstrip("'\"").lower())
    # The selection survives page changes and can be set from any book card.
    if st.session_state.get("similar_select") is None and st.session_state.get("similar_book") in options:
        st.session_state["similar_select"] = st.session_state["similar_book"]
    choice = st.selectbox(
        "Choose a book you liked",
        options,
        index=None,
        format_func=lambda t: f"{catalog.at[t, 'Display-Title']} — {catalog.at[t, 'Book-Author']}",
        placeholder="Start typing a title…",
        key="similar_select",
    )
    if choice is None:
        st.info("Type a few letters of a title, then pick it from the list.", icon=":material/arrow_upward:")
        return
    st.session_state["similar_book"] = choice

    book_hero(catalog, choice)
    st.header("Ten similar books")
    st.caption("Close in tags, summary, author and theme. Ratings are not used to pick them.")
    titles, _, _ = recommend_book_knn(books_df, choice, books_df_knn, knn_model, X_final, editions)
    if titles:
        book_grid(catalog, titles, key="similar")
    else:
        st.warning("No similar books found for this one.")
