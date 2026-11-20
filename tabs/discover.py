import streamlit as st

from ui import book_grid, go_to_similar


def show(catalog):
    st.title("Next Chapter")
    st.markdown('<p class="tagline">Find the book that comes after the one you just closed.</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Start from a book you loved to get ten close matches, or step into the shoes of a reader "
        "from the dataset to see what the model would suggest to them."
    )
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Start from a book", type="primary", icon=":material/library_books:", use_container_width=True):
        st.switch_page(st.session_state["pages"]["similar"])
    if c2.button("Surprise me", icon=":material/casino:", use_container_width=True):
        go_to_similar(catalog.sample(1).index[0])

    st.header("Most read")
    st.caption("The books with the most ratings in the dataset.")
    most_read = catalog.sort_values("Num-Ratings", ascending=False).head(10).index.tolist()
    book_grid(catalog, most_read, key="most-read")

    st.header("Highest rated")
    st.caption("Ranked by Bayesian average, so a book rated 10 by a single reader does not top the list.")
    top = catalog.sort_values("Bayes-Rating", ascending=False).head(10).index.tolist()
    book_grid(catalog, top, key="top-rated")
