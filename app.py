import streamlit as st

from tabs import about, discover, for_you, search, similar
from ui import build_catalog, dataset_counts, inject_css
from utils import load_model_and_data

st.set_page_config(
    page_title="Next Chapter",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

knn_model, book_titles, books_df, X_final, books_df_knn, editions, reader_model = load_model_and_data()
catalog = build_catalog(books_df)
counts = dataset_counts(books_df)
inject_css()

pages = {
    "discover": st.Page(lambda: discover.show(catalog), title="Discover", icon=":material/auto_stories:",
                        url_path="discover", default=True),
    "similar": st.Page(lambda: similar.show(catalog, books_df, books_df_knn, knn_model, X_final, editions),
                       title="Similar books", icon=":material/library_books:", url_path="similar"),
    "for_you": st.Page(lambda: for_you.show(catalog, books_df, reader_model, counts), title="For a reader",
                       icon=":material/person:", url_path="for-you"),
    "search": st.Page(lambda: search.show(catalog), title="Search", icon=":material/search:",
                      url_path="search"),
    "about": st.Page(lambda: about.show(counts), title="About", icon=":material/info:", url_path="about"),
}
st.session_state["pages"] = pages

st.navigation(list(pages.values()), position="top").run()

st.divider()
st.caption(
    f"**Next Chapter** · {counts['books']:,} books and {counts['ratings']:,} ratings from the Book-Crossing dataset, "
    "with summaries from Open Library. Covers may be missing for some books. · "
    "[Source code on GitHub](https://github.com/cnoret/book_recommender)"
)
