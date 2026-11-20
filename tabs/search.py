import streamlit as st

from ui import book_grid


def show(catalog):
    st.title("Search the catalog")
    query = st.text_input("Title or author", placeholder="e.g. Tolkien, harry potter, dune…",
                          icon=":material/search:")
    if not query:
        st.caption(f"{len(catalog):,} books to explore.")
        return

    q = query.strip().lower()
    hits = catalog[
        catalog["Display-Title"].str.lower().str.contains(q, regex=False)
        | catalog["Book-Author"].astype(str).str.lower().str.contains(q, regex=False)
    ].sort_values("Bayes-Rating", ascending=False)

    if hits.empty:
        st.info(f"No book matches “{query}”. Try a shorter word or an author's last name.",
                icon=":material/search_off:")
        return

    view = st.segmented_control("View", ["Covers", "Table"], default="Covers",
                                label_visibility="collapsed")
    st.markdown(f"**{len(hits)} book{'s' if len(hits) > 1 else ''}**, best rated first")
    if view == "Table":
        table = hits[["Display-Title", "Book-Author", "Avg-Rating", "Num-Ratings"]].rename(columns={
            "Display-Title": "Title", "Book-Author": "Author",
            "Avg-Rating": "Average rating", "Num-Ratings": "Ratings"})
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Average rating": st.column_config.NumberColumn(format="%.1f")})
    else:
        book_grid(catalog, hits.index[:30].tolist(), key="search")
        if len(hits) > 30:
            st.caption(f"Showing the first 30 of {len(hits)}. Switch to Table to see them all.")
