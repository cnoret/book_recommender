import streamlit as st


def show():
    st.title("About")
    st.markdown("""
This app recommends books in two complementary ways.

**Similar books** use a content-based nearest-neighbours model. Each book is described by its
tags, its summary, its author and a thematic cluster, and the app returns the ten closest books.
Ratings are deliberately left out, so a little-known book can still surface.

**Picks for a reader** use collaborative filtering (SVD): from the ratings of all readers, the
model predicts how a given reader would rate each book they have not read, and keeps the ten best.

Rankings by rating use a **Bayesian average**, which pulls books with few ratings towards the
global mean, so that a single 10/10 does not put a book at the top.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", "3,217")
    c2.metric("Readers", "1,914")
    c3.metric("Ratings", "42,331")
    st.markdown("""
**Data.** Ratings from the Book-Crossing dataset (1 to 10 scale), summaries and subjects from
Open Library, thematic clusters built with HDBSCAN.

**Limits.** The dataset dates from 2004, covers come from an old Amazon image service and some are
missing, and readers rated about 16 books each, which limits how personal the reader picks can be.
""")
