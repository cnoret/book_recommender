import streamlit as st

REPO = "https://github.com/cnoret/book_recommender"


def show():
    st.title("About Next Chapter")
    st.markdown('<p class="tagline">Find the book that comes after the one you just closed.</p>',
                unsafe_allow_html=True)
    st.markdown("""
Next Chapter suggests books in two complementary ways:

- **Books like this one**: a content-based nearest-neighbours model (KNN) that compares tags,
  descriptions, authors and thematic clusters.
- **Books for you**: a collaborative filtering model (SVD) that predicts how a reader would rate
  the books they have not read yet.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", "3,217")
    c2.metric("Readers", "1,914")
    c3.metric("Ratings", "42,331")

    st.header("How the recommendations work")
    st.subheader("Similar books")
    st.markdown("""
Each book is described by four blocks of features, each one vectorized and normalized on its own,
then weighted:

| Block | Encoding | Weight |
| --- | --- | --- |
| Open Library subjects and final tags | TF-IDF | 1.0 |
| Author | one-hot | 0.8 |
| Description | TF-IDF | 0.7 |
| HDBSCAN thematic cluster | one-hot | 0.3 |

Neighbours are found with cosine similarity. The rating is left out on purpose: similar books are
books that talk about the same things, not books that got the same mark.
""")
    st.subheader("Books for a reader")
    st.markdown("""
The SVD model from Surprise (50 factors) is trained on the reader × book ratings and predicts
how a reader would rate each book they have not rated; the app keeps the ten best.
""")
    st.subheader("Weighted ratings")
    st.markdown("""
Rankings by rating use a **Bayesian average**, which pulls books with few ratings towards the
global mean, so that a single 10/10 does not put a book at the top.
""")

    with st.expander("Evaluation"):
        st.markdown("""
**Similar books**: 500 random books, their 10 neighbours compared with 10 random books.

| Metric | KNN | Random |
| --- | --- | --- |
| Same author | 0.483 | 0.003 |
| Tags overlap (Jaccard) | 0.628 | 0.228 |
| Shared readers | 0.328 | 0.060 |

**Books for a reader**: on a 20 % held-out split, a rating of 8 or more counts as relevant.

| Model | RMSE | Precision@5 | Recall@5 |
| --- | --- | --- | --- |
| SVD | 1.495 | 0.733 | 0.435 |
| Baseline (user and book biases) | 1.512 | 0.743 | 0.439 |
| Random order | | 0.672 | 0.392 |

Ratings are sparse (about 22 per reader), so SVD barely beats the baseline for now.
""")

    st.header("Data and limits")
    st.markdown("""
Ratings come from the [Book-Crossing](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
dataset (1 to 10 scale), enriched with descriptions and subjects from
[Open Library](https://openlibrary.org/), then grouped into thematic clusters with HDBSCAN.

The dataset dates from 2004, covers come from an old Amazon image service and some are missing,
and readers rated few books each, which limits how personal the reader picks can be.
""")

    st.header("Project history")
    st.subheader("Version 1: a team project (January 2025)")
    st.markdown("""
Next Chapter began as a group project, built in a few days by four people:

| Member | Main contributions |
| --- | --- |
| [Christophe Noret](https://github.com/cnoret) | First Streamlit app, training script, Docker setup, Open Library scraper |
| [Fabien Messinger](https://github.com/f-msngr) | Data cleaning and exploration, split of the app into tabs, notebooks |
| [Florent Lebrun](https://github.com/FLebrun-67) | KNN and SVD models, final training pipeline and dataset |
| [Charles Dabin](https://github.com/charlesdab) | Tag processing and HDBSCAN thematic clusters |
""")
    st.subheader("Version 2: the rework (November 2026)")
    st.markdown("""
[Christophe Noret](https://github.com/cnoret) then took the project over alone and rebuilt most of it:

- **Better recommendations**: redesigned KNN features and a dedicated evaluation
  (same-author matches from 0.280 to 0.483, shared readers from 0.210 to 0.328).
- **A new interface**: pages instead of tabs, book cards, a Details panel, light and dark themes,
  and a layout that works on phones.
- **Reliability**: a cache for the scraper, fallbacks for missing descriptions and covers,
  automated tests.
- **Deployment and documentation**: pinned dependencies, a lighter Docker image, a full README.
""")

    st.header("Built with")
    st.markdown("""
[Streamlit](https://streamlit.io/) · [scikit-learn](https://scikit-learn.org/) ·
[Surprise](https://surpriselib.com/) · [pandas](https://pandas.pydata.org/) ·
[NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) ·
[Requests](https://requests.readthedocs.io/) · [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
""")
    st.link_button("Source code on GitHub", REPO, icon=":material/code:")
    st.caption("Released under the MIT License.")
