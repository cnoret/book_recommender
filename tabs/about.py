import streamlit as st

REPO = "https://github.com/cnoret/next-chapter"


def show(counts):
    st.title("About Next Chapter")
    st.markdown('<p class="tagline">Find the book that comes after the one you just closed.</p>',
                unsafe_allow_html=True)
    st.markdown("""
Next Chapter suggests books in two complementary ways:

- **Books like this one**: a content-based nearest-neighbours model (KNN) that compares tags,
  descriptions, authors and thematic clusters.
- **Books for a reader**: a hybrid model that starts from the books a reader loved and suggests
  books close to them in content and loved by the same readers.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Books", f"{counts['books']:,}")
    c2.metric("Readers", f"{counts['readers']:,}")
    c3.metric("Ratings", f"{counts['ratings']:,}")

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
books that talk about the same things, not books that got the same mark. Other editions of the
same book (*Alias Grace* and *Alias Grace : A Novel*) are left out of the suggestions.
""")
    st.subheader("Books for a reader")
    st.markdown("""
The app starts from the books the reader loved (rated 8 or more) and gives every other book two
scores: its similarity in content to those books, and how often the readers who loved the same
books also loved it. Both are standardized and added up, books they already rated are left out,
and each pick shows the loved book it comes from.
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
| Same author | 0.462 | 0.003 |
| Tags overlap (Jaccard) | 0.624 | 0.228 |
| Shared readers | 0.322 | 0.060 |

**Books for a reader**: 20 % of the ratings are held out. For 400 readers, each model ranks every
book they have not rated and is scored on the held-out books they loved.

| Model | Hit@10 | Recall@10 | Distinct books |
| --- | --- | --- | --- |
| Most read | 0.132 | 0.040 | 15 |
| **Hybrid (content + co-liked)** | **0.302** | **0.125** | **1,591** |

*Hit@10* is the share of readers with at least one loved book among their ten picks, and
*distinct books* counts the different books over the 400 lists. The app first used an SVD model:
with about 22 ratings per reader, it gave almost everyone the same list and found a loved book
for fewer than 5 % of readers.
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
  (same-author matches from 0.280 to 0.483, shared readers from 0.210 to 0.328), without
  other editions of the same book.
- **Personal picks**: a hybrid of content and co-liked books instead of SVD, with the reason for
  each pick; a loved book found for 30 % of readers instead of 5 %.
- **A new interface**: pages instead of tabs, book cards, a Details panel, light and dark themes,
  and a layout that works on phones.
- **Reliability**: a cache for the scraper, fallbacks for missing descriptions and covers,
  automated tests.
- **Deployment and documentation**: pinned dependencies, a lighter Docker image, a full README.
""")

    st.header("Built with")
    st.markdown("""
[Streamlit](https://streamlit.io/) · [scikit-learn](https://scikit-learn.org/) ·
[pandas](https://pandas.pydata.org/) ·
[NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) ·
[Requests](https://requests.readthedocs.io/) · [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)
""")
    st.link_button("Source code on GitHub", REPO, icon=":material/code:")
    st.caption("Released under the MIT License.")
