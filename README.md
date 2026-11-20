<h1 align="center">📚 Next Chapter</h1>

<p align="center"><em>Find the book that comes after the one you just closed.</em></p>

<p align="center">
  <a href="https://github.com/cnoret/book_recommender/actions/workflows/tests.yml"><img src="https://github.com/cnoret/book_recommender/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/built%20with-Streamlit-ff4b4b" alt="Built with Streamlit">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

**Next Chapter** is a book recommender built with Streamlit. It suggests books in two complementary ways:

- **Books like this one**: a content-based k-Nearest Neighbors model that compares tags, descriptions, authors and thematic clusters.
- **Books for a reader**: a hybrid model that starts from the books a reader loved and suggests books close to them in content and loved by the same readers.

The dataset is built from the [Book-Crossing](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) ratings, enriched with descriptions and subjects scraped from [Open Library](https://openlibrary.org/), then grouped into thematic clusters with HDBSCAN. The working dataset (`data/dataset_final3.csv`) holds **42,331 ratings** of **3,217 books** by **1,914 readers**, on a 1 to 10 scale.

It started as a team project in January 2025 and was later reworked from end to end by [Christophe Noret](https://github.com/cnoret): see [Project history](#-project-history).

---

## 🔍 Features

The app has five pages, reachable from the top navigation bar. Every book is shown as a card with its cover, author, average rating and number of ratings; its **Details** button opens the summary and a shortcut to similar books.

| Page | What it does |
| --- | --- |
| **Discover** | The ten most read books (by number of ratings) and the ten best rated (by Bayesian average), plus a "Surprise me" button. |
| **Similar books** | Pick a book and get the ten most similar ones from the KNN model. |
| **For a reader** | Choose a reader from the dataset (or a random one): the app shows the books they rated highest, then ten books they might like next, each with the loved book it comes from. Without a reader, the best books by Bayesian average. |
| **Search** | Search by title or author, sorted by Bayesian average, as covers or as a table. |
| **About** | How the recommendations work, evaluation, the data and its limits, and the project history with its credits. |

The look (colours, fonts, light and dark themes) is set in `.streamlit/config.toml`.

---

## 🧠 How the recommendations work

### Similar books (KNN)

Each book is described by four blocks of features, each one vectorized and L2-normalized on its own, then weighted (`FEATURE_WEIGHTS` in `train.py`):

| Block | Encoding | Weight |
| --- | --- | --- |
| Open Library subjects and final tags | TF-IDF | 1.0 |
| Author | one-hot | 0.8 |
| Description | TF-IDF (English stop words removed) | 0.7 |
| HDBSCAN cluster | one-hot | 0.3 |

Neighbors are found with cosine similarity. The rating is left out on purpose: similar books are books that talk about the same things, not books that got the same mark.

The catalog holds several editions of some books under slightly different titles (*Alias Grace* and *Alias Grace : A Novel*). `recommender.EditionIndex` recognizes them (same author, same main title, compatible subtitles), so a book never suggests its own other edition, and an edition is never suggested twice.

### Books for a reader (hybrid)

`recommender.ReaderRecommender` starts from the books the reader loved (rated 8 or more; all their books if none) and gives every other book two scores:

- **content**: its cosine similarity to the loved books, on the KNN features above;
- **co-liked**: how often the readers who loved the same books also loved it (item-to-item collaborative filtering).

Both scores are standardized, then added up. Books the reader has rated, and other editions of them, are left out. Each suggestion comes with the loved book that contributes the most to it, shown as "Because they liked…". The app builds this model at startup from the artifacts, in a fraction of a second.

### Weighted ratings

`train.py` adds three columns per book: `Avg-Rating` (mean rating), `Num-Ratings` (number of ratings) and `Bayes-Rating`, a Bayesian average that pulls books with few ratings towards the global mean (the prior is the median number of ratings per book). The Highest rated list, the search results and the guest recommendations are all sorted on `Bayes-Rating`.

---

## 📏 Evaluation

**Similar books.** `python evaluate_knn.py` takes 500 random books and compares their 10 neighbors with 10 random books:

| Metric | KNN | Random |
| --- | --- | --- |
| Same author | 0.462 | 0.003 |
| Tags overlap (Jaccard) | 0.624 | 0.228 |
| Shared readers (liked by someone who liked the query book) | 0.324 | 0.060 |

The shared readers metric is not used by the model, so it is an independent check that the neighbors make sense to real readers. The scores are a little lower than before other editions were left out, because an edition of the same book always counted as a perfect match.

**Books for a reader.** Printed by `python train.py`. 20 % of the ratings are held out; for 400 readers, each model ranks every book they have not rated, as the app does, and is scored on the held-out books they loved:

| Model | Hit@10 | Recall@10 | Distinct books | Overlap |
| --- | --- | --- | --- | --- |
| Most read | 0.115 | 0.044 | 15 | 0.899 |
| Bayesian average | 0.038 | 0.009 | 13 | 0.978 |
| SVD (predicted rating) | 0.048 | 0.012 | 76 | 0.527 |
| **Hybrid (content + co-liked)** | **0.305** | **0.135** | **1,602** | **0.010** |

*Hit@10*: share of readers with at least one loved book in their ten picks. *Distinct books*: different books over the 400 lists. *Overlap*: share of picks two readers have in common.

The app used to rank books by the rating an SVD model predicted. With about 22 ratings per reader, that prediction is mostly the book's own average, so almost everyone got the same list (half of it shared by any two readers) and it found a loved book for fewer than 5 % of readers. The hybrid finds one for 30 %, and its lists are personal.

---

## 🚀 Getting started

### Prerequisites

- Python 3.11 (see `runtime.txt`). `scikit-surprise` 1.1.4 does not build on recent Python versions such as 3.13.
- A C compiler, needed to build `scikit-surprise`.

### Run the app

```bash
git clone https://github.com/cnoret/book_recommender.git
cd book_recommender
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>. The trained models are already in `artifacts/`, so no training is needed to start the app.

### Retrain the models

```bash
python train.py        # rebuilds every file of artifacts/ from data/dataset_final3.csv (a few seconds)
python evaluate_knn.py # prints the similar-books metrics
```

### Run the tests

```bash
pip install pytest
pytest -q              # unit tests for recommender.py, utils.py and train.py, plus end-to-end runs of the app pages
```

The same tests run on GitHub Actions for every pull request and every push to `main`.

### Docker

```bash
docker build -t book-recommender .
docker run -p 7860:7860 book-recommender
```

The app is then available at <http://localhost:7860>. The image holds only the app code and `artifacts/`: `.dockerignore` keeps the data, notebooks, scraper and tests out of it.

---

## 🗂️ Project structure

```
book_recommender/
├── app.py                 # Streamlit entry point: loads the artifacts and sets up the pages
├── ui.py                  # Book cards, responsive grids and page styles
├── utils.py               # Loading and recommendation helpers used by the pages
├── recommender.py         # Editions of the same book, similar books, books for a reader
├── tabs/                  # One module per page
│   ├── discover.py        # Most read and highest rated books
│   ├── similar.py         # Similar books (KNN)
│   ├── for_you.py         # Picks for a reader (hybrid)
│   ├── search.py          # Search by title or author
│   └── about.py           # Method, evaluation, data, history and credits
├── .streamlit/config.toml # Theme (light and dark) and toolbar
├── train.py               # Trains the KNN model, evaluates the books for a reader, writes artifacts/
├── evaluate_knn.py        # Proxy metrics for the similar-books recommendations
├── tests/                 # pytest suite (recommender, utils, train, app)
├── .github/workflows/     # GitHub Actions: runs the tests
├── artifacts/             # Trained models and preprocessed data loaded by the app
│   ├── knn_model.pkl      # Content-based NearestNeighbors model
│   ├── X_final.pkl        # Feature matrix of the KNN model
│   ├── book_df_knn.pkl    # One row per book, aligned with X_final
│   ├── book_df.pkl        # Ratings with book details and weighted ratings
│   └── book_titles.pkl    # Titles offered in the book selector
├── data/
│   ├── Books.csv, Ratings.csv, Users.csv   # Raw Book-Crossing data
│   ├── dataset.csv                         # Intermediate dataset
│   ├── dataset_with_details.csv            # With scraped descriptions and subjects
│   ├── hdbscan_alternative_clusters_isbn_data.csv     # HDBSCAN clusters per ISBN
│   └── dataset_final3.csv                  # Final dataset used by train.py
├── notebooks/
│   ├── base.ipynb               # Cleaning and exploration of the raw data
│   ├── processor.ipynb          # Analysis and preparation of descriptions and tags
│   ├── clusters_creation.ipynb  # Tag processing and HDBSCAN clustering
│   └── trainSVD.ipynb           # SVD experiments
├── scrapper/
│   ├── scrapper.py              # Scrapes descriptions and subjects from Open Library
│   ├── scrapper_cache_check.py  # Inspects the scraping cache
│   └── requirements.txt         # Extra dependencies of the scraper
├── Dockerfile
├── .dockerignore              # Keeps data, notebooks and tests out of the Docker image
├── requirements.txt
├── runtime.txt
└── LICENSE
```

---

## 📜 Project history

### Version 1: a team project (January 2025)

Next Chapter began as a group project, built in a few days by four people:

| Member | Main contributions |
| --- | --- |
| [Christophe Noret](https://github.com/cnoret) | First Streamlit app, training script, Docker setup, Open Library scraper |
| [Fabien Messinger](https://github.com/f-msngr) | Data cleaning and exploration, split of the app into tabs, notebooks |
| [Florent Lebrun](https://github.com/FLebrun-67) | KNN and SVD models, final training pipeline and dataset |
| [Charles Dabin](https://github.com/charlesdab) | Tag processing and HDBSCAN thematic clusters |

### Version 2: the rework (November 2026)

Christophe Noret then took the project over alone and rebuilt most of it:

- **Better recommendations**: the KNN features were redesigned (separate, weighted blocks for tags, author, description and cluster), and `evaluate_knn.py` was added to measure them. Same-author matches went from 0.280 to 0.483 and shared readers from 0.210 to 0.328. Other editions of a book are now left out of its neighbours.
- **Books for a reader**: the SVD ranking, which gave almost everyone the same list, was replaced by a hybrid of content and co-liked books, with an explanation for each pick. It finds a loved book for 30 % of readers instead of 5 %.
- **A new interface**: page navigation instead of tabs, book cards with covers and ratings, a "Details" panel, light and dark themes, and a layout that works on phones.
- **Reliability**: a cache for the scraper, fallbacks for missing descriptions and covers, a pytest suite and GitHub Actions.
- **Deployment**: pinned dependencies and a lighter Docker image.
- **Documentation**: this README.

---

## 💪 Built with

- [Streamlit](https://streamlit.io/) for the interface
- [scikit-learn](https://scikit-learn.org/) for TF-IDF, one-hot encoding and the KNN model
- [Surprise](https://surpriselib.com/) for the SVD model used as a comparison in the evaluation
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for the data
- [Requests](https://requests.readthedocs.io/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for the scraping

---

## ⚖️ License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
