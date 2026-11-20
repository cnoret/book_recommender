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
- **Books for you**: a collaborative filtering model (SVD) that predicts how a reader would rate the books they have not read yet.

The dataset is built from the [Book-Crossing](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) ratings, enriched with descriptions and subjects scraped from [Open Library](https://openlibrary.org/), then grouped into thematic clusters with HDBSCAN. The working dataset (`data/dataset_final3.csv`) holds **42,331 ratings** of **3,217 books** by **1,914 readers**, on a 1 to 10 scale.

It started as a team project in January 2025 and was later reworked from end to end by [Christophe Noret](https://github.com/cnoret): see [Project history](#-project-history).

---

## 🔍 Features

The app has five pages, reachable from the top navigation bar. Every book is shown as a card with its cover, author, average rating and number of ratings; its **Details** button opens the summary and a shortcut to similar books.

| Page | What it does |
| --- | --- |
| **Discover** | The ten most read books (by number of ratings) and the ten best rated (by Bayesian average), plus a "Surprise me" button. |
| **Similar books** | Pick a book and get the ten most similar ones from the KNN model. |
| **For a reader** | Choose a reader from the dataset (or a random one): the app shows the books they rated highest, then the ten books the SVD model predicts they would like. Without a reader, the best books by Bayesian average. |
| **Search** | Search by title or author, sorted by Bayesian average, as covers or as a table. |
| **About** | How the recommendations work, the data and its limits. |

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

### Personal recommendations (SVD)

The SVD model from [Surprise](https://surpriselib.com/) (50 factors) is trained on the user × book ratings. `train.py` first evaluates it on a 20 % held-out split, next to a baseline, then retrains it on all the ratings before saving it.

### Weighted ratings

`train.py` adds three columns per book: `Avg-Rating` (mean rating), `Num-Ratings` (number of ratings) and `Bayes-Rating`, a Bayesian average that pulls books with few ratings towards the global mean (the prior is the median number of ratings per book). The Highest rated list, the search results and the guest recommendations are all sorted on `Bayes-Rating`.

---

## 📏 Evaluation

**Similar books.** `python evaluate_knn.py` takes 500 random books and compares their 10 neighbors with 10 random books:

| Metric | KNN | Random |
| --- | --- | --- |
| Same author | 0.483 | 0.003 |
| Tags overlap (Jaccard) | 0.628 | 0.228 |
| Shared readers (liked by someone who liked the query book) | 0.328 | 0.060 |

The shared readers metric is not used by the model, so it is an independent check that the neighbors make sense to real readers.

**Personal recommendations.** Printed by `python train.py` on the held-out split (precision and recall at 5, a rating of 8 or more counts as relevant):

| Model | RMSE | MAE | Precision@5 | Recall@5 |
| --- | --- | --- | --- | --- |
| SVD | 1.495 | 1.135 | 0.733 | 0.435 |
| Baseline (user and book biases) | 1.512 | 1.164 | 0.743 | 0.439 |
| Random order | | | 0.672 | 0.392 |

The ratings are sparse (about 22 per reader on average), so SVD barely beats the baseline for now. It remains a starting point for richer collaborative models.

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
pytest -q              # unit tests for utils.py and train.py, plus an end-to-end run of the app
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
├── utils.py               # Loading, KNN and SVD recommendation helpers
├── tabs/                  # One module per page
│   ├── discover.py        # Most read and highest rated books
│   ├── similar.py         # Similar books (KNN)
│   ├── for_you.py         # Picks for a reader (SVD)
│   ├── search.py          # Search by title or author
│   └── about.py           # Method, data and limits
├── .streamlit/config.toml # Theme (light and dark) and toolbar
├── train.py               # Trains the KNN and SVD models and writes artifacts/
├── evaluate_knn.py        # Proxy metrics for the similar-books recommendations
├── tests/                 # pytest suite (utils, train, app)
├── .github/workflows/     # GitHub Actions: runs the tests
├── artifacts/             # Trained models and preprocessed data loaded by the app
│   ├── knn_model.pkl      # Content-based NearestNeighbors model
│   ├── X_final.pkl        # Feature matrix of the KNN model
│   ├── book_df_knn.pkl    # One row per book, aligned with X_final
│   ├── svd_model.pkl      # SVD model
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

- **Better recommendations**: the KNN features were redesigned (separate, weighted blocks for tags, author, description and cluster), and `evaluate_knn.py` was added to measure them. Same-author matches went from 0.280 to 0.483 and shared readers from 0.210 to 0.328.
- **A new interface**: page navigation instead of tabs, book cards with covers and ratings, a "Details" panel, light and dark themes, and a layout that works on phones.
- **Reliability**: a cache for the scraper, fallbacks for missing descriptions and covers, a pytest suite and GitHub Actions.
- **Deployment**: pinned dependencies and a lighter Docker image.
- **Documentation**: this README.

---

## 💪 Built with

- [Streamlit](https://streamlit.io/) for the interface
- [scikit-learn](https://scikit-learn.org/) for TF-IDF, one-hot encoding and the KNN model
- [Surprise](https://surpriselib.com/) for the SVD model
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for the data
- [Requests](https://requests.readthedocs.io/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for the scraping

---

## ⚖️ License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
