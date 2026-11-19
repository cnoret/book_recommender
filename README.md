# 📚 Book Recommender System

A Streamlit application that recommends books in two complementary ways:

- **Books like this one**: a content-based k-Nearest Neighbors model that compares tags, descriptions, authors and thematic clusters.
- **Books for you**: a collaborative filtering model (SVD) that predicts how a reader would rate the books they have not read yet.

The dataset is built from the [Book-Crossing](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) ratings, enriched with descriptions and subjects scraped from [Open Library](https://openlibrary.org/), then grouped into thematic clusters with HDBSCAN. The working dataset (`data/dataset_final3.csv`) holds **42,331 ratings** of **3,217 books** by **1,914 readers**, on a 1 to 10 scale.

---

## 🔍 Features

The app is organized in tabs.

| Tab | What it does |
| --- | --- |
| 🦄 **Login / Guest** | Pick a reader from the dataset, or stay as a guest. The choice drives the personalized recommendations. |
| 🧑‍💻 **My Recommendations** | Ten books the selected reader has not rated yet, ranked by the rating the SVD model predicts. Guests get the ten best books by Bayesian average instead. Descriptions can be shown, and a history keeps the past recommendations of the session. |
| 📚 **Recommendations by books** | Pick a book and get the ten most similar ones from the KNN model, with covers and optional descriptions. |
| 🔍 **Search** | Keyword search on titles, with author, average rating and number of ratings, sorted by Bayesian average. A button also suggests a random book. |
| 📈 **Popular books** | The most rated books of the dataset. |
| ⭐ **Top-Rated books** | The best books by Bayesian average, so that a book rated 10 by a single reader does not top the list. |

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

`train.py` adds three columns per book: `Avg-Rating` (mean rating), `Num-Ratings` (number of ratings) and `Bayes-Rating`, a Bayesian average that pulls books with few ratings towards the global mean (the prior is the median number of ratings per book). The Top-Rated tab, the search results and the guest recommendations are all sorted on `Bayes-Rating`.

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
├── app.py                 # Streamlit entry point: loads the artifacts and lays out the tabs
├── utils.py               # Loading, KNN and SVD recommendation, poster and display helpers
├── tabs/                  # One module per tab
│   ├── tab0.py            # Login / Guest
│   ├── tab1.py            # My Recommendations (SVD)
│   ├── tab2.py            # Recommendations by books (KNN)
│   ├── tab3.py            # Search and random book
│   ├── tab5.py            # Popular books
│   └── tab6.py            # Top-Rated books
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

## 💪 Built with

- [Streamlit](https://streamlit.io/) for the interface
- [scikit-learn](https://scikit-learn.org/) for TF-IDF, one-hot encoding and the KNN model
- [Surprise](https://surpriselib.com/) for the SVD model
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for the data
- [Requests](https://requests.readthedocs.io/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for the scraping

---

## ⚖️ License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
