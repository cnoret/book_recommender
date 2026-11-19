# 📚 Book Recommender System

Welcome to the **Book Recommender System**! This project is a Streamlit-based application that provides personalized book recommendations using a k-Nearest Neighbors (kNN) model, and a collaborative filtering approach via SVD.

---

## 🔍 Features

### 🔢 General Statistics
- Displays the total number of books and users in the dataset.

### 🎮 Popular Books
- Shows the most popular books based on the number of ratings.

### 🌟 Top-Rated Books
- Highlights the top-rated books in the dataset.

### 🔍 Recommendations
- Provides book recommendations based on user selection.
- Allows users to rate recommendations and save their feedback.

### 🕵️‍♂️ Advanced Search
- Search for books by keywords.

### 🌐 Random Book Discovery
- Discover a random book from the dataset.

### 🎨 Visualizations
- View distributions of ratings and user interactions with books.

---

## 💪 Technologies Used

- **Streamlit**: For building the web-based user interface.
- **scikit-learn**: For implementing the kNN recommendation algorithm.
- **pandas**: For data manipulation.
- **plotly**: For creating interactive visualizations.
- **pickle**: For saving and loading the pre-trained model and datasets.
- **SVD (Singular Value Decomposition)**: For collaborative filtering-based recommendations.

---

## 🚀 How to Run the Application

### Prerequisites

- Python 3.9 or later
- Required Python libraries (listed in `requirements.txt`):

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/book-recommender-system.git
   cd book-recommender-system
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the required data and model files in the `artifacts/` directory:
   - `knn_model.pkl`
   - `svd_model.pkl`
   - `book_titles.pkl`
   - `book_df.pkl`

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501`.

---

## 🌄 File Structure

```
book-recommender-system/
├── app.py                        # Main application script  
├── utils.py                      # Functions library
├── tabs/                         # Tabs functions library
│   ├── tab0.py
│   ├── tab1.py
│   ├── tab2.py
│   ├── tab3.py
│   ├── tab4.py
│   ├── tab5.py
│   └── tab6.py
├── artifacts/                    # Contains model and dataset files
│   ├── svd_model.pkl             # Trained SVD model
│   ├── knn_model.pkl            # Trained KNN model
│   ├── book_df.pkl              # Book DataFrame
│   └── book_titles.pkl          # Titles of the books
├── data/                         # Raw and cleaned datasets
│   ├── dataset.csv              # Raw dataset with book info
│   ├── Ratings.csv              # Ratings data
│   ├── dataset_with_details.csv # Extended dataset
│   ├── cleaned_data.csv         # Preprocessed dataset
│   ├── Users.csv                # User data
│   └── Books.csv                # Book data
├── scrapper/                     # Scraping scripts
│   ├── scrapper.py              # Main scraping script
│   └── scrapper_cache_check.py  # Script to verify cached data
├── notebooks/                    # Jupyter notebooks for model training and analysis
│   ├── base.ipynb               # Basic EDA notebook
├── Dockerfile                    # Docker configuration file
├── train.py                      # Script for training the recommendation model
├── requirements.txt              # Dependencies for the project
├── runtime.txt                   # Specifies Python version for deployment
├── README.md                     # Project documentation
├── LICENSE                       # License information
└── .gitignore                    # Git ignore file
```

### Description of Key Files

- **artifacts/**: Contains all the generated artifacts from the training process, including the trained models and processed data.
- **data/**: Directory for raw and cleaned datasets.
- **scrapper/**: Contains the scraping scripts used to collect book and user data.
- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA) and model training.
- **Dockerfile**: Used for containerizing the application.
- **app.py**: The main application file for deploying the recommendation engine.
- **requirements.txt**: Lists all the Python dependencies required for the project.
- **runtime.txt**: Specifies the runtime environment for deployment (e.g., Python version).
- **train.py**: A standalone Python script for training the recommendation engine model.

---

## 🧠 Training and evaluation

`python train.py` rebuilds every file of `artifacts/` from `data/dataset_final3.csv`:

- **Similar books (KNN)**: cosine similarity on the Open Library subjects and final tags (TF-IDF), the descriptions (TF-IDF), the author and the HDBSCAN cluster (one-hot). Each block is normalized and weighted separately (`FEATURE_WEIGHTS` in `train.py`).
- **Weighted ratings**: `Avg-Rating`, `Num-Ratings` and a Bayesian average `Bayes-Rating`, used by the Top-Rated tab, the search and the guest recommendations.
- **SVD**: RMSE, MAE, precision@5 and recall@5 are printed on a 20 % held-out split, next to a baseline, then the final model is retrained on all the ratings.

`python evaluate_knn.py` prints proxy metrics for the similar-books recommendations (same author, tag overlap, shared readers), compared with random books.

---

## 🎨 Screenshots

### 🌐 Home Page
![Home Page](https://via.placeholder.com/600x300)

### 🔍 Recommendations
![Recommendations](https://via.placeholder.com/600x300)

### 🎨 Visualizations
![Visualizations](https://via.placeholder.com/600x300)

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thank you for exploring the Book Recommender System! 🚀