# tabs/tab6.py
import streamlit as st
from utils import show_book_grid

def show_top_rated_books(books_df):
    """Display top-rated books tab."""
    st.subheader("⭐ Top-rated books")
    
    top_rated_books = (
        books_df.drop_duplicates(subset="Book-Title")
        .sort_values(by="Bayes-Rating", ascending=False)
        .head(10)
    )
    
    show_book_grid(top_rated_books)
