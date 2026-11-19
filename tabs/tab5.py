# tabs/tab5.py
import streamlit as st
from utils import show_book_grid

def show_popular_books(books_df):
    """Display popular books tab."""
    st.subheader("📈 Popular books")
    
    popular_books = (
        books_df.sort_values(by="Rating-Count", ascending=False)
        .drop_duplicates(subset="Book-Title")
        .head(10)
    )

    show_book_grid(popular_books)
