"""Shared display helpers: the book catalog, book cards and grids."""
import html

import streamlit as st

from utils import DEFAULT_IMAGE, clean_description, clean_image_url

CSS = """
<style>
/* Book card */
.book-card { display: flex; flex-direction: column; gap: .35rem; }
.book-cover {
    position: relative; width: 100%; aspect-ratio: 2 / 3; overflow: hidden;
    border-radius: .45rem; border: 1px solid var(--book-border, rgba(128,128,128,.25));
    background: center / 60% no-repeat url("%(default)s"), rgba(128,128,128,.10);
    box-shadow: 0 1px 2px rgba(0,0,0,.08), 0 6px 16px rgba(0,0,0,.06);
}
.book-cover img { width: 100%%; height: 100%%; object-fit: cover; display: block; color: transparent; font-size: 0; }
[data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
.book-title {
    font-weight: 600; font-size: .95rem; line-height: 1.25; margin: .2rem 0 0 0;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    min-height: 2.5em;
}
.book-author { font-size: .85rem; opacity: .72; margin: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.book-meta { font-size: .8rem; opacity: .72; margin: 0; }
.book-meta .star { color: #B8860B; }
.book-badge { font-size: .75rem; font-weight: 600; opacity: .9; }
/* Book grids: 5 per row on desktop, 2 on phones, whatever the column width */
[data-testid="stGrid"] { grid-template-columns: repeat(5, minmax(0, 1fr)) !important; }
@media (max-width: 1100px) { [data-testid="stGrid"] { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; } }
@media (max-width: 820px)  { [data-testid="stGrid"] { grid-template-columns: repeat(3, minmax(0, 1fr)) !important; } }
@media (max-width: 560px)  { [data-testid="stGrid"] { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; } }
/* Tagline under the app name */
.tagline { font-family: Georgia, "Times New Roman", serif; font-style: italic; font-size: 1.35rem;
    opacity: .8; margin: -.6rem 0 .4rem 0; }
/* Selected book header */
.book-hero { display: flex; gap: 1.5rem; align-items: flex-start; }
.book-hero .book-cover { width: 150px; flex: 0 0 150px; }
.book-hero h3 { margin: 0 0 .25rem 0; }
@media (max-width: 640px) {
    .book-hero { gap: 1rem; }
    .book-hero .book-cover { width: 96px; flex-basis: 96px; }
}
</style>
"""


def inject_css():
    st.markdown(CSS.replace("%(default)s", DEFAULT_IMAGE.replace('"', "%22")).replace("%%", "%"),
                unsafe_allow_html=True)


def display_title(title):
    """Title as it should be read: HTML entities such as &amp; decoded."""
    return html.unescape(str(title))


@st.cache_data(show_spinner=False)
def build_catalog(_books_df):
    """One row per title with everything a card needs, indexed by the raw title."""
    catalog = (
        _books_df.drop_duplicates(subset="Book-Title")
        .set_index("Book-Title")[
            ["Book-Author", "Image-URL-L", "Description", "Avg-Rating", "Num-Ratings", "Bayes-Rating"]
        ]
        .copy()
    )
    catalog["Display-Title"] = [display_title(t) for t in catalog.index]
    catalog["Cover"] = catalog["Image-URL-L"].map(clean_image_url)
    catalog["Summary"] = catalog["Description"].map(clean_description)
    return catalog


@st.cache_data(show_spinner=False)
def dataset_counts(_books_df):
    """Number of books, readers and ratings, shown in the footer and the About page."""
    return {
        "books": _books_df["Book-Title"].nunique(),
        "readers": _books_df["User-ID"].nunique(),
        "ratings": len(_books_df),
    }


def card_html(title, author, cover, avg=None, count=None, badge=None):
    esc = html.escape
    meta = ""
    if avg is not None and count is not None:
        meta = (f'<p class="book-meta"><span class="star">★</span> {avg:.1f}/10'
                f' · {int(count)} rating{"s" if count != 1 else ""}</p>')
    badge_html = f'<p class="book-badge">{esc(badge)}</p>' if badge else ""
    return f"""
<div class="book-card">
  <div class="book-cover"><img src="{esc(cover, quote=True)}" alt="Cover of {esc(title, quote=True)}" loading="lazy"></div>
  <p class="book-title" title="{esc(title, quote=True)}">{esc(title)}</p>
  <p class="book-author">{esc(str(author))}</p>
  {meta}{badge_html}
</div>"""


def go_to_similar(raw_title):
    st.session_state["similar_book"] = raw_title
    st.session_state["similar_select"] = raw_title
    st.switch_page(st.session_state["pages"]["similar"])


def book_grid(catalog, titles, key, badges=None, min_width=140):
    """Responsive grid of book cards; each card has a Details popover."""
    titles = [t for t in titles if t in catalog.index]
    if not titles:
        return
    grid = st.grid(min_column_width=min_width, gap="medium")
    for i, raw in enumerate(titles):
        book = catalog.loc[raw]
        with grid.container():
            st.markdown(
                card_html(book["Display-Title"], book["Book-Author"], book["Cover"],
                          book["Avg-Rating"], book["Num-Ratings"],
                          badges[i] if badges else None),
                unsafe_allow_html=True,
            )
            with st.popover("Details", use_container_width=True):
                st.markdown(f"**{book['Display-Title']}**  \n{book['Book-Author']}")
                if book["Summary"]:
                    st.text(book["Summary"][:900] + ("…" if len(book["Summary"]) > 900 else ""))
                else:
                    st.caption("No summary available for this book.")
                if st.button("Find similar books", key=f"{key}-sim-{i}", type="primary",
                             use_container_width=True):
                    go_to_similar(raw)


def book_hero(catalog, raw):
    """Large header for one book: cover, title, author, rating and summary."""
    book = catalog.loc[raw]
    esc = html.escape
    summary = book["Summary"]
    if len(summary) > 420:
        summary = summary[:420].rsplit(" ", 1)[0] + "…"
    st.markdown(f"""
<div class="book-hero">
  <div class="book-cover"><img src="{esc(book['Cover'], quote=True)}" alt="Cover of {esc(book['Display-Title'], quote=True)}"></div>
  <div>
    <h3>{esc(book['Display-Title'])}</h3>
    <p class="book-author" style="white-space:normal">{esc(str(book['Book-Author']))}</p>
    <p class="book-meta"><span class="star">★</span> {book['Avg-Rating']:.1f}/10 · {int(book['Num-Ratings'])} ratings</p>
    <p style="font-size:.92rem;margin-top:.6rem">{esc(summary)}</p>
  </div>
</div>""", unsafe_allow_html=True)
