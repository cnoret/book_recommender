import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_discover_page_shows_both_lists():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [h.value for h in at.header] == ["Most read", "Highest rated"]
    # "Most read" counts real ratings, so the mass-mailed Wild Animus is not in it.
    most_read = _markdown(at).split("Highest rated")[0]
    assert "Wild Animus" not in most_read
    assert "The Lovely Bones" in most_read


def test_similar_books_page():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="most-read-sim-0").click().run()
    assert not at.exception
    assert at.title[0].value == "Books like this one"
    assert at.header[0].value == "Ten similar books"
    assert _markdown(at).count('class="book-card"') == 10

    at.selectbox(key="similar_select").set_value("Jurassic Park").run()
    assert not at.exception
    assert "The Lost World" in _markdown(at)


def test_titles_are_unescaped():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert "&amp;amp;" not in _markdown(at)
    assert "Griffin &amp; Sabine" in _markdown(at)  # escaped once, as HTML should be


def test_branding_and_counts():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.title[0].value == "Next Chapter"
    assert "Find the book that comes after the one you just closed." in _markdown(at)
    # The footer counts come from the data, not from a hard-coded text.
    assert any("3,217 books and 42,331 ratings" in c.value for c in at.caption)


def _page(name):
    """Runs one page alone: the pages of app.py are not files AppTest can switch to."""
    def script(page):
        import streamlit as st

        from tabs import about, for_you, search
        from ui import build_catalog, dataset_counts
        from utils import load_model_and_data

        _, _, books_df, _, _, _, reader_model = load_model_and_data()
        catalog = build_catalog(books_df)
        st.session_state["pages"] = {}
        if page == "for_you":
            for_you.show(catalog, books_df, reader_model, dataset_counts(books_df))
        elif page == "about":
            about.show(dataset_counts(books_df))
        else:
            search.show(catalog)

    return AppTest.from_function(script, args=(name,), default_timeout=120)


def test_reader_page_explains_each_pick():
    at = _page("for_you").run()
    assert not at.exception
    assert _markdown(at).count('class="book-card"') == 10  # general favourites

    at.selectbox(key="reader").set_value(at.selectbox(key="reader").options[0]).run()
    assert not at.exception
    assert [h.value for h in at.header] == ["What they loved", "What they might like next"]
    assert _markdown(at).count("Because they liked") == 10


def test_reader_picks_differ_between_readers():
    picks = []
    for option in (0, 50):
        at = _page("for_you").run()
        at.selectbox(key="reader").set_value(at.selectbox(key="reader").options[option]).run()
        cards = _markdown(at).split("What they might like next")[-1]
        picks.append(set(cards.split('class="book-title" title="')[1:]))
    assert len(picks[0] & picks[1]) < 5


def test_search_by_title_and_author():
    at = _page("search").run()
    at.text_input[0].input("tolkien").run()
    assert not at.exception
    assert "The Hobbit" in _markdown(at)

    at.text_input[0].input("(").run()  # not a regular expression
    assert not at.exception


def test_about_page_counts_come_from_the_data():
    at = _page("about").run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["3,217", "1,914", "42,331"]
