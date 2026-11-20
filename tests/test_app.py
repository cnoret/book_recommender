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


def test_branding_and_about_page():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert at.title[0].value == "Next Chapter"
    assert "Find the book that comes after the one you just closed." in _markdown(at)
