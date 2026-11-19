import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_runs_and_recommends():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception

    at.selectbox[1].set_value("Jurassic Park").run()
    next(b for b in at.button if b.key == "recommendations_knn").click().run()
    assert not at.exception
    assert any("The Lost World" in m.value for m in at.markdown)

    next(b for b in at.button if b.key == "recommendations_svd").click().run()
    assert not at.exception

    at.text_input[0].input("harry").run()
    assert not at.exception
    assert len(at.dataframe[0].value) > 0
