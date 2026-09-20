import pytest

from lecturize.translate import Catalog, Package, Translator

PKGS = [
    Package("en", "it", "1.0", "u1"),
    Package("it", "en", "1.0", "u2"),
    Package("fr", "en", "1.0", "u3"),
    Package("en", "fr", "1.0", "u4"),
]


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    cat = Catalog(tmp_path)
    monkeypatch.setattr(cat, "packages", lambda refresh=False: PKGS)
    return cat


def test_direct_route(catalog):
    assert [p.to_code for p in catalog.route("en", "it")] == ["it"]


def test_pivot_route(catalog):
    route = catalog.route("fr", "it")
    assert [(p.from_code, p.to_code) for p in route] == [("fr", "en"), ("en", "it")]


def test_same_language_is_empty(catalog):
    assert catalog.route("it", "it") == []


def test_missing_route(catalog):
    with pytest.raises(LookupError, match="la"):
        catalog.route("la", "it")


def test_batch_keeps_one_output_per_input(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "install", lambda pkg, progress=None: None)
    monkeypatch.setattr("lecturize.translate.Step.__init__", lambda self, d, device="cpu": None)
    monkeypatch.setattr(
        "lecturize.translate.Step.__call__", lambda self, texts: [t.upper() for t in texts]
    )
    tr = Translator("en", "it", catalog=catalog)
    out = tr.batch(["One. Two.", "", "Three."])
    assert out == ["ONE. TWO.", "", "THREE."]
    assert tr.describe() == "en -> it"
