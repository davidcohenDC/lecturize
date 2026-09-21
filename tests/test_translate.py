import pytest

from lecturize.translate import Catalog, NoRouteError, Package, Translator

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
    with pytest.raises(NoRouteError, match="from 'la'"):
        catalog.route("la", "it")


def test_batch_keeps_one_per_input(catalog, monkeypatch):
    monkeypatch.setattr(catalog, "install", lambda pkg, progress=None: None)
    monkeypatch.setattr("lecturize.translate.Step.__init__", lambda self, d, device="cpu": None)
    monkeypatch.setattr(
        "lecturize.translate.Step.__call__", lambda self, texts: [t.upper() for t in texts]
    )
    tr = Translator("en", "it", catalog=catalog)
    out = tr.batch(["One. Two.", "", "Three."])
    assert out == ["ONE. TWO.", "", "THREE."]
    assert tr.describe() == "en -> it"


def test_index_cache_is_used_and_recovers_from_a_corrupt_file(tmp_path, monkeypatch):
    import json
    import urllib.request

    payload = json.dumps(
        [{"from_code": "en", "to_code": "it", "package_version": "1.0", "links": ["u"]}]
    )
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=0):
        calls.append(req.full_url)
        return Resp(payload.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    cat = Catalog(tmp_path)
    assert [p.to_code for p in cat.packages()] == ["it"]
    assert [p.to_code for p in cat.packages()] == ["it"]
    assert len(calls) == 1  # second call served from disk
    (tmp_path / "index.json").write_text("{truncated", encoding="utf-8")
    assert [p.to_code for p in cat.packages()] == ["it"]
    assert len(calls) == 2  # corrupt cache replaced


def test_offline_without_cache_is_a_clear_error(tmp_path, monkeypatch):
    import urllib.request

    def down(req, timeout=0):
        raise OSError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    with pytest.raises(NoRouteError, match="offline"):
        Catalog(tmp_path).packages()
