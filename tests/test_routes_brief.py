import io


def test_brief_get_renders_form(client):
    resp = client.get("/brief")
    assert resp.status_code == 200
    assert b"Opportunity Brief" in resp.data


def test_brief_post_without_topic_redirects_with_flash(client):
    data = {"documents": (io.BytesIO(b"some text"), "doc.txt")}
    resp = client.post("/brief", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert b"Please provide a topic" in resp.data


def test_brief_post_creates_brief_and_renders_result(client):
    data = {
        "topic": "Embedded insurance for gig platforms",
        "documents": (io.BytesIO(b"Some market research about embedded insurance and gig platforms."), "research.txt"),
    }
    resp = client.post("/brief", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Embedded insurance for gig platforms" in resp.data
    assert b"Fake market size for testing" in resp.data
    assert b"confidence: medium" in resp.data


def test_briefs_list_shows_created_brief(client):
    data = {
        "topic": "Test topic for listing",
        "documents": (io.BytesIO(b"Some research text."), "research.txt"),
    }
    client.post("/brief", data=data, content_type="multipart/form-data")

    resp = client.get("/briefs")
    assert resp.status_code == 200
    assert b"Test topic for listing" in resp.data


def test_brief_dedupes_identical_document_content(client):
    """Uploading the exact same text twice should hit the content-hash cache,
    not create a second Document row with duplicate chunks."""
    from models import Document
    import app as app_module

    same_text = b"Identical research content for dedupe test."
    data1 = {"topic": "Dedupe test 1", "documents": (io.BytesIO(same_text), "a.txt")}
    data2 = {"topic": "Dedupe test 2", "documents": (io.BytesIO(same_text), "b.txt")}

    client.post("/brief", data=data1, content_type="multipart/form-data")
    client.post("/brief", data=data2, content_type="multipart/form-data")

    db = app_module.SessionLocal()
    count = db.query(Document).count()
    assert count == 1  # same content hash -> reused, not duplicated


def test_brief_blocked_once_daily_extraction_cap_reached(client, monkeypatch):
    """
    Safety-net regression test: once MAX_DAILY_EXTRACTIONS runs have
    happened in the trailing 24h, further /brief requests must be blocked
    with a clear message instead of silently making more Claude API calls.
    """
    from config import Config
    import app as app_module

    monkeypatch.setattr(Config, "MAX_DAILY_EXTRACTIONS", 2)

    def post_once(n):
        return client.post(
            "/brief",
            data={"topic": f"Cap test {n}", "documents": (io.BytesIO(b"Some text."), f"doc{n}.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

    post_once(1)
    post_once(2)
    resp = post_once(3)  # should now be blocked — cap is 2

    assert b"Daily extraction limit reached" in resp.data

    from models import RunHistory
    db = app_module.SessionLocal()
    # exactly 2 real runs happened; the 3rd request never reached extraction
    assert db.query(RunHistory).count() == 2
