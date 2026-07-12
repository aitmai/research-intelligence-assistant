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
