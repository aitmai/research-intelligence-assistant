def test_monitor_get_renders_form(client):
    resp = client.get("/monitor")
    assert resp.status_code == 200
    assert b"Signal Monitor" in resp.data


def test_monitor_post_without_ticker_redirects_with_flash(client):
    resp = client.post("/monitor", data={}, follow_redirects=True)
    assert b"Please provide a ticker" in resp.data


def test_monitor_post_with_sample_data_creates_signal(client):
    resp = client.post("/monitor", data={"ticker": "demo", "use_sample": "on"})
    assert resp.status_code == 200
    assert b"Signal Timeline: DEMO" in resp.data
    assert b"lowered" in resp.data  # from FakeExtractor
    assert b"fake key quote" in resp.data


def test_monitor_timeline_view_shows_past_signals(client):
    client.post("/monitor", data={"ticker": "demo", "use_sample": "on"})
    resp = client.get("/monitor/DEMO")
    assert resp.status_code == 200
    assert b"Signal Timeline: DEMO" in resp.data


def test_monitor_timeline_empty_for_unknown_ticker(client):
    resp = client.get("/monitor/UNKNOWN")
    assert resp.status_code == 200
    assert b"No signals yet for UNKNOWN" in resp.data
