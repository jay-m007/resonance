import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from resonance.analyze import analyze, impact
from resonance.cli import main
from resonance.cluster import UNSORTED, cluster
from resonance.describe import issue_type, keywords, severity
from resonance.ingest import IngestError, Ticket, clean, load_tickets, parse_time, redact
from resonance.simulate import simulate
from resonance.trends import trends

TOPICS = {
    "refund": ["where is my refund", "i want my money back for order", "refund still not received after two weeks", "how long does a refund take", "refund for cancelled order please"],
    "password": ["i forgot my password", "cannot reset password link expired", "password reset email never arrives", "how do i change my password", "locked out of my account password"],
    "shipping": ["my package has not arrived", "where is my delivery", "tracking number shows no update", "shipment delayed again", "delivery to wrong address"],
}


def pool(n_each=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for label, texts in TOPICS.items():
        for i in range(n_each):
            t = texts[i % len(texts)] + " " + rng.choice(["", "please", "today", "thanks", "asap"])
            out.append(Ticket(id=f"{label}-{i}", text=t, label=label, customer=f"c{i % 17}"))
    return out


def test_parse_time_formats():
    assert parse_time("2026-09-01T10:00:00Z") == datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    assert parse_time("1756720800").year == 2025
    assert parse_time("1756720800000").year == 2025
    assert parse_time("01/09/2026").tzinfo is not None
    assert parse_time("") is None and parse_time("not a date") is None


def test_load_csv_and_json(tmp_path):
    c = tmp_path / "t.csv"
    c.write_text("ticket_id,Message,created_at,customer_id\n1,Refund please,2026-09-01,a\n2,,2026-09-02,b\n3,Where is it,2026-09-03,c\n")
    ts = load_tickets(c)
    assert [t.text for t in ts] == ["Refund please", "Where is it"]  # empty text skipped
    assert ts[0].id == "1" and ts[0].customer == "a" and ts[0].created.day == 1
    j = tmp_path / "t.json"
    j.write_text(json.dumps({"tickets": [{"body": "hi", "date": "2026-01-01"}]}))
    assert load_tickets(j)[0].text == "hi"
    with pytest.raises(IngestError, match="no text column"):
        (tmp_path / "x.csv").write_text("a,b\n1,2\n")
        load_tickets(tmp_path / "x.csv")


def test_redact_and_clean():
    s = "Order {{Order Number}} for jay@example.com, call +91 98765 43210, see https://x.io/a"
    r = redact(s)
    assert "[order number]" in r and "[email]" in r and "[phone]" in r and "[link]" in r
    assert "example.com" not in r and "98765" not in r
    assert clean(s) == "order for , call , see"


def test_issue_type_and_severity():
    assert issue_type("The export button is not working") == "bug"
    assert issue_type("this is unacceptable, worst service") == "complaint"
    assert issue_type("I would like an option to export PDF") == "request"
    assert issue_type("how do I add a teammate?") == "how-to"
    assert severity("thanks") == 0
    assert severity("URGENT!!! STILL NOT FIXED, THIS IS RIDICULOUS") > 0.8


def test_clustering_recovers_topics():
    tickets = pool()
    from resonance.embed import embed

    X = embed([clean(t.text) for t in tickets])
    cl = cluster(X)
    assigned = cl.labels[cl.labels != UNSORTED]
    assert len(set(assigned)) >= 3
    # Each discovered topic should be dominated by one true label.
    for j in set(assigned):
        labels = [tickets[i].label for i in np.where(cl.labels == j)[0]]
        assert max(labels.count(l) for l in set(labels)) / len(labels) >= 0.8


def test_keywords_name_topics():
    kw = keywords([TOPICS["refund"] * 3, TOPICS["password"] * 3, TOPICS["shipping"] * 3])
    assert "refund" in kw[0] and "password" in kw[1]


def test_trend_detects_spike_without_false_alarms():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    times, labels = [], []
    rng = np.random.default_rng(1)
    for d in range(28):
        for topic in range(4):
            n = 20 if not (topic == 2 and d >= 21) else 60
            n = int(rng.poisson(n))
            times += [start + timedelta(days=d, hours=int(h)) for h in rng.integers(0, 24, n)]
            labels += [topic] * n
    tr, info = trends(np.array(labels), times, 4)
    assert [t.status for t in tr] == ["steady", "steady", "rising", "steady"]
    assert tr[2].ratio > 2.5 and info["unit"] == "day"


def test_no_dates_means_no_trends():
    tr, info = trends(np.array([0, 1, 0]), [None, None, None], 2)
    assert info is None and all(t.status == "n/a" for t in tr)


def test_impact_formula():
    assert impact(100, 0.0, "steady", 1.0) == 100
    assert impact(100, 0.5, "steady", 1.0) == 150
    assert impact(100, 0.0, "rising", 3.0) == 250
    assert impact(100, 0.0, "rising", 10.0) == 300  # capped


def test_end_to_end_ranks_the_spike_first():
    tickets, spike = simulate(pool(80), days=28, per_day=60, spike="password", seed=2)
    r = analyze(tickets)
    top = r["issues"][0]
    assert top["trend"]["status"] == "rising"
    assert top["label"]["majority"] == "password"
    rising = [i for i in r["issues"] if i["trend"]["status"] == "rising"]
    assert rising and all(i["label"]["majority"] == "password" for i in rising)  # no false alarms
    assert all("@" not in e["text"] for i in r["issues"] for e in i["examples"])
    json.dumps(r)  # serialisable


def test_cli_analyze_and_simulate(tmp_path, capsys):
    src = tmp_path / "pool.csv"
    rows = ["id,text,intent"] + [f"{t.id},{t.text},{t.label}" for t in pool(60)]
    src.write_text("\n".join(rows))
    sim = tmp_path / "sim.csv"
    assert main(["simulate", str(src), "-o", str(sim), "--spike", "refund", "--per-day", "40"]) == 0
    out = tmp_path / "r.json"
    assert main(["analyze", str(sim), "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "rising" in printed or "↑" in printed
    assert json.loads(out.read_text())["meta"]["tool"] == "resonance"
    assert main(["analyze", str(tmp_path / "missing.csv")]) == 2
