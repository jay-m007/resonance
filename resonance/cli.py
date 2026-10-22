"""resonance command-line tool."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import __version__
from .analyze import analyze
from .ingest import IngestError, load_tickets

BITEXT_URL = (
    "https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/main/data/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def _print_report(r: dict, limit: int) -> None:
    s = r["summary"]
    span = f", {s['first'][:10]} to {s['last'][:10]}" if s["first"] else ""
    print(f"Resonance v{r['meta']['version']}: {s['tickets']:,} tickets{span}, {r['meta']['duration_s']} s")
    print(f"{s['issues']} issues found, {s['rising']} rising, {s['unsorted']:,} tickets unsorted ({s['unsorted'] / s['tickets']:.0%})\n")
    print(f"{'#':>3}  {'impact':>7}  {'tickets':>7}  {'trend':<14} {'type':<10} issue")
    for i in r["issues"][:limit]:
        t = i["trend"]
        trend = {"rising": f"↑ {t['ratio']}x", "new": "new", "falling": f"↓ {t['ratio']}x", "steady": "steady", "n/a": "—"}[t["status"]]
        print(f"{i['rank']:>3}  {i['impact']:>7.0f}  {i['tickets']:>7,}  {trend:<14} {i['type']:<10} {i['name']}")
        if i["examples"]:
            print(f"{'':>34}e.g. \"{i['examples'][0]['text'][:90]}\"")
    if len(r["issues"]) > limit:
        print(f"\n…and {len(r['issues']) - limit} more in the JSON report.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="resonance", description="Turn support conversations into a ranked list of product issues.")
    p.add_argument("--version", action="version", version=f"resonance {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="find and rank the issues in a ticket export (CSV, JSON or JSONL)")
    a.add_argument("input")
    a.add_argument("-o", "--out", default="resonance-report.json", help="where to write the JSON report")
    a.add_argument("--embed", default="tfidf", choices=["tfidf", "st", "openai"], help="how to turn text into vectors (default: tfidf, no download)")
    a.add_argument("--k", type=int, help="force this many topics (k-means) instead of finding them automatically")
    a.add_argument("--text-col", help="name of the column holding the message text")
    a.add_argument("--time-col", help="name of the column holding the creation time")
    a.add_argument("--customer-col", help="name of the column identifying the customer")
    a.add_argument("--limit", type=int, help="only read the first N tickets")
    a.add_argument("--show", type=int, default=15, help="issues to print (default 15)")
    a.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("simulate", help="make a simulated month of tickets with a known spike (for demos and testing)")
    s.add_argument("input", help="a labelled dataset, e.g. the Bitext CSV")
    s.add_argument("-o", "--out", default="simulated-tickets.csv")
    s.add_argument("--spike", help="the intent that spikes in the final week (default: random)")
    s.add_argument("--factor", type=float, default=3.0, help="how much it spikes (default 3x)")
    s.add_argument("--days", type=int, default=28)
    s.add_argument("--per-day", type=int, default=120)
    s.add_argument("--seed", type=int, default=0)

    f = sub.add_parser("fetch-data", help="download the Bitext customer-support dataset (26,872 labelled messages, CDLA-Sharing 1.0)")
    f.add_argument("-o", "--out", default="data/bitext.csv")

    args = p.parse_args(argv)
    try:
        if args.cmd == "analyze":
            tickets = load_tickets(args.input, text_col=args.text_col, time_col=args.time_col, customer_col=args.customer_col, limit=args.limit)
            report = analyze(tickets, backend=args.embed, k=args.k, seed=args.seed)
            Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
            _print_report(report, args.show)
            print(f"\nReport written to {args.out}")
            return 0
        if args.cmd == "simulate":
            from .simulate import simulate, write_csv

            pool = load_tickets(args.input)
            tickets, spike = simulate(pool, days=args.days, per_day=args.per_day, spike=args.spike, spike_factor=args.factor, seed=args.seed)
            write_csv(tickets, args.out)
            print(f"Wrote {len(tickets):,} simulated tickets over {args.days} days to {args.out}. Spiking intent: {spike} (x{args.factor} in the final week).")
            print("This data is synthetic: real messages, made-up dates and customers.")
            return 0
        if args.cmd == "fetch-data":
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading the Bitext dataset to {out} …")
            urllib.request.urlretrieve(BITEXT_URL, out)
            print(f"Done ({out.stat().st_size / 1e6:.1f} MB). License: Community Data License Agreement – Sharing 1.0 (see the Bitext repository).")
            return 0
    except (IngestError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
