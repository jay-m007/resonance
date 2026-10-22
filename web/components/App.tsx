"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fmt, loudest, pct, STATUS, trendText, TYPE_ICON, TYPE_LABEL } from "@/lib/format";
import { isReport, type Issue, type IssueType, type Report } from "@/lib/types";
import { Spark, VolumeChart } from "./Charts";
import { Detail } from "./Detail";

type StatusFilter = "all" | "rising" | "bug" | "complaint";

export function App() {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [query, setQuery] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const accept = useCallback((data: unknown, source: string) => {
    if (!isReport(data)) {
      setError(`${source} isn't a Resonance report. Make one with: resonance analyze tickets.csv`);
      return;
    }
    setReport(data);
    setSelected(null);
    setFilter("all");
    setQuery("");
    setError(null);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const url = params.has("demo") ? "demo/report.json" : params.get("report") ?? "data/report.json";
    fetch(url)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && accept(d, url))
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, [accept]);

  const open = async (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    try {
      accept(JSON.parse(await f.text()), f.name);
    } catch {
      setError(`${f.name} isn't valid JSON.`);
    }
  };
  const demo = async () => {
    try {
      accept(await fetch("demo/report.json").then((r) => r.json()), "The demo report");
    } catch {
      setError("Couldn't load the demo. Open a report file instead.");
    }
  };

  const issues = useMemo(() => {
    if (!report) return [];
    const q = query.trim().toLowerCase();
    return report.issues.filter((i) => {
      if (filter === "rising" && i.trend.status !== "rising" && i.trend.status !== "new") return false;
      if ((filter === "bug" || filter === "complaint") && i.type !== filter) return false;
      if (q && !(i.name.toLowerCase().includes(q) || i.keywords.some((k) => k.includes(q)) || i.examples.some((e) => e.text.toLowerCase().includes(q)))) return false;
      return true;
    });
  }, [report, filter, query]);

  const drop = {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(true);
    },
    onDragLeave: (e: React.DragEvent) => e.currentTarget === e.target && setDragging(false),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      open(e.dataTransfer.files);
    },
  };
  const input = <input ref={fileInput} type="file" accept=".json,application/json" className="visually-hidden" onChange={(e) => open(e.target.files).then(() => (e.target.value = ""))} />;

  if (loading) return <div className="shell" aria-busy="true" />;

  if (!report) {
    return (
      <div className={`shell${dragging ? " shell--drop" : ""}`} {...drop}>
        {input}
        <main className="empty">
          <div className="empty__inner">
            <Wave />
            <h1>Find the problems hiding in your support queue.</h1>
            <p className="empty__lede">
              Resonance reads thousands of support conversations, groups them into issues, and ranks what to fix first: how many customers are affected, how upset they are, and what&apos;s growing fastest.
            </p>
            <div className="empty__actions">
              <button className="btn btn--primary btn--big" onClick={demo}>
                See a demo month
              </button>
              <button className="btn btn--big" onClick={() => fileInput.current?.click()}>
                Open a report
              </button>
            </div>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            <div className="empty__how">
              <h2>Analyse your own tickets</h2>
              <pre className="mono">resonance analyze tickets.csv -o report.json</pre>
              <p>Then open or drop report.json here. It stays in your browser; nothing is uploaded.</p>
            </div>
          </div>
        </main>
      </div>
    );
  }

  const top = loudest(report);
  const s = report.summary;
  const sel = report.issues.find((i) => i.id === selected) ?? null;
  const recent = report.periods?.recent_window ?? 0;

  return (
    <div className={`shell${dragging ? " shell--drop" : ""}`} {...drop}>
      {input}
      <header className="topbar">
        <div className="brand">
          <Wave small />
          <span>Resonance</span>
        </div>
        <div className="topbar__meta">
          {fmt(s.tickets)} tickets
          {s.first && s.last && (
            <span className="muted">
              {" "}
              · {new Date(s.first).toLocaleDateString()} – {new Date(s.last).toLocaleDateString()}
            </span>
          )}
        </div>
        <div className="topbar__actions">
          <button className="btn" onClick={() => fileInput.current?.click()}>
            Open report
          </button>
          <ThemeToggle />
        </div>
      </header>

      {report.meta.synthetic && (
        <div className="notice" role="note">
          Demo data: real support messages from the public Bitext dataset, with simulated dates and customers. One topic was made to spike in the final week.
        </div>
      )}
      {error && (
        <div className="notice notice--error" role="alert">
          {error}
          <button className="btn btn--ghost" onClick={() => setError(null)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}

      <main className={`page${sel ? " page--detail" : ""}`}>
        <div className="main">
          {top && (
            <section className="hero" aria-labelledby="hero-title">
              <div className="hero__text">
                <p className="hero__kicker">
                  {top.trend.status === "rising" || top.trend.status === "new" ? "Loudest problem right now" : "Biggest issue"}
                </p>
                <h1 id="hero-title" className="hero__title">
                  {top.name}
                </h1>
                <p className="hero__line">
                  <StatusBadge issue={top} /> {trendText(top, report)}. {fmt(top.tickets)} tickets
                  {top.customers ? `, ${fmt(top.customers)} customers` : ""}.
                </p>
                {top.examples[0] && <blockquote className="hero__quote">“{top.examples[0].text}”</blockquote>}
                <button className="btn btn--primary" onClick={() => setSelected(top.id)}>
                  Open this issue
                </button>
              </div>
              {report.periods && (
                <div className="hero__chart">
                  <VolumeChart series={top.trend.series} starts={report.periods.starts} unit={report.periods.unit} recent={recent} expected={top.trend.expected} height={190} />
                </div>
              )}
            </section>
          )}

          <section className="numbers" aria-label="Summary">
            <Num label="Issues found" value={fmt(s.issues)} />
            <Num label="Rising" value={fmt(s.rising)} tone={s.rising ? "critical" : undefined} />
            <Num label="Customers" value={s.customers ? fmt(s.customers) : "—"} />
            <Num label="Not sorted into an issue" value={pct(s.unsorted / s.tickets)} hint={`${fmt(s.unsorted)} tickets that didn't clearly match any issue`} />
          </section>

          <section className="list" aria-labelledby="list-title">
            <div className="list__head">
              <h2 id="list-title">Issues, ranked by impact</h2>
              <div className="filters">
                <div className="segmented" role="radiogroup" aria-label="Filter issues">
                  {(
                    [
                      ["all", "All"],
                      ["rising", "Rising"],
                      ["bug", "Bugs"],
                      ["complaint", "Complaints"],
                    ] as const
                  ).map(([id, label]) => (
                    <button key={id} role="radio" aria-checked={filter === id} onClick={() => setFilter(id)}>
                      {label}
                    </button>
                  ))}
                </div>
                <label className="visually-hidden" htmlFor="search">
                  Search issues
                </label>
                <input id="search" className="search" placeholder="Search issues and tickets…" value={query} onChange={(e) => setQuery(e.target.value)} />
              </div>
            </div>
            <p className="list__how">Impact = tickets × (1 + severity) × trend boost (up to 3× for rising issues).</p>

            {issues.length === 0 ? (
              <p className="muted">No issues match.</p>
            ) : (
              <table className="table">
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      #
                    </th>
                    <th scope="col">Issue</th>
                    <th scope="col">Type</th>
                    <th scope="col">Trend</th>
                    <th scope="col" className="num">
                      Tickets
                    </th>
                    <th scope="col" className="num">
                      Impact
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {issues.map((i) => (
                    <tr key={i.id} className={selected === i.id ? "is-selected" : ""}>
                      <td className="num mono muted">{i.rank}</td>
                      <td>
                        <button className="rowlink" onClick={() => setSelected(i.id)} aria-label={`Open ${i.name}`}>
                          <span className="rowlink__name">{i.name}</span>
                          <span className="rowlink__kw">{i.keywords.slice(0, 4).join(", ")}</span>
                        </button>
                      </td>
                      <td>
                        <TypeChip type={i.type} />
                      </td>
                      <td>
                        <div className="trendcell">
                          {report.periods && <Spark series={i.trend.series} recent={recent} label={`${i.name}: volume over time`} />}
                          <StatusBadge issue={i} compact />
                        </div>
                      </td>
                      <td className="num mono">{fmt(i.tickets)}</td>
                      <td className="num mono">{fmt(Math.round(i.impact))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </div>

        {sel && <Detail issue={sel} report={report} onClose={() => setSelected(null)} />}
      </main>
    </div>
  );
}

export function StatusBadge({ issue, compact }: { issue: Issue; compact?: boolean }) {
  const st = STATUS[issue.trend.status];
  if (issue.trend.status === "steady" && compact) return <span className="badge badge--neutral">Steady</span>;
  if (issue.trend.status === "n/a") return null;
  return (
    <span className={`badge badge--${st.tone}`}>
      <span aria-hidden="true">{st.icon}</span> {st.label}
      {(issue.trend.status === "rising" || issue.trend.status === "falling") && issue.trend.ratio !== null && <span className="mono"> {issue.trend.ratio}×</span>}
    </span>
  );
}

export function TypeChip({ type }: { type: IssueType }) {
  return (
    <span className={`type type--${type}`}>
      <span className="type__icon" aria-hidden="true">
        {TYPE_ICON[type]}
      </span>
      {TYPE_LABEL[type]}
    </span>
  );
}

function Num({ label, value, tone, hint }: { label: string; value: string; tone?: string; hint?: string }) {
  return (
    <div className="numcard" title={hint}>
      <span className={`numcard__value mono${tone ? ` tone-${tone}` : ""}`}>{value}</span>
      <span className="numcard__label">{label}</span>
    </div>
  );
}

function Wave({ small }: { small?: boolean }) {
  // A short signal: noise, then one tall peak. The product in one mark.
  const s = small ? 20 : 44;
  return (
    <svg width={s} height={s} viewBox="0 0 24 24" aria-hidden="true" className="wave">
      {[3, 5, 4, 6, 20, 7, 4, 5, 3].map((h, i) => (
        <rect key={i} x={1 + i * 2.5} y={(24 - h) / 2} width="1.6" height={h} rx="0.8" fill={i === 4 ? "var(--accent)" : "currentColor"} opacity={i === 4 ? 1 : 0.45} />
      ))}
    </svg>
  );
}

function ThemeToggle() {
  const [dark, setDark] = useState(true);
  useEffect(() => {
    const t = document.documentElement.dataset.theme;
    setDark(t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);
  return (
    <button
      className="btn btn--icon"
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      onClick={() => {
        const next = dark ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("rs-theme", next);
        } catch {
          /* not remembered in private mode */
        }
        setDark(!dark);
      }}
    >
      {dark ? "☀" : "☾"}
    </button>
  );
}
