"use client";

import { useEffect, useRef } from "react";
import { fmt, pct, trendText, TYPE_LABEL } from "@/lib/format";
import type { Issue, IssueType, Report } from "@/lib/types";
import { StatusBadge, TypeChip } from "./App";
import { VolumeChart } from "./Charts";

const TYPES: IssueType[] = ["bug", "complaint", "request", "how-to", "other"];

/** Everything about one issue: its trend, what kinds of tickets it holds, and what customers actually wrote. */
export function Detail({ issue, report, onClose }: { issue: Issue; report: Report; onClose: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  // Move focus into the panel when a different issue opens, and give it back when the panel closes.
  // Depends only on the issue, so typing in the search box doesn't pull focus away.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    return () => {
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [issue.id]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && close.current();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, []);
  const totalTyped = TYPES.reduce((a, t) => a + (issue.types[t] ?? 0), 0) || 1;

  return (
    <aside className="detail" aria-labelledby="detail-title" tabIndex={-1} ref={ref}>
      <div className="detail__head">
        <div>
          <p className="detail__rank mono">#{issue.rank}</p>
          <h2 id="detail-title">{issue.name}</h2>
        </div>
        <button className="btn btn--icon" onClick={onClose} aria-label="Close issue">
          ✕
        </button>
      </div>

      <div className="detail__badges">
        <TypeChip type={issue.type} />
        <StatusBadge issue={issue} />
      </div>

      <dl className="facts">
        <div>
          <dt>Tickets</dt>
          <dd className="mono">
            {fmt(issue.tickets)} <span className="muted">({pct(issue.share)})</span>
          </dd>
        </div>
        {issue.customers !== null && (
          <div>
            <dt>Customers</dt>
            <dd className="mono">{fmt(issue.customers)}</dd>
          </div>
        )}
        <div>
          <dt>Severity</dt>
          <dd>
            <span className="meter" aria-label={`Severity ${Math.round(issue.severity * 100)} out of 100`}>
              <span style={{ width: `${Math.max(2, issue.severity * 100)}%` }} />
            </span>
            <span className="mono">{Math.round(issue.severity * 100)}</span>
          </dd>
        </div>
        <div>
          <dt>Impact</dt>
          <dd className="mono">{fmt(Math.round(issue.impact))}</dd>
        </div>
      </dl>

      {report.periods && (
        <section>
          <h3>Volume</h3>
          <p className="detail__sub">{trendText(issue, report)}.</p>
          <VolumeChart series={issue.trend.series} starts={report.periods.starts} unit={report.periods.unit} recent={report.periods.recent_window} expected={issue.trend.expected} />
        </section>
      )}

      {issue.brief && (
        <section className="brief">
          <h3>AI product brief</h3>
          <div className="brief__text">{issue.brief}</div>
          <p className="muted small">Written by an AI model from the tickets below. Check it before acting.</p>
        </section>
      )}

      <section>
        <h3>Keywords</h3>
        <ul className="chips">
          {issue.keywords.map((k) => (
            <li key={k} className="chip mono">
              {k}
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>What kind of tickets</h3>
        <ul className="typebars">
          {TYPES.filter((t) => issue.types[t]).map((t) => (
            <li key={t}>
              <span className="typebars__label">{TYPE_LABEL[t]}</span>
              <span className="typebars__track">
                <span className="typebars__fill" style={{ width: `${(issue.types[t] / totalTyped) * 100}%` }} />
              </span>
              <span className="mono typebars__n">{fmt(issue.types[t])}</span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3>What customers wrote</h3>
        <ul className="quotes">
          {issue.examples.map((e) => (
            <li key={e.id}>
              <q>{e.text}</q>
              <span className="mono muted small">#{e.id}</span>
            </li>
          ))}
        </ul>
        <p className="muted small">The tickets closest to the centre of this issue, with personal details removed.</p>
      </section>

      {issue.label?.majority && (
        <section>
          <h3>Matches known label</h3>
          <p>
            <span className="mono">{issue.label.majority}</span> for {pct(issue.label.purity)} of these tickets. Shown because the data came with labels; Resonance didn&apos;t use them to find the issue.
          </p>
        </section>
      )}
    </aside>
  );
}
