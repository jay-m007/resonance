"use client";

/** Shown if rendering fails, e.g. a report that passed validation but is still inconsistent. */
export default function Error({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <main className="empty">
      <div className="empty__inner">
        <h1>That report couldn&apos;t be shown.</h1>
        <p className="empty__lede">{error.message}</p>
        <div className="empty__actions">
          <button className="btn btn--primary btn--big" onClick={() => (window.location.href = window.location.pathname)}>
            Start over
          </button>
          <button className="btn btn--big" onClick={reset}>
            Try again
          </button>
        </div>
        <div className="empty__how">
          <p>Re-create it with the current version: resonance analyze tickets.csv -o report.json</p>
        </div>
      </div>
    </main>
  );
}
