import type { Issue, IssueType, Report, TrendStatus } from "./types";

export const TYPE_LABEL: Record<IssueType, string> = { bug: "Bug", complaint: "Complaint", request: "Request", "how-to": "How-to", other: "Other" };
export const TYPE_ICON: Record<IssueType, string> = { bug: "✕", complaint: "!", request: "+", "how-to": "?", other: "·" };

export const STATUS: Record<TrendStatus, { label: string; icon: string; tone: string }> = {
  rising: { label: "Rising", icon: "▲", tone: "critical" },
  new: { label: "New", icon: "✦", tone: "warning" },
  falling: { label: "Falling", icon: "▼", tone: "good" },
  steady: { label: "Steady", icon: "—", tone: "neutral" },
  "n/a": { label: "No dates", icon: "", tone: "neutral" },
};

export const fmt = (n: number) => n.toLocaleString();
export const pct = (x: number) => `${Math.round(x * 100)}%`;

/** "▲ 2.7× this week" style text for a trend. */
export function trendText(issue: Issue, report: Report): string {
  const t = issue.trend;
  const unit = report.periods?.unit === "week" ? `last ${report.periods.recent_window} weeks` : `last ${report.periods?.recent_window ?? 7} days`;
  if (t.status === "rising" || t.status === "falling") return `${t.ratio}× expected in the ${unit}`;
  if (t.status === "new") return `new in the ${unit}`;
  if (t.status === "steady") return t.ratio ? `${t.ratio}× expected, within normal range` : "steady";
  return "no timestamps in the data";
}

export function dateLabel(iso: string, unit: "day" | "week"): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, unit === "day" ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "2-digit" });
}

export function loudest(report: Report): Issue | null {
  const rising = report.issues.filter((i) => i.trend.status === "rising" || i.trend.status === "new");
  if (rising.length) return [...rising].sort((a, b) => b.impact - a.impact)[0];
  return report.issues[0] ?? null;
}
