export type IssueType = "bug" | "complaint" | "request" | "how-to" | "other";
export type TrendStatus = "rising" | "falling" | "steady" | "new" | "n/a";

export interface Example {
  id: string;
  text: string;
}

export interface Issue {
  id: number;
  rank: number;
  name: string;
  keywords: string[];
  tickets: number;
  share: number;
  customers: number | null;
  types: Record<IssueType, number>;
  type: IssueType;
  severity: number;
  impact: number;
  cohesion: number;
  trend: { status: TrendStatus; series: number[]; latest: number; expected: number; ratio: number | null; p_value: number | null };
  examples: Example[];
  ticket_ids: string[];
  label?: { majority: string | null; purity: number };
  brief?: string;
}

export interface Report {
  meta: { tool: "resonance"; version: string; generated_at: string; duration_s: number; embedding: string; clustering: string; synthetic?: boolean; source?: string };
  summary: { tickets: number; issues: number; unsorted: number; rising: number; first: string | null; last: string | null; customers: number | null };
  periods: { unit: "day" | "week"; starts: string[]; totals: number[]; recent_window: number } | null;
  issues: Issue[];
  unsorted_examples: Example[];
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);
const num = (x: unknown) => typeof x === "number" && Number.isFinite(x);

/** Validates a report before the app relies on its shape. */
export function isReport(x: unknown): x is Report {
  if (!isObj(x) || !isObj(x.meta) || x.meta.tool !== "resonance" || !isObj(x.summary) || !Array.isArray(x.issues)) return false;
  if (!num(x.summary.tickets) || !num(x.summary.issues) || !num(x.summary.unsorted)) return false;
  if (x.periods !== null && x.periods !== undefined) {
    const p = x.periods as Record<string, unknown>;
    if (!Array.isArray(p.starts) || !Array.isArray(p.totals) || !num(p.recent_window)) return false;
  }
  return (x.issues as unknown[]).every((i) => {
    if (!isObj(i) || typeof i.name !== "string" || !num(i.tickets) || !num(i.rank) || !num(i.impact) || !num(i.severity)) return false;
    if (!Array.isArray(i.keywords) || !Array.isArray(i.examples) || !isObj(i.types) || !isObj(i.trend)) return false;
    const t = i.trend as Record<string, unknown>;
    return typeof t.status === "string" && Array.isArray(t.series) && (t.series as unknown[]).every(num);
  });
}
