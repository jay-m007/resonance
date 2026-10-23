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
const str = (x: unknown) => typeof x === "string";
const STATUSES = ["rising", "falling", "steady", "new", "n/a"];
const TYPES = ["bug", "complaint", "request", "how-to", "other"];

/** Validates a report before the app relies on its shape, so a bad file shows a message instead of a blank page. */
export function isReport(x: unknown): x is Report {
  if (!isObj(x) || !isObj(x.meta) || x.meta.tool !== "resonance" || !isObj(x.summary) || !Array.isArray(x.issues)) return false;
  if (!num(x.summary.tickets) || (x.summary.tickets as number) < 1 || !num(x.summary.issues) || !num(x.summary.unsorted)) return false;
  let periodsLen: number | null = null;
  if (x.periods !== null && x.periods !== undefined) {
    const p = x.periods;
    if (!isObj(p) || (p.unit !== "day" && p.unit !== "week") || !Array.isArray(p.starts) || !p.starts.every(str) || !Array.isArray(p.totals) || !num(p.recent_window)) return false;
    periodsLen = p.starts.length;
    if ((p.recent_window as number) < 0 || (p.recent_window as number) > periodsLen) return false;
  }
  if (x.unsorted_examples !== undefined && !(Array.isArray(x.unsorted_examples) && x.unsorted_examples.every((e) => isObj(e) && str(e.text)))) return false;
  return (x.issues as unknown[]).every((i) => {
    if (!isObj(i) || !str(i.name) || !num(i.id) || !num(i.tickets) || !num(i.rank) || !num(i.impact) || !num(i.severity) || !num(i.share)) return false;
    if (!TYPES.includes(i.type as string) || !isObj(i.types) || !Object.values(i.types).every(num)) return false;
    if (!Array.isArray(i.keywords) || !i.keywords.every(str)) return false;
    if (!Array.isArray(i.examples) || !i.examples.every((e) => isObj(e) && str(e.text) && (str(e.id) || num(e.id)))) return false;
    if (i.brief !== undefined && !str(i.brief)) return false;
    if (i.customers !== null && i.customers !== undefined && !num(i.customers)) return false;
    if (i.label !== undefined && !(isObj(i.label) && (i.label.majority === null || str(i.label.majority)) && num(i.label.purity))) return false;
    if (!isObj(i.trend)) return false;
    const t = i.trend;
    if (!STATUSES.includes(t.status as string) || !Array.isArray(t.series) || !t.series.every(num) || !num(t.expected) || !num(t.latest)) return false;
    if (periodsLen !== null && t.series.length !== periodsLen) return false;
    return (t.ratio === null || num(t.ratio)) && (t.p_value === null || num(t.p_value));
  });
}
