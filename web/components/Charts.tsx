"use client";

import { useState } from "react";
import { dateLabel } from "@/lib/format";

/**
 * A tiny bar strip for table rows: daily or weekly volume, the recent window
 * in the accent colour. Single series, so no legend; the row says what it is.
 */
export function Spark({ series, recent, label }: { series: number[]; recent: number; label: string }) {
  const w = 120;
  const h = 28;
  const max = Math.max(1, ...series);
  const gap = 1;
  const bw = Math.max(1, (w - gap * (series.length - 1)) / Math.max(1, series.length));
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} className="spark">
      {series.map((v, i) => {
        const bh = Math.max(v > 0 ? 1.5 : 0, (v / max) * (h - 2));
        const isRecent = i >= series.length - recent;
        return <rect key={i} x={i * (bw + gap)} y={h - bh} width={bw} height={bh} rx={Math.min(1.5, bw / 2)} className={isRecent ? "spark__recent" : "spark__bar"} />;
      })}
    </svg>
  );
}

/**
 * Volume over time for one issue: bars per period, the recent window
 * highlighted, and a dashed line at the level expected from the earlier
 * periods (expected ÷ window, per period). Hover or focus a bar for its value.
 */
export function VolumeChart(props: { series: number[]; starts: string[]; unit: "day" | "week"; recent: number; expected: number; height?: number }) {
  const { series, starts, unit, recent } = props;
  const [hover, setHover] = useState<number | null>(null);
  const W = 640;
  const H = props.height ?? 200;
  const pad = { l: 34, r: 8, t: 12, b: 26 };
  const perPeriod = recent > 0 ? props.expected / recent : 0;
  const max = Math.max(1, ...series, perPeriod) * 1.1;
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const bw = iw / Math.max(1, series.length);
  const y = (v: number) => pad.t + ih - (v / max) * ih;
  const ticks = [0, Math.round(max / 2), Math.round(max * 0.95)].filter((v, i, a) => a.indexOf(v) === i);
  const recentStart = series.length - recent;
  const xLabels = series.map((_, i) => i).filter((i) => i === 0 || i === recentStart || i === series.length - 1);

  return (
    <div className="vchart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Tickets per ${unit}. Recent ${recent} ${unit}s highlighted; dashed line is the expected level.`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="vchart__grid" />
            <text x={pad.l - 6} y={y(t)} className="vchart__tick" textAnchor="end" dominantBaseline="middle">
              {t}
            </text>
          </g>
        ))}
        <rect x={pad.l + recentStart * bw} y={pad.t} width={recent * bw} height={ih} className="vchart__window" />
        {series.map((v, i) => (
          <rect
            key={i}
            x={pad.l + i * bw + 1}
            y={y(v)}
            width={Math.max(1, bw - 2)}
            height={pad.t + ih - y(v)}
            rx={Math.min(3, bw / 3)}
            className={`${i >= recentStart ? "vchart__recent" : "vchart__bar"}${hover === i ? " vchart__hover" : ""}`}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {perPeriod > 0 && <line x1={pad.l + recentStart * bw} x2={W - pad.r} y1={y(perPeriod)} y2={y(perPeriod)} className="vchart__expected" />}
        {xLabels.map((i) => (
          <text key={i} x={pad.l + i * bw + bw / 2} y={H - 8} className="vchart__tick" textAnchor={i === 0 ? "start" : i === series.length - 1 ? "end" : "middle"}>
            {starts[i] ? dateLabel(starts[i], unit) : ""}
          </text>
        ))}
      </svg>
      <div className="vchart__foot" aria-live="polite">
        {hover !== null ? (
          <span>
            <b className="mono">{series[hover]}</b> tickets, {unit} of {dateLabel(starts[hover], unit)}
          </span>
        ) : (
          <span className="vchart__legend">
            <span className="legend-item">
              <span className="key key--bar" aria-hidden="true" /> earlier
            </span>
            <span className="legend-item">
              <span className="key key--recent" aria-hidden="true" /> recent {recent} {unit}s
            </span>
            {perPeriod > 0 && (
              <span className="legend-item">
                <span className="key key--expected" aria-hidden="true" /> expected ({perPeriod.toFixed(1)} per {unit})
              </span>
            )}
          </span>
        )}
      </div>
    </div>
  );
}
