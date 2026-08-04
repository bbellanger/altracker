#!/usr/bin/env python3
"""
Aim Lab -> Apex Legends training analysis pipeline.

Two commands:

    python aimlab_pipeline.py ingest <folder_with_new_csv_exports>
        Reads every *.csv in the given folder (fresh Aim Lab exports),
        normalizes them, and appends any session not already recorded
        into master_aimlab_long.csv (deduped on datetime + task variant).

    python aimlab_pipeline.py report
        Reads master_aimlab_long.csv (+ apex_sessions.csv if present)
        and writes report.html with trend charts, a cross-task
        correlation matrix, a "leverage" ranking of which drills are
        improving fastest / relate most to other skills, and (once you
        have Apex data logged) correlations against your actual
        in-game results.

Re-run `ingest` every time you export a fresh batch from Aim Lab
(weekly is a good cadence). Re-run `report` any time you want an
updated read on your training.
"""

import sys
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
MASTER_PATH = BASE_DIR / "master_aimlab_long.csv"
APEX_PATH = BASE_DIR / "apex_sessions.csv"
REPORT_PATH = BASE_DIR / "report.html"

# Columns from Aim Lab exports that are metadata, not performance metrics
META_COLS = {"createDate", "taskName", "mode", "weaponName", "map", "version"}

# The single metric every task type shares, used for cross-task comparison
PRIMARY_METRIC = "score"


def task_variant_from_filename(path):
    """CircleshotUltimate.csv -> CircleshotUltimate"""
    return Path(path).stem


def read_aimlab_csv(path):
    """
    Read an Aim Lab export tolerantly. Two known quirks of these files:
      1. Every data row ends with a trailing comma (one extra empty field
         vs. the header).
      2. Aim Lab sometimes changes a task's export schema over time, so a
         single file can have a short header from when it was first created
         but later rows with more fields (new columns Aim Lab added since).
    Rather than let pandas' strict tokenizer error out on ragged rows, we
    parse by hand: keep the first len(header) fields of every row (extra
    trailing fields — whether blank padding or newer unrecognized columns —
    are dropped), and pad short rows with blanks.
    """
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    ncols = len(header)
    data_rows = []
    for r in rows[1:]:
        if not r or all(c == "" for c in r):
            continue
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            r = r[:ncols]
        data_rows.append(r)

    df = pd.DataFrame(data_rows, columns=header)
    for c in df.columns:
        if c not in META_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ingest(folder):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        return

    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH, parse_dates=["session_datetime"])
    else:
        master = pd.DataFrame(
            columns=["session_datetime", "task_variant", "task_name", "metric", "value"]
        )

    existing_keys = set(
        zip(master["session_datetime"].astype(str), master["task_variant"])
    )

    new_rows = []
    for f in csv_files:
        variant = task_variant_from_filename(f)
        try:
            df = read_aimlab_csv(f)
        except Exception as e:
            print(f"  ! skipped {f.name}: {e}")
            continue
        if df.empty:
            continue

        df["session_datetime"] = pd.to_datetime(
            df["createDate"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
        metric_cols = [c for c in df.columns if c not in META_COLS and c != "session_datetime"]

        for _, row in df.iterrows():
            key = (str(row["session_datetime"]), variant)
            if key in existing_keys:
                continue  # already ingested this session
            task_name = row.get("taskName", variant)
            for m in metric_cols:
                val = row[m]
                if pd.isna(val):
                    continue
                new_rows.append(
                    {
                        "session_datetime": row["session_datetime"],
                        "task_variant": variant,
                        "task_name": task_name,
                        "metric": m,
                        "value": val,
                    }
                )
            existing_keys.add(key)

    if not new_rows:
        print("No new sessions found (everything already ingested).")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([master, new_df], ignore_index=True)
    combined = combined.sort_values("session_datetime")
    combined.to_csv(MASTER_PATH, index=False)

    n_sessions = new_df.drop_duplicates(["session_datetime", "task_variant"]).shape[0]
    print(f"Ingested {n_sessions} new session(s) across {new_df['task_variant'].nunique()} task(s).")
    print(f"Master dataset now has {combined.drop_duplicates(['session_datetime','task_variant']).shape[0]} total sessions.")
    print(f"Saved to {MASTER_PATH}")


def _score_wide(master):
    """Pivot to one row per session, one column per task_variant, values = score."""
    scores = master[master["metric"] == PRIMARY_METRIC].copy()
    wide = scores.pivot_table(
        index="session_datetime", columns="task_variant", values="value", aggfunc="mean"
    )
    return wide.sort_index()


def _trend_slope(dates, values):
    """Simple linear regression slope of value vs. session order (not calendar time,
    so sessions count evenly regardless of gaps)."""
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    return slope


def report():
    if not MASTER_PATH.exists():
        print(f"No master dataset yet. Run `ingest` first.")
        return

    master = pd.read_csv(MASTER_PATH, parse_dates=["session_datetime"])
    wide = _score_wide(master)
    n_sessions_total = wide.shape[0]

    variants = sorted(master["task_variant"].unique())

    # --- per-task trend + basic stats ---
    trend_rows = []
    for v in variants:
        series = wide[v].dropna() if v in wide else pd.Series(dtype=float)
        n = len(series)
        if n == 0:
            continue
        slope = _trend_slope(series.index, series.values)
        pct_change = None
        if n >= 2:
            pct_change = (series.values[-1] - series.values[0]) / series.values[0] * 100
        trend_rows.append(
            {
                "task_variant": v,
                "n_sessions": n,
                "latest_score": round(series.values[-1], 0) if n else np.nan,
                "mean_score": round(series.mean(), 0),
                "std_score": round(series.std(), 1) if n > 1 else np.nan,
                "slope_per_session": round(slope, 1) if not np.isnan(slope) else np.nan,
                "pct_change_first_to_last": round(pct_change, 1) if pct_change is not None else np.nan,
            }
        )
    trend_df = pd.DataFrame(trend_rows).sort_values("slope_per_session", ascending=False)

    # --- cross-task correlation (needs overlapping sessions / enough rows) ---
    corr = wide.corr(min_periods=3)  # NaN-heavy until you have enough weeks of data

    # --- leverage ranking: tasks whose score correlates most with the average of all others ---
    leverage_rows = []
    if wide.shape[1] > 1:
        avg_other = {}
        for v in variants:
            if v not in wide.columns:
                continue
            others = [c for c in wide.columns if c != v]
            if not others:
                continue
            avg_other_series = wide[others].mean(axis=1)
            paired = pd.concat([wide[v], avg_other_series], axis=1).dropna()
            if len(paired) >= 3:
                r = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            else:
                r = np.nan
            leverage_rows.append({"task_variant": v, "corr_with_other_tasks_avg": round(r, 2) if not pd.isna(r) else np.nan})
    leverage_df = pd.DataFrame(leverage_rows)

    # --- Apex correlation, if logged (one row per match; any numeric columns work) ---
    apex_summary_html = ""
    apex_corr_html = ""
    apex = None
    if APEX_PATH.exists():
        apex = pd.read_csv(APEX_PATH)
        try:
            # format="mixed" lets each row be parsed independently, so it's fine
            # if you log some dates as 08/09/2026 and others as 2026-08-09.
            apex["date"] = pd.to_datetime(apex["date"], errors="coerce", format="mixed")
        except TypeError:
            # older pandas without format="mixed" support
            apex["date"] = pd.to_datetime(apex["date"], errors="coerce")
        n_unparsed = apex["date"].isna().sum()
        apex = apex.dropna(subset=["date"])
        if n_unparsed:
            print(f"  ! {n_unparsed} row(s) in apex_sessions.csv had an unparseable date and were skipped.")

    # Friendlier labels for known columns; anything else falls back to the column name.
    STAT_LABELS = {
        "kills": "Avg kills",
        "damage": "Avg damage",
        "placement": "Avg placement",
        "knockdowns": "Avg knockdowns",
        "assists": "Avg assists",
    }

    if apex is None or apex.empty:
        apex_summary_html = (
            "<p class='muted'>No Apex match log yet. Add one row per game to "
            "<code>apex_sessions.csv</code> (date, plus whatever match stats you're tracking, e.g. "
            "kills, damage, placement, knockdowns, assists, notes) and re-run this report to see "
            "how your Aim Lab scores relate to actual in-game results.</p>"
        )
    else:
        # Any column that isn't date/notes and is numeric gets tracked automatically.
        apex_numeric_cols = [
            c for c in apex.columns
            if c not in ("date", "notes") and pd.api.types.is_numeric_dtype(apex[c])
        ]

        n_games = len(apex)
        stat_cards = [f'<div class="stat"><div class="stat-value">{n_games}</div><div class="stat-label">Games logged</div></div>']
        if "placement" in apex_numeric_cols:
            avg_placement = apex["placement"].mean()
            win_rate = (apex["placement"] == 1).mean() * 100
            stat_cards.append(f'<div class="stat"><div class="stat-value">{avg_placement:.1f}</div><div class="stat-label">Avg placement</div></div>')
            stat_cards.append(f'<div class="stat"><div class="stat-value">{win_rate:.0f}%</div><div class="stat-label">Win rate</div></div>')
        for c in apex_numeric_cols:
            if c == "placement":
                continue  # already shown above
            label = STAT_LABELS.get(c, f"Avg {c}")
            stat_cards.append(f'<div class="stat"><div class="stat-value">{apex[c].mean():.1f}</div><div class="stat-label">{label}</div></div>')
        apex_summary_html = f'<div class="stat-row">{"".join(stat_cards)}</div>'

        # weekly aggregates from per-game apex log — built dynamically from whatever columns exist
        apex_wk = apex.copy()
        apex_wk["week"] = apex_wk["date"].dt.to_period("W").dt.start_time
        agg_spec = {f"avg_{c}": (c, "mean") for c in apex_numeric_cols}
        if "placement" in apex_numeric_cols:
            agg_spec["win_rate"] = ("placement", lambda s: (s == 1).mean() * 100)
        agg_spec["games_played"] = ("date", "count")
        weekly_apex = apex_wk.groupby("week").agg(**agg_spec)

        wk = wide.copy()
        wk["week"] = wk.index.to_period("W").start_time
        wk_avg = wk.groupby("week").mean(numeric_only=True)

        joined = wk_avg.join(weekly_apex, how="inner")
        apex_metric_cols = [c for c in weekly_apex.columns if c != "games_played"]
        if len(joined) >= 3:
            cross_corr = joined.corr().loc[wk_avg.columns, apex_metric_cols]
            apex_corr_html = (
                "<p class='muted'>Weekly-average Aim Lab score per drill vs. weekly Apex results "
                "(Pearson correlation, -1 to 1). Higher |r| = stronger relationship.</p>"
                + cross_corr.round(2).to_html(classes="tbl")
            )
        else:
            apex_corr_html = (
                f"<p class='muted'>{len(joined)} overlapping week(s) of Aim Lab + Apex data so far "
                f"&mdash; need at least 3 weeks of both to compute correlations. Keep logging.</p>"
            )

    # --- charts (dark theme, embedded as base64 so the report is a single shareable file) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64

    ACCENT = "#ff5c3e"
    ACCENT2 = "#3ecbff"
    BG = "#12141a"
    GRID = "#262a34"
    TEXT = "#c9ccd6"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "grid.color": GRID,
        "font.family": "sans-serif",
        "font.size": 10,
    })

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    chart_cards = []
    for v in variants:
        if v not in wide.columns:
            continue
        series = wide[v].dropna()
        if len(series) < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 3))
        ax.plot(series.index, series.values, marker="o", color=ACCENT, linewidth=2, markersize=5)
        ax.fill_between(series.index, series.values, series.values.min(), color=ACCENT, alpha=0.08)
        ax.set_title(v, color="#ffffff", fontsize=12, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.4, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        b64 = fig_to_base64(fig)
        chart_cards.append(
            f'<div class="chart-card"><img src="data:image/png;base64,{b64}" alt="{v} trend"></div>'
        )

    # --- generated timestamp / header stats ---
    from datetime import datetime
    generated_at = datetime.now().strftime("%B %d, %Y — %H:%M")
    date_min = master["session_datetime"].min()
    date_max = master["session_datetime"].max()
    date_range = f"{date_min:%b %d, %Y} → {date_max:%b %d, %Y}" if pd.notna(date_min) else "—"

    def table_html(df, extra_class=""):
        if df is None or df.empty:
            return "<p class='muted'>Not enough data yet.</p>"
        return df.to_html(index=False, classes=f"tbl {extra_class}", border=0)

    if not corr.empty:
        corr_display = corr.round(2).reset_index().rename(columns={"index": "task_variant"})
        corr_table_html = table_html(corr_display)
    else:
        corr_table_html = table_html(None)

    # --- assemble HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aim Training Report</title>
<style>
  :root {{
    --bg: #0b0c10;
    --panel: #14161d;
    --panel-2: #1a1d26;
    --border: #262a34;
    --text: #e7e9ee;
    --muted: #8b909c;
    --accent: #ff5c3e;
    --accent2: #3ecbff;
    --good: #4ade80;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: radial-gradient(circle at top, #14161d 0%, #0b0c10 55%);
    color: var(--text);
    font-family: 'Segoe UI', -apple-system, Roboto, Arial, sans-serif;
    margin: 0;
    padding: 0 0 60px 0;
  }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 28px; }}
  header {{
    padding: 56px 0 32px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
  }}
  .eyebrow {{
    color: var(--accent);
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 10px;
  }}
  h1 {{
    font-size: 40px;
    margin: 0 0 8px 0;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff, #c9ccd6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .subtitle {{ color: var(--muted); font-size: 15px; margin: 0 0 24px 0; }}
  .meta-row {{ display: flex; gap: 28px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }}
  .meta-row b {{ color: var(--text); }}
  h2 {{
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 4px 0;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  h2::before {{
    content: "";
    width: 4px;
    height: 20px;
    background: var(--accent);
    border-radius: 2px;
    display: inline-block;
  }}
  section {{ margin-bottom: 44px; }}
  .muted {{ color: var(--muted); font-size: 13.5px; margin: 6px 0 16px 0; line-height: 1.5; }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 26px;
  }}
  .stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .stat {{
    flex: 1;
    min-width: 120px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: left;
  }}
  .stat-value {{ font-size: 26px; font-weight: 800; color: #ffffff; }}
  .stat-label {{ font-size: 12px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  table.tbl {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 4px;
    overflow: hidden;
    border-radius: 10px;
  }}
  table.tbl th {{
    background: var(--panel-2);
    color: var(--muted);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    text-align: right;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }}
  table.tbl th:first-child, table.tbl td:first-child {{ text-align: left; }}
  table.tbl td {{
    padding: 10px 14px;
    text-align: right;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  table.tbl tr:last-child td {{ border-bottom: none; }}
  table.tbl tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 18px;
  }}
  .chart-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
  }}
  .chart-card img {{ width: 100%; display: block; border-radius: 8px; }}
  footer {{
    margin-top: 60px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
    text-align: center;
  }}
  a {{ color: var(--accent2); }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Training Report</div>
    <h1>Aim Lab &rarr; Apex Legends</h1>
    <p class="subtitle">Tracking training consistency, drill trends, and (once logged) how it carries over to real matches.</p>
    <div class="meta-row">
      <span>Generated <b>{generated_at}</b></span>
      <span>Session window <b>{date_range}</b></span>
      <span><b>{n_sessions_total}</b> sessions logged</span>
      <span><b>{len(variants)}</b> drills tracked</span>
    </div>
  </header>

  <section>
    <h2>Per-drill trend summary</h2>
    <p class="muted">slope_per_session = average score change per logged session (not per calendar day). Positive = improving.</p>
    <div class="panel">{table_html(trend_df)}</div>
  </section>

  <section>
    <h2>Which drills track your overall aim</h2>
    <p class="muted">corr_with_other_tasks_avg: how closely a drill's score tracks the average of all your other drills.
    High = a good single proxy for overall aim health. Needs 3+ overlapping sessions per drill to compute.</p>
    <div class="panel">{table_html(leverage_df)}</div>
  </section>

  <section>
    <h2>Cross-drill correlation matrix</h2>
    <p class="muted">Score correlation between every pair of drills. Values near 1 move together; near 0 are independent skills.</p>
    <div class="panel">{corr_table_html}</div>
  </section>

  <section>
    <h2>Trend charts</h2>
    <div class="chart-grid">
      {"".join(chart_cards) if chart_cards else "<p class='muted'>Need 2+ sessions per drill to plot a trend.</p>"}
    </div>
  </section>

  <section>
    <h2>Apex Legends performance</h2>
    {apex_summary_html}
    <div class="panel">{apex_corr_html if apex_corr_html else "<p class='muted'>Log Apex matches to unlock this section.</p>"}</div>
  </section>

  <footer>
    Re-run <code>aimlab_pipeline.py ingest</code> after every Aim Lab session and <code>report</code> to refresh this page.
  </footer>

</div>
</body></html>
"""

    REPORT_PATH.write_text(html)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("ingest", "report"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python aimlab_pipeline.py ingest <folder>")
            sys.exit(1)
        ingest(sys.argv[2])
    else:
        report()
