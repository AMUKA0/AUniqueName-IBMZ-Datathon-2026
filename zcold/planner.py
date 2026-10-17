"""[7] ColdWatch Planner: from risk forecasts to a daily outreach plan.

Every morning, using only that day's forecast (rolling, no hindsight), a mixed-integer programme
chooses which priority neighbourhoods the outreach teams visit:

  maximise   sum_i x_i * n_i * v_i * P_i            expected vulnerable older people protected
  subject to sum_i x_i <= K                         daily visit capacity across Greater Manchester
             sum_{i in b} x_i <= cap_b              each borough's teams (share of older-alone pop.)
             sum_{i in g} x_i >= floor_g            fairness: every need type gets >= 15% of visits
             x_i = 0 if visited in the last 7 days, or P_i < 0.2
             x_i in {0, 1}
  n_i = older people living alone, v_i = vulnerability index, P_i = P(payment trigger in 7 days)
  Solver: scipy.optimize.milp (HiGHS).

Back-test on every test winter (2022-23 onward; the LSTM never saw them). An area is 'reached in
time' for a cold event if it was visited in the 7 days before its station's freezing week completed.
Strategies compared at the same daily capacity:
  ColdWatch planner    LSTM P(trigger)
  Persistence planner  same optimiser, naive 'tomorrow = today' forecast
  Static rota          no forecast; cycles areas by vulnerability, SAME total visits as ColdWatch
  Reactive             today's system: outreach starts when the payment triggers -> 0% in time

python -m zcold.planner   -> planner_eval.csv, planner_replay.json, images/planner_reach.png
"""
import json, math
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from .paths import PROC, ROOT
from . import scoring

KS = (10, 20, 40, 80)          # visits per day (one visit = one neighbourhood outreach push)
REVISIT = 7                    # a visit covers a week (matches the 7-day trigger window)
FLOOR = 0.15                   # minimum share of the day's visits for each need type
ALERT_P = scoring.PREP_P       # plan only where P(trigger) >= 0.2
WINTER = [11, 12, 1, 2, 3]
REPLAY = ("2022-12-01", "2022-12-14")
DAY = pd.Timedelta(days=1)


# ------------------------------------------------------------------ inputs
def load_areas():
    """Priority neighbourhoods (High / Very high vulnerability) with people, weights and teams."""
    v = scoring.load_vulnerability()
    v = v[v.vuln_band.isin(scoring.HIGH_BANDS)].dropna(subset=["station_name"]).copy()
    v["n"] = (v.pct_alone_66plus / 100 * v.n_households).round(0)
    v["w"] = v.n * v.vuln_index / 100
    v["need"] = v.archetype.astype(str).str.split(":").str[0]
    return v.reset_index(drop=True)


def load_forecasts():
    p = pd.read_csv(PROC / "hazard_test_predictions.csv", parse_dates=["date"])
    return {name: p.pivot_table(index="date", columns="station_name", values=col)
            for name, col in (("ColdWatch", "p_trigger"), ("Persistence", "p_persistence"), ("label", "label"))}


def trigger_events(start, end):
    """Days each station's first qualifying 7-day window completes (mean tmean <= 0, all Nov-Mar)."""
    t = pd.read_csv(PROC / "station_daily_temp.csv", parse_dates=["date"])
    rows = []
    for s, g in t.sort_values("date").groupby("station_name"):
        g = g.drop_duplicates("date").set_index("date").asfreq("D")
        mean7 = g.tmean.rolling(7, min_periods=7).mean()
        in_winter = pd.Series(g.index.month.isin(WINTER), index=g.index).astype(int)
        q = (mean7 <= 0) & (in_winter.rolling(7, min_periods=7).sum() == 7)
        onset = q & ~q.shift(1, fill_value=False)
        rows += [(s, d) for d in g.index[onset.values] if start <= d <= end]
    return pd.DataFrame(rows, columns=["station_name", "onset"])


def borough_caps(areas, K):
    share = areas.groupby("lad_name").n.sum() / areas.n.sum()
    return {b: math.ceil(K * s) for b, s in share.items()}


# ------------------------------------------------------------------ optimiser
def solve_day(value, borough, need, caps, K):
    """Indices of the areas to visit today (MILP; drops the fairness floor only if infeasible)."""
    m = len(value)
    k = int(min(K, m))
    bs = np.unique(borough)
    rows, lo, hi = [np.ones(m)], [0], [k]
    for b in bs:
        rows.append((borough == b).astype(float)); lo.append(0); hi.append(caps[b])
    usable = min(k, sum(min(caps[b], int((borough == b).sum())) for b in bs))
    floors = [((need == g).astype(float), min(int((need == g).sum()), math.ceil(FLOOR * usable)))
              for g in np.unique(need)]
    for with_floor in (True, False):
        R, L, U = list(rows), list(lo), list(hi)
        if with_floor:
            for r, f in floors:
                R.append(r); L.append(f); U.append(np.inf)
        res = milp(c=-value, constraints=LinearConstraint(np.array(R), L, U),
                   integrality=np.ones(m), bounds=Bounds(0, 1))
        if res.status == 0:
            return np.flatnonzero(res.x > 0.5)
    return np.argsort(-value)[:k]


def plan(areas, P, K, dates):
    """Rolling daily plan. Returns DataFrame(date, area) of visits."""
    caps = borough_caps(areas, K)
    st, bor, need, w = (areas[c].values for c in ("station_name", "lad_name", "need", "w"))
    last = np.full(len(areas), -10 ** 6)
    out = []
    for d in dates:
        if d not in P.index:
            continue
        p = P.loc[d].reindex(st).fillna(0).values
        idx = np.flatnonzero((p >= ALERT_P) & (d.toordinal() - last >= REVISIT))
        if len(idx) == 0:
            continue
        chosen = idx[solve_day(w[idx] * p[idx], bor[idx], need[idx], caps, K)]
        last[chosen] = d.toordinal()
        out += [(d, i) for i in chosen]
    return pd.DataFrame(out, columns=["date", "area"])


def rota(areas, per_day, dates):
    """Forecast-free baseline: cycle through areas in vulnerability order every winter day."""
    order = np.argsort(-areas.w.values)
    out, acc, ptr = [], 0.0, 0
    for d in dates:
        if d.month not in WINTER:
            continue
        acc += per_day
        n = int(acc); acc -= n
        for _ in range(n):
            out.append((d, order[ptr % len(order)])); ptr += 1
    return pd.DataFrame(out, columns=["date", "area"])


# ------------------------------------------------------------------ back-test
def evaluate(visits, areas, events, labels):
    n, st = areas.n.values, areas.station_name.values
    v = visits.assign(station=st[visits.area.values]) if len(visits) else visits.assign(station=[])
    at_risk = reached = 0.0
    for s, e in events.itertuples(index=False):
        at_risk += n[st == s].sum()
        hit = v[(v.station == s) & (v.date >= e - 7 * DAY) & (v.date < e)].area.unique()
        reached += n[hit].sum()
    # useful visit = a freezing week really did complete within 7 days (the observed label)
    lab = labels.stack()
    useful = lab.reindex(pd.MultiIndex.from_arrays([v.date, v.station])).fillna(0).values == 1
    return {"reached_in_time_pct": round(100 * reached / at_risk, 1) if at_risk else 0.0,
            "visits": len(v), "false_alarm_visits_pct": round(100 * (1 - useful.mean()), 1) if len(v) else 0.0}


def chart(df):
    import matplotlib.pyplot as plt
    col = {"ColdWatch planner": "#e11d48", "Persistence planner": "#64748b", "Static rota (same visits)": "#0ea5e9"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, c in col.items():
        g = df[df.strategy == name]
        ax[0].plot(g.capacity, g.reached_in_time_pct, "o-", color=c, lw=2.5, label=name)
        ax[1].plot(g.capacity, g.false_alarm_visits_pct, "o-", color=c, lw=2.5, label=name)
    ax[0].axhline(0, color="#111827", lw=1, ls=":")
    ax[0].text(KS[0], 1.5, "Reactive (today): 0%, outreach starts after the trigger", fontsize=9)
    ax[0].set(title="Older people living alone reached in the week BEFORE the trigger",
              xlabel="outreach capacity (neighbourhood visits per day)", ylabel="% of priority residents")
    ax[1].set(title="Visits spent on false alarms", xlabel="outreach capacity (visits per day)", ylabel="% of visits")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(KS); a.set_xticklabels(KS); a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    fig.suptitle("ColdWatch Planner: back-test on every test winter (2022-23 onward)", fontweight="bold")
    fig.tight_layout()
    (ROOT / "images").mkdir(exist_ok=True)
    fig.savefig(ROOT / "images/planner_reach.png", dpi=150, bbox_inches="tight")
    return fig


def replay_json(areas, P, events):
    """Day-by-day plans for the map timeline, one per capacity option."""
    dates = pd.date_range(*REPLAY)
    total = float(areas.n.sum())
    out = {"total": int(total), "plans": {}}
    for K in KS:
        vis = plan(areas, P, K, dates)
        seen, days = set(), {}
        for d in dates:
            today = vis[vis.date == d].area.tolist()
            seen |= set(today)
            days[d.strftime("%Y-%m-%d")] = {
                "codes": areas.lsoa21cd.values[today].tolist(), "visits": len(today),
                "people": int(areas.n.values[today].sum()),
                "pct": round(100 * areas.n.values[list(seen)].sum() / total, 1)}
        out["plans"][str(K)] = days
    json.dump(out, open(PROC / "planner_replay.json", "w"))
    return out


def run():
    areas, P = load_areas(), load_forecasts()
    dates = P["ColdWatch"].index.sort_values()
    events = trigger_events(dates.min() + 7 * DAY, dates.max())
    print(f"{len(areas)} priority neighbourhoods, {areas.n.sum():,.0f} older people living alone, "
          f"{areas.lad_name.nunique()} boroughs")
    print(f"Cold events in the test winters (freezing week completed):\n{events.to_string(index=False)}\n")
    rows = []
    for K in KS:
        cw = plan(areas, P["ColdWatch"], K, dates)
        rows.append({"capacity": K, "strategy": "ColdWatch planner", **evaluate(cw, areas, events, P["label"])})
        pe = plan(areas, P["Persistence"], K, dates)
        rows.append({"capacity": K, "strategy": "Persistence planner", **evaluate(pe, areas, events, P["label"])})
        per_day = len(cw) / max(1, sum(d.month in WINTER for d in dates))
        rows.append({"capacity": K, "strategy": "Static rota (same visits)",
                     **evaluate(rota(areas, per_day, dates), areas, events, P["label"])})
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "planner_eval.csv", index=False)
    print("Back-test (reached in time = visited in the 7 days before the payment trigger):")
    print(df.pivot(index="capacity", columns="strategy", values="reached_in_time_pct").to_string())
    print("\nVisits used / % on false alarms:")
    print(df.pivot(index="capacity", columns="strategy", values="visits").to_string())
    print(df.pivot(index="capacity", columns="strategy", values="false_alarm_visits_pct").to_string())
    print("Reactive (today's system): 0% reached in time by construction.\n")
    chart(df)
    rep = replay_json(areas, P["ColdWatch"], events)
    d9 = rep["plans"]["40"].get("2022-12-09", {})
    print(f"Replay, 40 visits/day, 9 Dec 2022: {d9.get('visits')} visits today, "
          f"{d9.get('pct')}% of priority older people reached so far")
    print("Saved planner_eval.csv, planner_replay.json, images/planner_reach.png")
    return df


if __name__ == "__main__":
    run()
