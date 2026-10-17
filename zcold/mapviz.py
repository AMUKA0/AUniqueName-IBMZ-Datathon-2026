"""Interactive ColdWatch map (folium only - no geopandas).

Replay mode : python -m zcold.mapviz 2022-12-09        -> outputs/coldwatch_map_2022-12-09.html
Live mode   : python -m zcold.mapviz live [any-season] -> outputs/coldwatch_map_live.html
Notebook    : from zcold import mapviz; mapviz.build("2022-12-09")

Layers: triage (recoloured by the timeline), top-priority pins with Granite briefs,
heat map of people at risk, weather stations. Glass-panel UI, dark/light basemaps.
"""
import copy, json, re, sys
import numpy as np, pandas as pd
import folium
from folium.plugins import HeatMap
from .paths import PROC, ROOT
from . import scoring

OUT = ROOT / "outputs"
COL = {"1": "#f43f5e", "2": "#f59e0b", "3": "#64748b"}
PRIORITY_LABEL = {"1": "ACT NOW", "2": "PREPARE", "3": "MONITOR"}

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--glass:rgba(9,14,26,.82);--line:rgba(148,163,184,.18);--ink:#e2e8f0;--muted:#94a3b8;
      --red:#f43f5e;--amber:#f59e0b;--slate:#64748b;--cyan:#38bdf8}
.leaflet-container{font-family:Inter,system-ui,sans-serif;background:#0b1120}
.cw-panel{position:fixed;z-index:9999;background:var(--glass);backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px;
  box-shadow:0 12px 32px rgba(0,0,0,.38);color:var(--ink);font-family:Inter,system-ui,sans-serif}
#cw-head{top:16px;left:16px;padding:14px 16px;width:380px;max-width:calc(100vw - 32px)}
.cw-brand{display:flex;align-items:center;gap:8px;font-weight:800;font-size:18px;letter-spacing:-.01em}
.cw-brand .flake{color:var(--cyan);font-size:18px}
.cw-pill{margin-left:auto;font-size:10px;font-weight:700;letter-spacing:.08em;padding:3px 8px;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);display:flex;align-items:center;gap:6px}
.cw-pill.live{color:#fecdd3;border-color:rgba(244,63,94,.5)}
.cw-pill .dot{width:7px;height:7px;border-radius:50%;background:var(--muted)}
.cw-pill.live .dot{background:var(--red);animation:cwpulse 1.6s infinite}
.cw-sub{color:var(--muted);font-size:12px;margin:3px 0 12px}
.cw-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.cw-kpi{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.cw-kpi b{display:block;font-size:20px;font-weight:800;line-height:1.1}
.cw-kpi span{font-size:10.5px;color:var(--muted);line-height:1.25;display:block;margin-top:2px}
.cw-legend{display:flex;gap:14px;margin-top:12px;font-size:11.5px;color:var(--muted)}
.cw-legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.cw-hint{margin-top:8px;font-size:11px;color:#64748b}
#cw-bar{bottom:22px;left:50%;transform:translateX(-50%);padding:12px 16px;width:600px;max-width:calc(100vw - 32px)}
.cw-row1{display:flex;align-items:center;gap:14px}
#cw-play{width:38px;height:38px;border-radius:50%;border:0;background:var(--red);color:#fff;font-size:14px;
  cursor:pointer;box-shadow:0 0 0 4px rgba(244,63,94,.18);flex:none}
#cw-date{font-weight:700;font-size:15px;min-width:110px;text-align:right}
#cw-range{flex:1;-webkit-appearance:none;appearance:none;height:4px;border-radius:4px;background:#334155;outline:none}
#cw-range::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#fff;
  box-shadow:0 0 0 4px rgba(244,63,94,.35);cursor:pointer}
#cw-range::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;border:0;cursor:pointer}
#cw-stats{font-size:11.5px;color:var(--muted);margin-top:8px}
.leaflet-control-layers{background:var(--glass)!important;backdrop-filter:blur(10px);border:1px solid var(--line)!important;
  border-radius:12px!important;color:var(--ink);font:500 12px Inter,sans-serif;box-shadow:0 12px 32px rgba(0,0,0,.38)!important}
.leaflet-control-layers-separator{border-top-color:var(--line)!important}
.leaflet-bar a{background:var(--glass)!important;color:var(--ink)!important;border-color:var(--line)!important}
.leaflet-control-attribution{background:rgba(9,14,26,.6)!important;color:#64748b!important}
.leaflet-control-attribution a{color:#94a3b8!important}
.leaflet-control-scale-line{background:rgba(9,14,26,.6);color:#cbd5e1;border-color:#64748b}
.leaflet-tooltip{background:var(--glass)!important;color:var(--ink)!important;border:1px solid var(--line)!important;
  border-radius:10px!important;box-shadow:0 8px 24px rgba(0,0,0,.4)!important;font:500 12px Inter,sans-serif!important}
.leaflet-tooltip-left:before,.leaflet-tooltip-right:before{display:none}
.leaflet-popup-content-wrapper{background:rgba(9,14,26,.94)!important;color:var(--ink)!important;
  border:1px solid var(--line);border-radius:14px!important;box-shadow:0 16px 40px rgba(0,0,0,.5)!important}
.leaflet-popup-tip{background:rgba(9,14,26,.94)!important}
.leaflet-popup-content{margin:14px 16px!important}
.leaflet-container a.leaflet-popup-close-button{color:#94a3b8!important}
.cw-card{font-family:Inter,sans-serif;width:340px;color:#f1f5f9!important}
.cw-card *{color:inherit}
.cw-card table,.cw-card td,.cw-card th{width:100%;color:#f1f5f9!important;background:transparent!important}
.leaflet-popup-content table,.leaflet-popup-content td,.leaflet-popup-content th{color:#f1f5f9!important;
  background:transparent!important;border:0!important;padding:0!important}
.cw-card .cw-kpi b{color:#ffffff!important;font-size:22px}
.cw-card .cw-kpi span{color:#cbd5e1!important;font-size:11px}
.leaflet-control-layers label,.leaflet-control-layers span{color:#e2e8f0!important}
.cw-card-h{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.cw-badge{font-size:10px;font-weight:800;letter-spacing:.08em;padding:3px 7px;border-radius:6px;color:#fff}
.cw-badge.p1{background:var(--red)}.cw-badge.p2{background:var(--amber);color:#1f1300}.cw-badge.p3{background:var(--slate)}
.cw-card-t{font-size:17px;font-weight:800;color:#ffffff!important}
.cw-card-s{font-size:12px;color:#94a3b8!important}
.cw-card .cw-kpis{margin:12px 0}
.cw-card .cw-kpi{background:rgba(255,255,255,.06);border-color:rgba(148,163,184,.25)}
.cw-r{font-size:13px;line-height:1.4;margin:8px 0;color:#e2e8f0!important}
.cw-r label{display:block;font-size:10px;font-weight:700;letter-spacing:.08em;color:#94a3b8!important;
  text-transform:uppercase;margin-bottom:1px}
.cw-act{margin-top:12px;padding:10px 12px;border-radius:10px;background:rgba(56,189,248,.12);
  border:1px solid rgba(56,189,248,.45);font-size:13px;line-height:1.5;color:#f0f9ff!important}
.cw-act label{display:block;font-size:10px;font-weight:800;letter-spacing:.08em;color:#7dd3fc!important;margin-bottom:4px}
.cw-src,.cw-foot{font-size:11px;color:#94a3b8!important;margin-top:8px}
.cw-pin{width:28px;height:28px;border-radius:50%;background:var(--red);color:#fff;border:2px solid #fff;
  font:800 13px Inter,sans-serif;display:flex;align-items:center;justify-content:center;
  animation:cwpulse 1.8s infinite}
.cw-st{display:flex;align-items:center;gap:6px;white-space:nowrap;font:700 11px Inter,sans-serif;color:#e0f2fe;
  text-shadow:0 1px 3px #000}
.cw-st i{width:12px;height:12px;border-radius:50%;background:var(--cyan);border:2px solid #fff;
  box-shadow:0 0 12px rgba(56,189,248,.9)}
@keyframes cwpulse{0%{box-shadow:0 0 0 0 rgba(244,63,94,.7)}70%{box-shadow:0 0 0 12px rgba(244,63,94,0)}
  100%{box-shadow:0 0 0 0 rgba(244,63,94,0)}}
</style>"""


# ------------------------------------------------------------------ geometry
def _round(c):
    if isinstance(c[0], (int, float)):
        return [round(c[0], 4), round(c[1], 4)]
    return [_round(x) for x in c]


def load_geo():
    gj = json.load(open(PROC / "lsoa_gm.geojson"))
    for i, f in enumerate(gj["features"]):
        f["id"] = str(i)
        f["geometry"]["coordinates"] = _round(f["geometry"]["coordinates"])
    return gj


def centroids(gj):
    rows = []
    for f in gj["features"]:
        pts = []
        def walk(c):
            if isinstance(c[0], (int, float)):
                pts.append(c)
            else:
                for x in c:
                    walk(x)
        walk(f["geometry"]["coordinates"])
        a = np.array(pts)
        rows.append((f["properties"]["LSOA21CD"], a[:, 1].mean(), a[:, 0].mean()))
    return pd.DataFrame(rows, columns=["lsoa21cd", "lat", "lon"])


# ------------------------------------------------------------------ data
def _enrich(df):
    df = df.copy()
    df["alone_66plus_hh"] = (df.pct_alone_66plus / 100 * df.n_households).round(0)
    df["expected_at_risk"] = (df.p_trigger_7d * df.alone_66plus_hh).round(1)
    return df


def daily_scores(start, end):
    vuln = scoring.load_vulnerability()
    preds = pd.read_csv(PROC / "hazard_test_predictions.csv", parse_dates=["date"])
    days = {}
    for d in pd.date_range(start, end):
        h = (preds[preds.date == d].rename(columns={"p_trigger": "p_trigger_7d"})
             [["station_name", "p_trigger_7d", "pred_tmean_7d"]])
        if not h.empty:
            days[d] = _enrich(scoring.score(vuln, h))
    return days


def live_scores(refresh=True, ignore_season=False):
    """This week, looking forward: live Open-Meteo data -> LSTM -> P(trigger) -> triage."""
    if refresh or not (PROC / "station_forecast.csv").exists():
        from . import hazard                      # imported only when a live refresh is needed
        print(hazard.forecast_now(ignore_season=ignore_season).to_string(index=False))
    fc = pd.read_csv(PROC / "station_forecast.csv")
    day = pd.Timestamp(fc.forecast_date.iloc[0])
    df = _enrich(scoring.score(scoring.load_vulnerability(),
                               fc[["station_name", "p_trigger_7d", "pred_tmean_7d"]]))
    df.to_csv(PROC / "lsoa_risk.csv", index=False)
    return {day: df}, day


def _extras(date):
    """Granite briefs {code: (action, sources)} and register counts {lsoa_name: text}."""
    briefs, reg = {}, {}
    bp, rp = PROC / f"briefs_{date}.csv", PROC / f"register_agg_{date}.csv"
    if bp.exists():
        b = pd.read_csv(bp, dtype={"lsoa21cd": str})
        for _, r in b.iterrows():
            action = r.get("granite_action") if "granite_action" in b else None
            if not isinstance(action, str):
                m = re.search(r"Action \(Granite\):\s*(.*)", str(r.brief))
                action = m.group(1) if m else ""
            src = re.search(r"Sources:\s*(.*)", str(r.brief))
            briefs[r.lsoa21cd] = (action, src.group(1) if src else "")
    if rp.exists():
        r = pd.read_csv(rp)
        reg = {n: f"{a} on the register &middot; {b} aged 80+ &middot; {c} with medical equipment"
               for n, a, b, c in zip(r.lsoa21nm, r.registered_at_risk, r.aged_80plus, r.medical_equipment)}
    return briefs, reg


def _card(r, briefs, reg, when):
    p = str(r.priority)[0]
    html = (f"<div class='cw-card'><div class='cw-card-h'><span class='cw-badge p{p}'>{PRIORITY_LABEL[p]}</span>"
            f"<span class='cw-card-t'>{r.lsoa21nm}</span><span class='cw-card-s'>{r.lad_name}</span></div>"
            f"<div class='cw-kpis'><div class='cw-kpi'><b>{r.p_trigger_7d:.0%}</b><span>freeze-week chance</span></div>"
            f"<div class='cw-kpi'><b>{r.vuln_index:.0f}</b><span>vulnerability / 100</span></div>"
            f"<div class='cw-kpi'><b>~{r.alone_66plus_hh:.0f}</b><span>older people living alone</span></div></div>"
            f"<div class='cw-r'><label>Need type</label>{str(r.archetype).split(': ', 1)[-1]}</div>"
            f"<div class='cw-r'><label>Why this area</label>{r.driver_1}; {r.driver_2}</div>")
    if reg.get(r.lsoa21nm):
        html += f"<div class='cw-r'><label>Council register (pseudonymised)</label>{reg[r.lsoa21nm]}</div>"
    code = r["lsoa21cd"] if "lsoa21cd" in r.index else r.name     # works for indexed and plain rows
    if code in briefs:
        action, src = briefs[code]
        html += f"<div class='cw-act'><label>IBM GRANITE &middot; RECOMMENDED ACTION</label>{action}</div>"
        if src:
            html += f"<div class='cw-src'>Sources: {src}</div>"
    return html + f"<div class='cw-foot'>{when}</div></div>"


# ------------------------------------------------------------------ map
def build(date="2022-12-09", days_before=8, days_after=5, top_n=5, ignore_season=False, refresh=True):
    """date='YYYY-MM-DD' -> replay with timeline; date='live' -> this week's forecast."""
    gj = load_geo()
    cents = centroids(gj)
    live = date == "live"
    if live:
        days, focus = live_scores(refresh=refresh, ignore_season=ignore_season)
    else:
        focus = pd.Timestamp(date)
        days = daily_scores(focus - pd.Timedelta(days=days_before), focus + pd.Timedelta(days=days_after))
    df = days[focus].merge(cents, on="lsoa21cd", how="left")
    briefs, reg = _extras(date)
    when = f"Live forecast issued {focus:%d %b %Y}" if live else f"Replay snapshot for {focus:%d %b %Y}"

    m = folium.Map(location=[53.50, -2.30], zoom_start=10, tiles=None, control_scale=True,
                   zoom_control=False, prefer_canvas=True)
    m.get_root().header.add_child(folium.Element(CSS))
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(esri.format("World_Dark_Gray_Base"), attr="Tiles &copy; Esri", name="Dark map",
                     max_zoom=16).add_to(m)                                   # first = default
    folium.TileLayer(esri.format("World_Light_Gray_Base"), attr="Tiles &copy; Esri", name="Light map",
                     max_zoom=16, show=False).add_to(m)

    # 1. triage layer
    info = df.set_index("lsoa21cd")
    gj_focus = copy.deepcopy(gj)
    for f in gj_focus["features"]:
        code = f["properties"]["LSOA21CD"]
        if code in info.index:
            r = info.loc[code]
            f["properties"].update({"name": r.lsoa21nm, "priority": r.priority,
                                    "freeze": f"{r.p_trigger_7d:.0%}",
                                    "people": f"~{r.alone_66plus_hh:.0f} older people alone",
                                    "card": _card(r, briefs, reg, when)})
        else:
            f["properties"].update({"name": f["properties"].get("LSOA21NM", code), "priority": "3 - Monitor",
                                    "freeze": "", "people": "", "card": ""})

    def style(f):
        p = f["properties"]["priority"][0]
        return {"fillColor": COL[p], "fillOpacity": {"1": .72, "2": .40, "3": .03}[p],
                "color": {"1": "#fda4af", "2": "#fcd34d", "3": "#475569"}[p],
                "weight": {"1": .8, "2": .5, "3": .25}[p], "opacity": .7}

    geo = folium.GeoJson(
        gj_focus, name="Triage areas", style_function=style,
        highlight_function=lambda f: {"weight": 2.5, "color": "#ffffff", "fillOpacity": .9},
        tooltip=folium.GeoJsonTooltip(fields=["name", "priority", "freeze", "people"],
                                      aliases=["Area", "Priority", "Freeze-week chance", "Reach"],
                                      sticky=True, style="padding:6px 8px"),
        popup=folium.GeoJsonPopup(fields=["card"], labels=False, max_width=380),
    ).add_to(m)

    # 2. heat map
    pts = df.dropna(subset=["lat", "lon"])
    HeatMap(pts[["lat", "lon", "expected_at_risk"]].values.tolist(), name="Heat: people at risk",
            radius=20, blur=24, min_opacity=.15, max_zoom=13, show=False,
            gradient={.2: "#fde68a", .45: "#f59e0b", .7: "#f43f5e", 1: "#881337"}).add_to(m)

    # 3. weather stations
    st = (pd.read_csv(PROC / "lsoa_to_station.csv").drop_duplicates("station_name")
            .merge(df.drop_duplicates("station_name")[["station_name", "p_trigger_7d"]], on="station_name"))
    fg = folium.FeatureGroup(name="Weather stations").add_to(m)
    for _, s in st.iterrows():
        lab = f"<div class='cw-st'><i></i>{s.station_name} &middot; <span class='cw-st-p' data-st='{s.station_name}'>{s.p_trigger_7d:.0%}</span></div>"
        folium.Marker([s.lat, s.lon], icon=folium.DivIcon(html=lab, icon_size=(160, 16), icon_anchor=(7, 8)),
                      tooltip=f"{s.station_name} weather station: chance a freezing 7-day window completes this week").add_to(fg)

    # 4. top-priority areas with Granite briefs: dashed outline + pulsing numbered pins
    top_codes = list(briefs)[:top_n] if briefs else (
        df[df.priority.str.startswith("1")].sort_values("expected_at_risk", ascending=False)
        .lsoa21cd.head(top_n).tolist())
    if top_codes:
        top_fg = folium.FeatureGroup(name=f"Top {len(top_codes)} priority areas").add_to(m)
        folium.GeoJson({"type": "FeatureCollection",
                        "features": [f for f in gj_focus["features"] if f["properties"]["LSOA21CD"] in top_codes]},
                       style_function=lambda f: {"fill": False, "color": "#ffffff", "weight": 2.5,
                                                 "dashArray": "5,4"}).add_to(top_fg)
        for i, code in enumerate(top_codes, 1):
            if code not in info.index:
                continue
            r = df[df.lsoa21cd == code].iloc[0]
            folium.Marker([r.lat, r.lon], icon=folium.DivIcon(html=f"<div class='cw-pin'>{i}</div>",
                                                               icon_size=(28, 28), icon_anchor=(14, 14)),
                          tooltip=f"#{i} {r.lsoa21nm}",
                          popup=folium.Popup(_card(r, briefs, reg, when), max_width=380)).add_to(top_fg)

    # 5. header panel
    act = df[df.priority.str.startswith("1")]
    pmax = df.p_trigger_7d.max()
    pill = ("<span class='cw-pill live'><span class='dot'></span>LIVE</span>" if live else
            "<span class='cw-pill'><span class='dot'></span>REPLAY</span>")
    sub = (f"Forecast issued {focus:%a %d %b %Y} &middot; next 7 days" if live else
           f"December 2022 cold spell &middot; model never saw this winter")
    hint = ("Live Open-Meteo data &rarr; LSTM on IBM Z &rarr; triage. Payments can only trigger Nov&ndash;Mar."
            if live else "Hover an area &middot; click a numbered pin for its Granite brief &middot; press &#9654; to replay")
    head = f"""
<div id="cw-head" class="cw-panel">
  <div class="cw-brand">ColdWatch{pill}</div>
  <div class="cw-sub">Greater Manchester cold-weather outreach &middot; {sub}</div>
  <div class="cw-kpis">
    <div class="cw-kpi"><b id="cw-k-act" style="color:var(--red)">{len(act)}</b><span>areas to act on now</span></div>
    <div class="cw-kpi"><b id="cw-k-people">{act.alone_66plus_hh.sum():,.0f}</b><span>older people living alone in them</span></div>
    <div class="cw-kpi"><b id="cw-k-freeze" style="color:var(--cyan)">{pmax:.0%}</b><span>highest freeze-week chance</span></div>
  </div>
  <div class="cw-legend"><span><i style="background:var(--red)"></i>Act now</span>
    <span><i style="background:var(--amber)"></i>Prepare</span><span><i style="background:var(--slate)"></i>Monitor</span>
    <span><i style="background:#fff;border-radius:50%"></i>Granite brief</span></div>
  <div class="cw-hint">{hint}</div>
</div>"""
    m.get_root().html.add_child(folium.Element(head))
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds([[pts.lat.min(), pts.lon.min()], [pts.lat.max(), pts.lon.max()]])
    m.get_root().script.add_child(folium.Element(
        f'window.addEventListener("load", function() {{ L.control.zoom({{position: "bottomright"}}).addTo({m.get_name()}); }});'))
    if len(days) > 1:
        _add_timeline(m, geo, days, focus)
    return m


def _add_timeline(m, geo, days, focus):
    """Bottom timeline: play + slider; recolours the triage layer and updates the header KPIs."""
    dates = sorted(days)
    keys = [d.strftime("%Y-%m-%d") for d in dates]
    data, stats = {}, {}
    for k, d in zip(keys, dates):
        dd = days[d]
        data[k] = {c: [str(p)[0], f"{q:.0%}"] for c, p, q in zip(dd.lsoa21cd, dd.priority, dd.p_trigger_7d)}
        act = dd.priority.str.startswith("1")
        st = dd.drop_duplicates("station_name")
        stats[k] = {"act": int(act.sum()), "prep": int(dd.priority.str.startswith("2").sum()),
                    "people": int(dd.loc[act, "alone_66plus_hh"].sum()), "pmax": f"{dd.p_trigger_7d.max():.0%}",
                    "st": {s: f"{p:.0%}" for s, p in zip(st.station_name, st.p_trigger_7d)}}
    start = keys.index(focus.strftime("%Y-%m-%d")) if focus in dates else 0
    html = f"""
<div id="cw-bar" class="cw-panel">
  <div class="cw-row1">
    <button id="cw-play" title="Replay the week">&#9654;</button>
    <input id="cw-range" type="range" min="0" max="{len(keys) - 1}" value="{start}">
    <div id="cw-date"></div>
  </div>
  <div id="cw-stats"></div>
</div>"""
    js = f"""
window.addEventListener("load", function() {{
  var DATA = {json.dumps(data)}, STATS = {json.dumps(stats)}, DATES = {json.dumps(keys)};
  var COL = {json.dumps(COL)}, FO = {{"1": .72, "2": .40, "3": .03}}, W = {{"1": .8, "2": .5, "3": .25}};
  var EDGE = {{"1": "#fda4af", "2": "#fcd34d", "3": "#475569"}};
  var NAME = {{"1": "1 - Act now", "2": "2 - Prepare", "3": "3 - Monitor"}};
  var geo = {geo.get_name()}, cur = DATES[{start}];
  function styleFor(f) {{
    var x = DATA[cur][f.properties.LSOA21CD], p = x ? x[0] : "3";
    return {{fillColor: COL[p], fillOpacity: FO[p], color: EDGE[p], weight: W[p], opacity: .7}};
  }}
  geo.options.style = styleFor;
  function show(i) {{
    cur = DATES[i];
    geo.setStyle(styleFor);
    geo.eachLayer(function(l) {{
      var x = DATA[cur][l.feature.properties.LSOA21CD];
      if (x) {{ l.feature.properties.priority = NAME[x[0]]; l.feature.properties.freeze = x[1]; }}
    }});
    var s = STATS[cur];
    document.getElementById("cw-date").textContent = new Date(cur).toLocaleDateString("en-GB",
        {{weekday: "short", day: "numeric", month: "short", year: "numeric"}});
    document.getElementById("cw-k-act").textContent = s.act;
    document.getElementById("cw-k-people").textContent = s.people.toLocaleString();
    document.getElementById("cw-k-freeze").textContent = s.pmax;
    document.querySelectorAll(".cw-st-p").forEach(function(e) {{ e.textContent = s.st[e.dataset.st] || ""; }});
    document.getElementById("cw-stats").innerHTML = "<b style='color:#f43f5e'>" + s.act + "</b> act now &middot; <b style='color:#f59e0b'>" +
        s.prep + "</b> prepare &middot; freeze-week chance " +
        Object.keys(s.st).map(function(k) {{ return k + " <b>" + s.st[k] + "</b>"; }}).join(" &middot; ");
  }}
  var r = document.getElementById("cw-range"), btn = document.getElementById("cw-play"), t = null;
  r.addEventListener("input", function() {{ show(+r.value); }});
  btn.onclick = function() {{
    if (t) {{ clearInterval(t); t = null; btn.innerHTML = "&#9654;"; return; }}
    btn.innerHTML = "&#10074;&#10074;"; r.value = 0; show(0);
    t = setInterval(function() {{
      if (+r.value >= DATES.length - 1) {{ clearInterval(t); t = null; btn.innerHTML = "&#9654;"; return; }}
      r.value = +r.value + 1; show(+r.value);
    }}, 900);
  }};
  show({start});
}});"""
    m.get_root().html.add_child(folium.Element(html))
    m.get_root().script.add_child(folium.Element(js))


def run(date="2022-12-09", ignore_season=False):
    OUT.mkdir(exist_ok=True)
    m = build(date, ignore_season=ignore_season)
    path = OUT / f"coldwatch_map_{date}.html"
    m.save(str(path))
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB) - download it and open in a browser")


if __name__ == "__main__":
    # python -m zcold.mapviz 2022-12-09 | live | live any-season
    run(sys.argv[1] if len(sys.argv) > 1 else "2022-12-09",
        ignore_season=len(sys.argv) > 2 and sys.argv[2] == "any-season")
