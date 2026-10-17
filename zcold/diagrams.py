"""Architecture / flow diagrams for the ColdWatch story notebook (matplotlib only).

python -m zcold.diagrams   -> images/architecture.png, images/decision_flow.png, images/lstm_model.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .paths import ROOT

IMG = ROOT / "images"
INK, MUTED, LINE = "#0f172a", "#475569", "#cbd5e1"
RED, AMBER, CYAN, GREEN, VIOLET = "#e11d48", "#d97706", "#0284c7", "#059669", "#7c3aed"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def _box(ax, x, y, w, h, title, sub="", fc="#ffffff", ec=LINE, tc=INK, lw=1.4, r=0.12, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.2, color=MUTED, zorder=3)


def _badge(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=7.4, fontweight="bold", color="white", zorder=4,
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.35", fc=color, ec="none"))


def _arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.6, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                shrinkA=2, shrinkB=2, mutation_scale=14), zorder=1)


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ 1. architecture
def architecture(path=None):
    fig, ax = _canvas(16, 9)
    ax.text(0.3, 8.6, "ColdWatch: system architecture on IBM LinuxONE", fontsize=16, fontweight="bold", color=INK)

    # sources
    ax.text(1.45, 7.85, "OPEN DATA", ha="center", fontsize=9, fontweight="bold", color=MUTED)
    srcs = [("IoD 2025", "deprivation, housing"), ("Census 2021", "age, alone, disability"),
            ("ONS postcodes", "+ LSOA boundaries"), ("DWP CWP stations", "postcode → station"),
            ("ERA5 + Open-Meteo", "history + live weather"), ("GOV.UK policy", "CWP, Warm Home Discount")]
    for i, (t, s) in enumerate(srcs):
        _box(ax, 0.3, 6.9 - i * 1.12, 2.3, 0.85, t, s, fs=9.5)

    # LinuxONE container
    ax.add_patch(FancyBboxPatch((3.3, 0.35), 9.0, 7.75, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc="#f8fafc", ec=INK, lw=2, zorder=0))
    ax.text(3.55, 7.8, "IBM LinuxONE (z17)  ·  Docker container  ·  Telum on-chip AI accelerator",
            fontsize=10.5, fontweight="bold", color=INK)

    _box(ax, 3.7, 6.4, 3.4, 1.0, "Data pipeline", "pandas · 1,702 GM neighbourhoods")
    _box(ax, 7.6, 6.4, 4.3, 1.0, "Vulnerability index + archetypes", "weighted percentiles · KMeans (k=3)")
    _box(ax, 3.7, 4.65, 3.4, 1.25, "LSTM forecaster", "30 days → next 7 daily temps", ec=RED, lw=2)
    _badge(ax, 5.4, 4.86, "NNPA · ibm-zdnn-plugin", RED)
    _box(ax, 7.6, 4.65, 4.3, 1.25, "P(Cold Weather Payment trigger)", "Monte-Carlo over model errors · DWP rule")
    _box(ax, 3.7, 2.95, 8.2, 1.2, "Risk triage", "risk = vulnerability × P(trigger)  →  Act now / Prepare / Monitor")
    _box(ax, 3.7, 0.7, 4.3, 1.75, "IBM Granite LLM + BM25 RAG", "grounded action per area · number guardrail",
         ec=RED, lw=2)
    _badge(ax, 5.85, 1.05, "NNPA · llama.cpp zDNN build", RED)
    _box(ax, 8.4, 0.7, 3.5, 1.75, "Encrypted register", "AES-256-GCM at rest · pseudonymised IDs", ec=CYAN, lw=2)
    _badge(ax, 10.15, 1.05, "CPACF crypto", CYAN)

    _arrow(ax, 7.1, 6.9, 7.6, 6.9)
    _arrow(ax, 7.1, 5.27, 7.6, 5.27)
    ax.plot([11.9, 12.1, 12.1], [6.9, 6.9, 3.55], color=MUTED, lw=1.6, zorder=1)   # index -> triage (right side)
    _arrow(ax, 12.1, 3.55, 11.9, 3.55)
    _arrow(ax, 9.75, 4.65, 9.75, 4.15)
    _arrow(ax, 5.4, 6.4, 5.4, 5.9)
    _arrow(ax, 5.85, 2.95, 5.85, 2.45)
    _arrow(ax, 10.15, 2.95, 10.15, 2.45)
    _arrow(ax, 8.4, 1.3, 8.0, 1.3, color=CYAN)
    ax.text(8.2, 1.55, "IDs only", ha="center", fontsize=7.5, color=CYAN)

    # source arrows into container
    for i in range(6):
        y = 6.9 - i * 1.12 + 0.42
        tgt = 6.9 if i < 4 else (5.27 if i == 4 else 1.6)
        _arrow(ax, 2.6, y, 3.7, tgt, color=LINE, lw=1.1)

    # outputs
    ax.text(14.25, 7.85, "OUTPUTS", ha="center", fontsize=9, fontweight="bold", color=MUTED)
    outs = [("Interactive map", "timeline replay · live mode"), ("Ranked worklist", "people at risk first"),
            ("Granite briefs", "one action per area"), ("Officer worklist", "names stay encrypted"),
            ("Evidence", "metrics · benchmarks")]
    for i, (t, s) in enumerate(outs):
        _box(ax, 13.1, 6.9 - i * 1.3, 2.3, 0.95, t, s, fs=9.5)
    for i in range(5):                                   # container -> each output
        y = 6.9 - i * 1.3 + .47
        _arrow(ax, 12.3, y, 13.1, y, color=MUTED, lw=1.3)
    return _save(fig, path or IMG / "architecture.png")


# ------------------------------------------------------------------ 2. decision flow
def decision_flow(path=None):
    fig, ax = _canvas(16, 4.6)
    ax.text(0.3, 4.15, "Decision flow: six questions, one action", fontsize=16, fontweight="bold", color=INK)
    steps = [("WHO", "is vulnerable?", "Vulnerability index", "6 weighted inputs", GREEN, None),
             ("WHAT KIND", "of need?", "KMeans archetypes", "3 need types", GREEN, None),
             ("WHEN", "will it freeze?", "LSTM forecast", "7 daily temperatures", RED, "NNPA"),
             ("HOW LIKELY", "is a payment trigger?", "Monte-Carlo P(trigger)", "DWP 7-day ≤ 0 °C rule", AMBER, None),
             ("WHERE", "first?", "Risk triage", "people at risk ranked", AMBER, None),
             ("WHAT", "should we do?", "IBM Granite + RAG", "guardrailed action", RED, "NNPA")]
    w, gap = 2.35, 0.25
    for i, (q, q2, model, sub, col, badge) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 3.2, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc="#ffffff", ec=col, lw=2.2, zorder=2))
        ax.text(x + w / 2, 3.3, q, ha="center", fontsize=13, fontweight="bold", color=col, zorder=3)
        ax.text(x + w / 2, 2.95, q2, ha="center", fontsize=9, color=MUTED, zorder=3)
        ax.plot([x + 0.25, x + w - 0.25], [2.65, 2.65], color=LINE, lw=1, zorder=3)
        ax.text(x + w / 2, 2.15, model, ha="center", fontsize=10, fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, 1.8, sub, ha="center", fontsize=8.3, color=MUTED, zorder=3)
        if badge:
            _badge(ax, x + w / 2, 1.05, "Telum AI accelerator", RED)
        else:
            ax.text(x + w / 2, 1.05, "LinuxONE CPU", ha="center", fontsize=8, color=MUTED, zorder=3)
        if i < len(steps) - 1:
            _arrow(ax, x + w, 2.1, x + w + gap, 2.1, color=INK, lw=1.8)
    return _save(fig, path or IMG / "decision_flow.png")


# ------------------------------------------------------------------ 3. LSTM model
def lstm_model(path=None):
    fig, ax = _canvas(16, 4.4)
    ax.text(0.3, 3.95, "Hazard model: LSTM on IBM Z → probability of a Cold Weather Payment trigger",
            fontsize=15, fontweight="bold", color=INK)
    blocks = [("Input", "30 days × 5 features\ntmean · tmin · tmax\nsin/cos day-of-year", "#ffffff", LINE),
              ("LSTM 32", "return sequences\n+ dropout 0.2", "#fff1f2", RED),
              ("LSTM 16", "+ dropout 0.2", "#fff1f2", RED),
              ("Dense 16 → 7", "change vs today\nfor days +1 … +7", "#fff1f2", RED),
              ("+ today's temp", "7 daily forecasts\n(residual target)", "#ffffff", LINE),
              ("Monte-Carlo ×500", "correlated errors\nfrom validation", "#fffbeb", AMBER),
              ("P(trigger)", "any 7-day mean ≤ 0 °C\nin Nov–Mar, next 7 days", "#ecfeff", CYAN)]
    w, gap = 2.0, 0.27
    for i, (t, s, fc, ec) in enumerate(blocks):
        x = 0.3 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.9), w, 2.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=2, zorder=2))
        ax.text(x + w / 2, 2.75, t, ha="center", fontsize=11, fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, 1.85, s, ha="center", va="center", fontsize=8.4, color=MUTED, zorder=3, linespacing=1.4)
        if i < len(blocks) - 1:
            _arrow(ax, x + w, 2.1, x + w + gap, 2.1, color=INK, lw=1.6)
    ax.plot([0.3 + 1 * (w + gap), 0.3 + 3 * (w + gap) + w], [0.6, 0.6], color=RED, lw=2)
    _badge(ax, 0.3 + 2 * (w + gap) + w / 2, 0.6, "inference on Telum AI accelerator via ibm-zdnn-plugin", RED)
    ax.text(0.3, 0.12, "Train 1991–2019 · validate 2019–22 · test winters 2022–23 → 2025–26 (unseen)  ·  "
            "Brier 0.017 vs 0.039 persistence  ·  AUC 0.956  ·  up to ~30k forecasts/s",
            fontsize=9, color=MUTED)
    return _save(fig, path or IMG / "lstm_model.png")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_all():
    return [architecture(), decision_flow(), lstm_model()]


if __name__ == "__main__":
    for p in render_all():
        print("Saved", p)
