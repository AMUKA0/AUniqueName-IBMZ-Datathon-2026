"""[5] Outreach briefs: IBM Granite (llama.cpp built with zDNN) + BM25 policy retrieval.

Design for a small base model (pattern from IBM's loan_demo/rag_utils.py):
  - CODE writes every fact (area, freeze chance, people, drivers) -> no invented numbers
  - GRANITE writes ONE action line, grounded in retrieved policy text, via a one-example prompt
  - temperature 0, short max_tokens, stop at newline
Privacy: Granite only ever receives area-level aggregates - never names or addresses.

CLI: python -m zcold.briefs 2022-12-09     (uses lsoa_risk_replay_2022-12-09.csv)
     python -m zcold.briefs                (uses lsoa_risk.csv)
"""
import os, re, sys, glob, time
import pandas as pd
from rank_bm25 import BM25Okapi
from .paths import RAW, PROC

BASE_MODEL = "/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf"
INSTRUCT_MODEL = "/home/ibm-user/granite-3.3-2b-instruct-be.Q4_K_M.gguf"   # used if present
POLICY_DIR = RAW / "policy"

# Fallback policy notes (short factual summaries of GOV.UK pages) - used only if POLICY_DIR is empty
FALLBACK_POLICY = {
    "Cold Weather Payment (GOV.UK)":
        "Cold Weather Payment: paid when the average temperature in an area is recorded or forecast "
        "to be zero degrees Celsius or below over 7 consecutive days. 25 pounds for each 7-day period "
        "of very cold weather between 1 November 2026 and 31 March 2027. Paid automatically to people "
        "getting certain benefits, such as Pension Credit. Eligible people do not need to apply.",
    "Warm Home Discount 2026-27 (GOV.UK)":
        "Warm Home Discount: a one-off 150 pound discount off the electricity bill for winter 2026 to "
        "2027. The qualifying date is 23 August 2026. Most eligible households get it automatically, "
        "for example people getting Pension Credit or other means-tested benefits. People who think "
        "they qualify but have not received a letter should contact the Warm Home Discount helpline.",
    "ColdWatch outreach checklist (team-written)":
        "Outreach actions before a freeze: phone or visit older people living alone, check the heating "
        "works and is affordable to run, check food and medicine supplies, signpost local warm spaces, "
        "and check whether residents claim Pension Credit, which unlocks Cold Weather Payments and the "
        "Warm Home Discount. For disabled residents, check care arrangements and transport to appointments.",
}

# One worked example per need type: Granite adapts the matching pattern to each area.
# The same actions double as the guardrail fallback if generation is rejected.
APPROVED_ACTIONS = {
    "disability": "Prioritise visits to disabled and unwell residents, check care arrangements and "
                  "heating, and check Pension Credit claims so Cold Weather Payments are paid.",
    "alone":      "Make welfare calls to older people living alone, check the heating is on and food "
                  "and medicine supplies are in, and signpost local warm spaces.",
    "housing":    "Target homes with poor heating, give advice on running heating affordably, and "
                  "signpost warm spaces and help with transport.",
}
EXAMPLES = (
    "Area: Example 001A | Need: disability + poor health | Freeze chance: 70% | Older people living alone: about 120\n"
    f"Action: {APPROVED_ACTIONS['disability']}\n\n"
    "Area: Example 002B | Need: affluent older people alone | Freeze chance: 60% | Older people living alone: about 200\n"
    f"Action: {APPROVED_ACTIONS['alone']}\n\n"
    "Area: Example 003C | Need: younger, poor housing + no car | Freeze chance: 65% | Older people living alone: about 40\n"
    f"Action: {APPROVED_ACTIONS['housing']}\n")


def approved_action(need):
    n = need.lower()
    if "disab" in n or "health" in n:
        return APPROVED_ACTIONS["disability"]
    if "housing" in n or "car" in n:
        return APPROVED_ACTIONS["housing"]
    return APPROVED_ACTIONS["alone"]


def guardrail(text, allowed_text):
    """Reject output that is empty or contains a number not present in the prompt facts/policy."""
    if len(text.split()) < 6:
        return False, "too short"
    bad = set(re.findall(r"\d+", text)) - set(re.findall(r"\d+", allowed_text))
    return (False, f"unsupported numbers {sorted(bad)}") if bad else (True, "ok")


# ------------------------------------------------------------------ retrieval
def load_policy():
    docs = {}
    for path in sorted(glob.glob(str(POLICY_DIR / "*"))):
        name = os.path.basename(path)
        try:
            if path.lower().endswith(".pdf"):
                from pypdf import PdfReader
                text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
            else:
                text = open(path, encoding="utf-8", errors="ignore").read()
            text = re.sub(r"<[^>]+>", " ", text)              # strip HTML if saved as .html
            docs[name] = re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            print(f"  skipped {name}: {e}")
    if not docs:
        print("No files in data/raw/policy -> using built-in fallback policy notes")
        docs = dict(FALLBACK_POLICY)
    return docs


def build_index(docs, chunk_words=60):
    chunks, sources = [], []
    for name, text in docs.items():
        words = text.split()
        for i in range(0, len(words), chunk_words):
            chunks.append(" ".join(words[i:i + chunk_words])); sources.append(name)
    bm25 = BM25Okapi([c.lower().split() for c in chunks])
    return bm25, chunks, sources


def retrieve(bm25, chunks, sources, query, k=2):
    scores = bm25.get_scores(query.lower().split())
    top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return [chunks[i] for i in top], sorted({sources[i] for i in top})


# ------------------------------------------------------------------ model
def load_granite(n_ctx=4096, n_threads=None):
    from llama_cpp import Llama
    path = INSTRUCT_MODEL if os.path.exists(INSTRUCT_MODEL) else BASE_MODEL
    n_threads = n_threads or os.cpu_count()
    print(f"Loading Granite: {os.path.basename(path)} (llama.cpp zDNN build, {n_threads} threads)")
    return Llama(model_path=path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def action_line(llm, context, area_line, max_tokens=60):
    prompt = ("You are a council welfare planning assistant. For each area, write ONE action line "
              "for the outreach team that fits the area's need, using only the policy notes.\n\n"
              f"Policy notes:\n{context}\n\n{EXAMPLES}\n{area_line}\nAction:")
    t0 = time.perf_counter()
    out = llm(prompt, max_tokens=max_tokens, temperature=0, repeat_penalty=1.1, stop=["\n", "Area:"])
    secs = time.perf_counter() - t0
    text = out["choices"][0]["text"].strip()
    toks = out.get("usage", {}).get("completion_tokens", 0)
    return text, secs, toks


# ------------------------------------------------------------------ briefs
def _need(archetype):
    return re.sub(r"^A\d+:\s*", "", str(archetype))


def make_brief(llm, idx, row):
    alone = row.get("alone_66plus_hh", row["pct_alone_66plus"] / 100 * row["n_households"])
    need = _need(row["archetype"])
    query = f"{need} {row['driver_1']} {row['driver_2']} cold weather freeze heating pension credit"
    ctx_chunks, srcs = retrieve(*idx, query)
    area_line = (f"Area: {row['lsoa21nm']} | Need: {need} | Freeze chance: {row['p_trigger_7d']:.0%} "
                 f"| Older people living alone: about {alone:.0f}")
    context = "\n".join(ctx_chunks)
    action, secs, toks = action_line(llm, context, area_line)
    ok, why = guardrail(action, context + " " + area_line + " " + EXAMPLES)
    if not ok:
        print(f"  guardrail rejected Granite output ({why}): {action!r}")
        action = approved_action(need) + " [approved fallback]"
    text = (f"[{row['priority'].split(' - ')[-1].upper()}] {row['lsoa21nm']} ({row['lad_name']}) - "
            f"freeze chance {row['p_trigger_7d']:.0%}, vulnerability {row['vuln_index']:.0f}/100\n"
            f"Why: {row['driver_1']}; {row['driver_2']}. Need type: {need}.\n"
            f"Reach: ~{row['n_households']:.0f} households, ~{alone:.0f} older people living alone.\n"
            f"Action (Granite): {action}\n"
            f"Sources: {', '.join(srcs)}")
    return {"lsoa21cd": row["lsoa21cd"], "lsoa21nm": row["lsoa21nm"], "archetype": row["archetype"],
            "brief": text, "granite_action": action, "guardrail": why,
            "gen_seconds": round(secs, 2), "gen_tokens": toks}


def select_areas(df, n=5):
    """Top area for each need type (so briefs show tailoring), then the next highest-priority areas."""
    pri = df[df.priority.str.startswith("1")]
    pool = pri if not pri.empty else df
    picks = pool.groupby("archetype", sort=False).head(1)
    missing = df[~df.archetype.isin(picks.archetype)].groupby("archetype", sort=False).head(1)
    picks = pd.concat([picks, missing])
    rest = pool[~pool.lsoa21cd.isin(picks.lsoa21cd)]
    return pd.concat([picks, rest]).head(max(n, len(picks)))


def run(as_of=None, n=5):
    path = PROC / ("lsoa_risk.csv" if as_of is None else f"lsoa_risk_replay_{as_of}.csv")
    df = pd.read_csv(path, dtype={"lsoa21cd": str})
    top = select_areas(df, n)
    print(f"Briefing {len(top)} areas ({top.archetype.nunique()} need types) from {path.name}")
    idx = build_index(load_policy())
    llm = load_granite()
    rows = []
    for _, r in top.iterrows():
        b = make_brief(llm, idx, r)
        rows.append(b)
        print("\n" + b["brief"] + f"\n  (generated in {b['gen_seconds']}s, {b['gen_tokens']} tokens)")
    out = pd.DataFrame(rows)
    tot_t, tot_s = out.gen_tokens.sum(), out.gen_seconds.sum()
    print(f"\nGranite on IBM Z: {tot_t} tokens in {tot_s:.1f}s ({tot_t / max(tot_s, 1e-9):.1f} tokens/s)")
    dest = PROC / f"briefs_{as_of or 'live'}.csv"
    out.to_csv(dest, index=False)
    print("Saved", dest)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
