"""[1] Vulnerability index + [1b] vulnerability archetypes (KMeans)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .paths import PROC

# feature: (weight, plain-English driver label used in Granite briefs)
WEIGHTS = {
    "pct_75plus":       (0.20, "high share of residents aged 75+"),
    "pct_alone_66plus": (0.20, "many people aged 66+ living alone"),
    "indoors_score":    (0.20, "poor housing condition / heating"),
    "idaopi_score":     (0.15, "older people on low incomes"),
    "pct_disabled":     (0.15, "high disability rate"),
    "pct_no_car":       (0.10, "low car access"),
}
SHORT = {"pct_75plus": "very elderly", "pct_alone_66plus": "older people alone",
         "indoors_score": "poor housing", "idaopi_score": "low-income pensioners",
         "pct_disabled": "disability", "pct_no_car": "no car",
         "income_score": "income-deprived", "health_score": "poor health"}
ARCH_FEATURES = list(WEIGHTS) + ["income_score", "health_score"]
BANDS = ["Very low", "Low", "Medium", "High", "Very high"]


def load_features():
    return pd.read_csv(PROC / "lsoa_features.csv", dtype={"lsoa21cd": str})


def vulnerability_index(df):
    df = df.copy()
    pct = pd.DataFrame({f: df[f].rank(pct=True) for f in WEIGHTS})   # 0-1 within GM
    contrib = pct * pd.Series({f: w for f, (w, _) in WEIGHTS.items()})
    df["vuln_index"] = (contrib.sum(axis=1) * 100).round(1)
    order = np.argsort(-contrib.to_numpy(), axis=1)[:, :2]
    cols = contrib.columns.to_numpy()
    df["driver_1"] = [WEIGHTS[cols[i]][1] for i in order[:, 0]]
    df["driver_2"] = [WEIGHTS[cols[i]][1] for i in order[:, 1]]
    df["vuln_band"] = pd.qcut(df["vuln_index"], 5, labels=BANDS)
    return df


def validate(df):
    rows = []
    for v in ["fuel_poor_pct", "imd_score", "health_score"]:
        rho, p = spearmanr(df["vuln_index"], df[v])
        rows.append((v, round(rho, 2), f"{p:.1e}"))
    return pd.DataFrame(rows, columns=["benchmark", "spearman_rho", "p_value"])


def archetypes(df, k_range=range(3, 7), seed=42):
    X = StandardScaler().fit_transform(df[ARCH_FEATURES])
    sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=seed).fit_predict(X))
           for k in k_range}
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    cent = pd.DataFrame(km.cluster_centers_, columns=ARCH_FEATURES)
    names = {}
    for i, row in cent.iterrows():
        top = [f for f in row.nlargest(2).index if row[f] > 0.3]
        label = " + ".join(SHORT[f] for f in top) if top else "lower need"
        tags = []
        if row["income_score"] < -0.5: tags.append("affluent")
        if row["pct_75plus"] < -0.5: tags.append("younger")
        names[i] = f"A{i}: " + (", ".join(tags) + " – " if tags else "") + label
    df = df.copy()
    df["archetype"] = pd.Series(km.labels_, index=df.index).map(names)
    cent.index = [names[i] for i in cent.index]
    return df, sil, cent.round(2)


def run():
    df = vulnerability_index(load_features())
    print("Validation (Spearman):\n", validate(df).to_string(index=False))
    df, sil, cent = archetypes(df)
    print("\nSilhouette by k:", {k: round(s, 3) for k, s in sil.items()})
    print("\nArchetype centroids (z-scores):\n", cent.to_string())
    summary = df.groupby("archetype").agg(n_lsoas=("lsoa21cd", "size"),
                                          households=("n_households", "sum"),
                                          mean_vuln=("vuln_index", "mean")).round(1)
    print("\n", summary.to_string())
    top = df.sort_values("vuln_index", ascending=False).head(10)
    print("\nTop 10 LSOAs:\n", top[["lsoa21nm", "vuln_index", "archetype",
                                     "driver_1", "driver_2"]].to_string(index=False))
    print("\nHouseholds in 'Very high' band:",
          int(df.loc[df.vuln_band == "Very high", "n_households"].sum()))
    df.to_csv(PROC / "lsoa_vulnerability.csv", index=False)
    print("Saved", PROC / "lsoa_vulnerability.csv")


if __name__ == "__main__":
    run()
