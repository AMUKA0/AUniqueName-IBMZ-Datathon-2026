"""Privacy layer: 'the mainframe knows who Margaret is - the AI never does'.

1. Generate a SYNTHETIC register of vulnerable residents (fake names; phone numbers in Ofcom's
   07700 900xxx drama range; real GM postcodes, but no real people).
2. Encrypt it at rest with AES-256 (AES-GCM via `cryptography` if installed, else `openssl`).
   On IBM Z, AES runs on the CPACF on-chip crypto instructions.
3. On a scoring run: decrypt IN MEMORY only, match residents to 'Act now' areas, and produce
   - an encrypted officer worklist (names + phones, for the council only), and
   - a pseudonymised, aggregated view - the ONLY thing passed to Granite.
Key is kept outside the repo (~/.zcold_key, chmod 600) or in env var ZCOLD_KEY (hex).

CLI: python -m zcold.privacy 2022-12-09
"""
import os, io, sys, hmac, hashlib, secrets, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
from .paths import PROC, ROOT

SECURE = ROOT / "data" / "secure"
REGISTER_ENC = SECURE / "register.enc"
KEY_FILE = Path.home() / ".zcold_key"

FIRST = ["Margaret", "Jean", "Patricia", "Barbara", "Joan", "Dorothy", "Brenda", "Sheila", "Irene",
         "Eileen", "John", "David", "Michael", "Peter", "Brian", "Alan", "Kenneth", "Derek", "Ronald", "Keith"]
LAST = ["Smith", "Jones", "Taylor", "Brown", "Wilson", "Evans", "Thomas", "Roberts", "Walker", "Wright",
        "Thompson", "Hughes", "Edwards", "Green", "Hall", "Wood", "Harris", "Clarke", "Jackson", "Lewis"]


# ------------------------------------------------------------------ key + crypto
def load_key():
    if os.environ.get("ZCOLD_KEY"):
        return bytes.fromhex(os.environ["ZCOLD_KEY"])
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_hex(32))
        KEY_FILE.chmod(0o600)
        print(f"Generated new 256-bit key at {KEY_FILE} (chmod 600, outside the repo)")
    return bytes.fromhex(KEY_FILE.read_text().strip())


def _backend():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa
        return "AES-256-GCM (cryptography)"
    except ImportError:
        pass
    try:
        from Crypto.Cipher import AES  # noqa
        return "AES-256-GCM (pycryptodome)"
    except ImportError:
        return "AES-256-CBC + PBKDF2 (openssl)"


def encrypt(data: bytes, key: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        return b"GCM1" + nonce + AESGCM(key).encrypt(nonce, data, None)
    except ImportError:
        pass
    try:
        from Crypto.Cipher import AES
        c = AES.new(key, AES.MODE_GCM)
        ct, tag = c.encrypt_and_digest(data)
        return b"PCD1" + c.nonce + tag + ct                     # 16-byte nonce, 16-byte tag
    except ImportError:
        r = subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-salt",
                            "-pass", f"pass:{key.hex()}"], input=data, capture_output=True, check=True)
        return b"OSL1" + r.stdout


def decrypt(blob: bytes, key: bytes) -> bytes:
    tag, body = blob[:4], blob[4:]
    if tag == b"GCM1":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM(key).decrypt(body[:12], body[12:], None)
    if tag == b"PCD1":
        from Crypto.Cipher import AES
        return AES.new(key, AES.MODE_GCM, nonce=body[:16]).decrypt_and_verify(body[32:], body[16:32])
    r = subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000",
                        "-pass", f"pass:{key.hex()}"], input=body, capture_output=True, check=True)
    return r.stdout


def pseudonym(person_id: str, key: bytes) -> str:
    """Keyed hash: stable within the council, meaningless outside it, not reversible without the key."""
    return "HH-" + hmac.new(key, person_id.encode(), hashlib.sha256).hexdigest()[:10]


# ------------------------------------------------------------------ synthetic register
def generate_register(rate=0.05, seed=42):
    """~5% of each LSOA's older people living alone are 'on the register'."""
    rng = np.random.default_rng(seed)
    feats = pd.read_csv(PROC / "lsoa_features.csv", dtype={"lsoa21cd": str})
    pcs = pd.read_csv(PROC / "postcode_lsoa_gm.csv", dtype=str)
    by_lsoa = pcs.groupby("lsoa21cd")["postcode"].apply(list)
    rows, pid = [], 0
    for _, r in feats.iterrows():
        n = int(rng.poisson(r.pct_alone_66plus / 100 * r.n_households * rate))
        codes = by_lsoa.get(r.lsoa21cd, [])
        for _ in range(n):
            pid += 1
            rows.append({
                "person_id": f"P{pid:06d}",
                "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                "phone": f"07700 900{rng.integers(0, 1000):03d}",          # Ofcom fiction range
                "postcode": rng.choice(codes) if codes else "",
                "lsoa21cd": r.lsoa21cd,
                "age": int(rng.integers(66, 97)),
                "lives_alone": True,
                "pension_credit": bool(rng.random() < 0.3),
                "medical_equipment": bool(rng.random() < 0.08),
                "synthetic": True,
            })
    return pd.DataFrame(rows)


def create_register():
    key = load_key()
    reg = generate_register()
    SECURE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    REGISTER_ENC.write_bytes(encrypt(reg.to_csv(index=False).encode(), key))
    secs = time.perf_counter() - t0
    print(f"Synthetic register: {len(reg):,} residents in {reg.lsoa21cd.nunique()} LSOAs")
    print(f"Encrypted with {_backend()} in {secs * 1000:.0f} ms -> {REGISTER_ENC} "
          f"({REGISTER_ENC.stat().st_size / 1024:.0f} KB). No plaintext copy is written.")


# ------------------------------------------------------------------ matching
def match(as_of):
    key = load_key()
    reg = pd.read_csv(io.BytesIO(decrypt(REGISTER_ENC.read_bytes(), key)), dtype={"lsoa21cd": str})
    risk = pd.read_csv(PROC / f"lsoa_risk_replay_{as_of}.csv", dtype={"lsoa21cd": str})
    act = risk[risk.priority.str.startswith("1")][["lsoa21cd", "lsoa21nm", "priority", "risk_score", "p_trigger_7d"]]
    hits = reg.merge(act, on="lsoa21cd")
    hits["priority_score"] = (hits.age >= 80).astype(int) + 2 * hits.medical_equipment.astype(int)
    hits = hits.sort_values(["risk_score", "priority_score"], ascending=False)

    # (a) officer worklist - identifiable, stays encrypted on the mainframe
    work_enc = SECURE / f"worklist_{as_of}.enc"
    work_enc.write_bytes(encrypt(hits.to_csv(index=False).encode(), key))

    # (b) what the AI sees - pseudonymised + aggregated, no names/phones/postcodes
    hits["household_ref"] = hits.person_id.map(lambda p: pseudonym(p, key))
    ai_view = hits[["household_ref", "lsoa21nm", "age", "medical_equipment", "pension_credit"]].copy()
    ai_view["age"] = np.where(ai_view.age >= 80, "80+", "66-79")
    agg = (hits.groupby("lsoa21nm").agg(registered_at_risk=("person_id", "size"),
                                        aged_80plus=("age", lambda a: int((a >= 80).sum())),
                                        medical_equipment=("medical_equipment", "sum"))
               .sort_values("registered_at_risk", ascending=False))
    agg.to_csv(PROC / f"register_agg_{as_of}.csv")

    print(f"Decrypted in memory only. {len(hits):,} registered residents are in 'Act now' areas.")
    print(f"Officer worklist (identifiable) -> {work_enc} [encrypted]")
    print("\nWHAT THE MAINFRAME HOLDS (first 3, synthetic):")
    print(hits[["name", "phone", "postcode", "age", "medical_equipment"]].head(3).to_string(index=False))
    print("\nWHAT GRANITE RECEIVES (pseudonymised):")
    print(ai_view.head(3).to_string(index=False))
    print("\nAggregated per area for briefs:\n", agg.head(5).to_string())


def run(as_of="2022-12-09"):
    print("Crypto backend:", _backend())
    if not REGISTER_ENC.exists():
        create_register()
    match(as_of)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "2022-12-09")
