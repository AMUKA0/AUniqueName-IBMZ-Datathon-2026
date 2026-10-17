# ColdWatch: real-time cold-weather triage on IBM LinuxONE

**Team AUniqueName · IBM Z Datathon 2026 · Event code SSF2026**

> Which neighbourhoods need help *before* the freeze, and what should outreach teams do there?

ColdWatch combines a **vulnerability index** for every neighbourhood in Greater Manchester with an **LSTM freeze forecast running on the IBM Telum on-chip AI accelerator**, ranks where to act first, and uses **IBM Granite** to write a grounded, guardrailed action brief for each priority area. Personal data stays encrypted on the mainframe; the AI only ever sees pseudonymised aggregates.

**Theme:** Real-time AI for Critical Decisions (*Detect · Decide · Respond*), with an *AI Secured* privacy layer.

---

## The problem
Cold homes are a serious health risk for older, isolated and disabled people. Help exists (Cold Weather Payments, the Warm Home Discount, council welfare calls), but it is **reactive**: a Cold Weather Payment only triggers once an area's average temperature has been at or below 0 °C for 7 consecutive days, and outreach teams have no ranked view of *which streets to prioritise* or *what kind of help* each needs.

## What ColdWatch does
| Question | Model | Runs on |
|---|---|---|
| **Who** is vulnerable? | Transparent weighted index (6 Census / IoD inputs, percentile-ranked) | LinuxONE CPU |
| **What kind** of need? | KMeans archetypes (3 need types) | LinuxONE CPU |
| **When** will it freeze? | LSTM forecasting each of the next 7 days' mean temperature | **Telum AI accelerator** (`ibm-zdnn-plugin`) |
| **How likely** a payment trigger? | Monte-Carlo over the LSTM's correlated errors, using DWP's 7-day ≤ 0 °C rule | LinuxONE CPU |
| **Where** first? | Risk = vulnerability × P(trigger) → Act now / Prepare / Monitor, ranked by people at risk | LinuxONE CPU |
| **What** should we do? | IBM Granite + BM25 retrieval over GOV.UK policy, with a numeric guardrail | **Telum AI accelerator** (llama.cpp zDNN build) |
| **Who may see what?** | AES-256-GCM encrypted register; pseudonymised IDs passed to AI | LinuxONE |

## Architecture

```mermaid
flowchart LR
    subgraph SRC["Open data"]
        IOD["IoD 2025"]
        CEN["Census 2021"]
        ONS["ONS postcodes + LSOA boundaries"]
        DWP["DWP Cold Weather Payment stations"]
        WX["ERA5 history + Open-Meteo live"]
        POL["GOV.UK policy text"]
    end

    subgraph Z["IBM LinuxONE (z17) · Docker container · Telum on-chip AI accelerator"]
        PIPE["Data pipeline<br/>pandas · 1,702 neighbourhoods"]
        VULN["Vulnerability index<br/>+ KMeans archetypes"]
        LSTM["LSTM forecaster<br/>Telum AI accelerator · ibm-zdnn-plugin"]
        PTRIG["P(trigger)<br/>Monte-Carlo · DWP 7-day ≤ 0 °C rule"]
        TRI["Risk triage<br/>Act now / Prepare / Monitor"]
        GRAN["IBM Granite + BM25 RAG<br/>Telum AI accelerator · llama.cpp zDNN · guardrail"]
        REG["Encrypted register<br/>AES-256-GCM · pseudonymised IDs"]
    end

    subgraph OUT["Outputs"]
        MAP["Interactive map<br/>timeline replay · live"]
        WL["Ranked worklist"]
        BR["Granite briefs"]
        OW["Officer worklist (encrypted)"]
    end

    IOD & CEN & ONS & DWP --> PIPE --> VULN --> TRI
    WX --> LSTM --> PTRIG --> TRI
    TRI --> GRAN
    TRI --> REG
    REG -- "IDs only" --> GRAN
    POL --> GRAN
    TRI --> MAP & WL
    GRAN --> BR
    REG --> OW
```

### Hazard model

```mermaid
flowchart LR
    I["30 days × 5 features"] --> L1["LSTM 32 + dropout"] --> L2["LSTM 16 + dropout"] --> D["Dense 16 → 7<br/>change vs today"]
    D --> T["+ today's temp<br/>7 daily forecasts"] --> MC["Monte-Carlo ×500<br/>correlated errors"] --> P["P(trigger)<br/>any 7-day mean ≤ 0 °C, Nov–Mar"]
```

Rendered diagrams: [`images/architecture.png`](images/architecture.png) · [`images/decision_flow.png`](images/decision_flow.png) · [`images/lstm_model.png`](images/lstm_model.png)

---

## How IBM Z / LinuxONE is used
| Capability | Where | Evidence |
|---|---|---|
| **Telum on-chip AI accelerator (NNPA)** via `ibm-zdnn-plugin` | LSTM inference (TensorFlow 2.18) | `nnpa` CPU facility; `_z_ops.so` and `libzdnn.so` loaded in the inference process; batch-throughput benchmark |
| **Telum AI accelerator** via llama.cpp built with `-DGGML_ZDNN=ON` | IBM Granite 4.0 briefs (big-endian GGUF for s390x) | `libggml-zdnn` backend library |
| **CPACF on-chip crypto** | Platform AES for encryption at rest | `msa` CPU facility flags |
| **Containerised workload** | Entire pipeline runs in the LinuxONE `notebook` container on a persistent volume | `docker ps` / `docker inspect` |

Screenshots: [`evidence/`](evidence/)

## Results (unseen test winters 2022–23 → 2025–26)
*Values from the executed notebook [`ColdWatch_Story.ipynb`](ColdWatch_Story.ipynb).*

| Metric | LSTM | Persistence baseline |
|---|---|---|
| Brier score (lower = better) | **0.017** | 0.039 |
| AUC | **0.956** | — |
| Precision / recall at P ≥ 0.5 | **61% / 52%** | 38% / 70% |
| False alarms | **11** | 37 |
| Forecast error, day 7 | **2.59 °C** | 3.29 °C |

- **Replay of the December 2022 cold spell:** the Rochdale station area moved to *Prepare* 5 days and *Act now* 3 days before the first freezing week completed. On 9 Dec 2022: **268 neighbourhoods on Act now, about 29,400 older people living alone**.
- **Throughput:** up to about **30,000 LSTM forecasts per second** on IBM Z (batch 10,000).
- **Index validation:** Spearman ρ = 0.32 against official fuel poverty, 0.33 against IMD and health deprivation.

---

## Repository structure
```
zcold/                    # Python package (all models + pipeline)
  structural.py           # [1] vulnerability index + KMeans archetypes
  hazard.py               # [2] LSTM on IBM Z, Monte-Carlo P(trigger), benchmark, live forecast
  scoring.py              # [3] risk triage (replay or live)
  privacy.py              # encrypted synthetic register + pseudonymised AI view
  briefs.py               # [4] IBM Granite + BM25 briefs with guardrail
  mapviz.py               # interactive map (timeline replay, live mode)
  diagrams.py             # architecture / flow diagrams
src/                      # data extraction scripts (IoD, Census, ONSPD, boundaries, stations, ERA5, labels)
ColdWatch_Story.ipynb     # end-to-end run-through on LinuxONE (executed, with outputs)
images/                   # rendered diagrams
models/                   # trained LSTM + metrics
outputs/                  # interactive map HTML (prototype)
evidence/                 # platform evidence screenshots
```

## How to run (LinuxONE Community Cloud)
Everything runs inside the provided `notebook` container (Python `/opt/venv`, TensorFlow + `ibm-zdnn-plugin`, llama.cpp zDNN build). The project lives in the shared volume: host `~/shared/zcold` = container `/home/ibm-user/shared/zcold`.

```bash
# extra packages (pure Python / C, no TensorFlow or NumPy changes)
docker exec notebook /opt/venv/bin/pip install -r /home/ibm-user/shared/zcold/requirements.txt

# 1. data: download sources listed below into data/raw/, then run the extract scripts in src/
# 2. pipeline
cd ~/shared/zcold
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.structural
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.hazard
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.scoring 2022-12-09
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.privacy 2022-12-09
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.briefs 2022-12-09
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.mapviz 2022-12-09
docker exec -w /home/ibm-user/shared/zcold notebook python -m zcold.mapviz live
```
Or open `ColdWatch_Story.ipynb` in Jupyter and **Run All**.

Data files are not committed (size and licence terms); all sources are open and listed below.

## Data sources
| Dataset | Publisher | Licence |
|---|---|---|
| [English Indices of Deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025) | MHCLG | OGL v3.0 |
| [Census 2021 bulk tables](https://www.nomisweb.co.uk/sources/census_2021_bulk) (TS007A, TS003, TS038, TS045) | ONS / Nomis | OGL v3.0 |
| [ONS Postcode Directory, May 2026](https://open-geography-portalx-ons.hub.arcgis.com/datasets/ons::ons-postcode-directory-may-2026) | ONS | OGL v3.0 (contains Royal Mail, OS and ONS data) |
| [LSOA December 2021 boundaries (BGC)](https://open-geography-portalx-ons.hub.arcgis.com/datasets/ons::lower-layer-super-output-areas-december-2021-boundaries-ew-bgc-v5-2) | ONS | OGL v3.0 (contains OS data) |
| [Sub-regional fuel poverty, 2024 data](https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2026-2024-data) (validation only) | DESNZ | OGL v3.0 |
| [Cold Weather Payment estimates + station/postcode file](https://www.gov.uk/government/statistics/cold-weather-payment-estimates-2025-to-2026) | DWP | OGL v3.0 |
| [UK synoptic and climate stations](https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-synoptic-and-climate-stations) (coordinates) | Met Office | Met Office terms |
| [Historical Weather API (ERA5)](https://open-meteo.com/en/docs/historical-weather-api) and Forecast API | Open-Meteo / Copernicus | CC BY 4.0 |
| [Cold Weather Payment](https://www.gov.uk/cold-weather-payment), [Warm Home Discount](https://www.gov.uk/the-warm-home-discount-scheme) policy text | GOV.UK | OGL v3.0 |
| Basemap tiles | Esri (World Dark/Light Gray Canvas) | Esri terms |

## Privacy and responsible AI
- **Privacy by design:** the resident register is **synthetic** (fake names, Ofcom fiction-range phone numbers), encrypted with **AES-256-GCM** at rest, and decrypted only in memory. Granite receives **pseudonymised, aggregated** inputs only. Names never reach the AI.
- **Grounded generation:** code writes every number in a brief; Granite writes one action line from retrieved policy text; a **guardrail rejects any number not present in the source facts** and substitutes an approved action.
- **Human in the loop:** ColdWatch prioritises outreach; it does not decide entitlement. Decisions stay with council officers.

## Limitations
- Area-level (LSOA) data identifies *where* to look, not which individual is at risk.
- ERA5 reanalysis approximates, but is not identical to, the Met Office station readings DWP uses.
- Vulnerability weights are expert judgement (no LSOA-level outcome data to learn them from); they are transparent and validated against official measures.
- The LSTM learns from past temperatures, so it cannot foresee the *onset* of a cold snap; production use would add weather-forecast inputs.
- Two DWP stations cover Greater Manchester, so within a station area ranking is driven by vulnerability.
- Two LSOAs (central Manchester, districts M60/M61) are mapped by inference; see `src/station_mapping.py`.

## Open-source software
[TensorFlow](https://github.com/tensorflow/tensorflow) ·
[ibm-zdnn-plugin](https://pypi.org/project/ibm-zdnn-plugin/) ·
[IBM zDNN](https://github.com/IBM/zDNN) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp) ·
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) ·
[IBM Granite](https://huggingface.co/ibm-granite) ·
[rank_bm25](https://github.com/dorianbrown/rank_bm25) ·
[scikit-learn](https://github.com/scikit-learn/scikit-learn) ·
[SciPy](https://github.com/scipy/scipy) ·
[pandas](https://github.com/pandas-dev/pandas) ·
[NumPy](https://github.com/numpy/numpy) ·
[Matplotlib](https://github.com/matplotlib/matplotlib) ·
[folium](https://github.com/python-visualization/folium) ·
[Leaflet](https://github.com/Leaflet/Leaflet) ·
[PyCryptodome](https://github.com/Legrandin/pycryptodome) ·
[openpyxl](https://foss.heptapod.net/openpyxl/openpyxl) ·
[odfpy](https://github.com/eea/odfpy)

## Acknowledgements
- IBM LinuxONE Community Cloud and the IBM Z Datathon AI labs. Patterns adapted from the IBM reference notebooks (*Loan Approval: Predictive + Generative AI* and *Fraud detection LSTM*), credited here. All ColdWatch code was written during the datathon (17–18 October 2026).
- Shooting Stars Foundation and the IBM Z mentors.

## Team
- Hector Pearson: models, pipeline, map, notebook
- Daniel *(surname)*: data engineering, weather stations, labels, policy integration
