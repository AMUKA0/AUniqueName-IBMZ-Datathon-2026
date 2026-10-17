# Evidence index

| ID | File | What it shows | Rubric area |
|---|---|---|---|
| A1 | A1_lab_fraud_lstm.png | IBM lab completed: fraud detection LSTM | Lab completion |
| A2 | A2_lab_loan_predictive_generative.png | IBM lab completed: loan approval, predictive + generative | Lab completion |
| B0 | B0_vulnerability_validation.png | Vulnerability index validated (Spearman ρ ≈ 0.33), 3 KMeans archetypes | AI / methodology |
| B1 | B1_cpu_facilities.png | z17 CPU facilities incl. NNPA (Telum AI accelerator) | IBM Z hardware |
| B2 | B2_zdnn_tensorflow.png | ibm-zdnn-plugin registered with TensorFlow | IBM Z hardware |
| B3 | B3_zdnn_loaded_in_inference.png | libzdnn loaded during LSTM inference | IBM Z hardware |
| B4 | B4_granite_zdnn_build.png | llama.cpp built with zDNN backend | IBM Z hardware |
| B5 | B5_lstm_training.png | LSTM training run (seed 42) | AI model |
| B6 | B6_lstm_metrics.png | Test winters: MAE, Brier, false alarms vs persistence | AI model / validation |
| B7 | B7_granite_generation.png | Granite briefs + number guardrail | Generative AI |
| C1 | C1_throughput_benchmark.png | Batch inference up to ~30,000 sequences/s | Performance |
| C2 | C2_national_scale.png | All 33,755 English LSOAs scored in 1.5 s | Scalability |
| C3 | C3_resources.png | CPU / memory footprint | Performance |
| D0 | D0_environment.png | LinuxONE environment overview | Deployment |
| D1 | D1_docker_ps.png | Running container on LinuxONE | Deployment |
| D2 | D2_docker_inspect.png | Container image / mounts | Deployment |
| D3 | D3_pipeline_in_container.png | Full pipeline executed in container | Deployment |
| E1 | E1_privacy_split.png | Mainframe holds identities; Granite receives pseudonyms only | Privacy (Theme 2) |
| E2 | E2_encrypted_at_rest.png | Worklist encrypted at rest (AES-256-GCM) | Privacy / security |
| E3 | E3_crypto_backend.png | Crypto backend in use | Privacy / security |
| F1 | F1_dec2022_warning_ramp.png | Dec 2022 replay: Prepare 5 days, Act 3 days before freeze | Impact / validation |
| F2 | F2_triage_summary.png | 268 Act now / 411 Prepare / 1,023 Monitor (9 Dec 2022) | Impact |
| F3 | F3_map_triage.png | Interactive triage map | Demo / UX |
| F4 | F4_map_brief_card.png | Map card with Granite brief | Demo / UX |
| F5 | F5_map_14dec.png | Timeline replay, 14 Dec 2022 | Demo / UX |
| F6 | F6_map_live.png | Live mode (current forecast) | Real-time |
| — | zdnn_check.txt | Raw zDNN library check output | IBM Z hardware |
| — | ColdWatch_Story_executed.html | Full executed notebook | Reproducibility |