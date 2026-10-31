# Neural Interfaces 2026 — Track 02 | Spectral BCI Decoder

**A fully trained, independently audited EEG decoding model with a completed official Codabench warm-up evaluation.**

[Official competition / Track 02](https://www.codabench.org/competitions/17982/) · [Download the exact accepted ZIP](Agnuxo_SpectralBCI_Track02.zip) · [Original public data (NEMAR nm000290)](https://nemar.org/dataexplorer/detail?dataset_id=nm000290)

Neural Interfaces 2026 Track 02 challenges participants to decode cued mental commands from short EEG windows. This open-source research entry provides an entirely CPU-compatible, causal, interpretable spectral model trained on the official public PROTEUS dataset. No cloud API, participant-ID lookup, hidden competition label or pretrained network is used.

## Official result and scientific evidence

| Item | Verified result |
|---|---|
| Codabench warm-up entry | **1023163** |
| Official evaluation state | **Finished** on 31 October 2026 at 09:14 CET |
| Official rounded score | **0.38 balanced accuracy** |
| Independent cross-subject holdout score | **0.37535842293906807** |
| Training / validation / test | 1,521 / 379 / 755 EEG windows |
| Train / validation / test participants | **6 / 2 / 2, mutually disjoint** |
| Validated official context cells | 8 held-out subject × recording × interface cells |
| Five independent quality-control experiments | **5/5 passed** |
| Exact ZIP SHA-256 | **d5a1f2bef886fcd775b5a41338612f016087f0c6a4b321dc3ef33e27987de0cd** |

![Codabench scoring result](docs/codabench_finished_20261031.png)

The official score reproduces the independently calculated held-out score after rounding. The reference floor for three equally likely classes is approximately 0.333, but that theoretical value is not a substitute for a same-data head-to-head baseline. **This is a warm-up submission, not a sealed-phase competition entry or a prize result.** The sealed phase opens 5 November and closes 21 November 2026.

## Method

Each input is one `(B, 41, 480)` EEG window sampled at 120 Hz. The deterministic feature extractor subtracts each observed channel's temporal mean, then calculates a Hann-tapered Fourier transform with six bandpowers: 4–8, 8–12, 12–16, 16–22, 22–30 and 30–40 Hz. Absolute log-power and across-channel relative log-power yield 492 interpretable features. A class-balanced multinomial logistic regression is trained on these features, with scaling statistics frozen from the public training partition.

Model selection tested C = 0.005, 0.05, 0.5 and 5.0 on the validation participants only; C = **0.005** was selected. The final frozen model was refitted on train plus validation subjects, and the two held-out participants were evaluated only after selecting the model. Predicted labels are 0, 1 and 2, in the official order. There are no recurrent cross-session states, and no participant metadata is used during inference.

## Independent checks

All five external checks executed on the *exact packaged ZIP*, with seed **20261031**. Their machine-readable raw results are in [independent_20261031.json](independent_20261031.json).

1. ZIP structure, 492-feature dimensions, valid classes and finite coefficients: **PASS**.
2. Identical predictions for batch mode and one-window streaming across 30 windows: **PASS**, zero disagreements.
3. Perturbing future windows cannot change the first 11 decisions: **PASS**, zero leakage violations.
4. All-zero, NaN, infinity and grossly outlying input windows return valid finite class IDs: **PASS**.
5. Eight-cell holdout accuracy independently recalculated to **0.3753584229**. A 2,000-resample cell bootstrap gave **95% CI [0.34896, 0.40868]**: **PASS**.

## Files

| File | Purpose |
|---|---|
| [submission.py](submission.py) | Self-contained frozen inference implementation conforming to `CompetSolver` |
| [weights.npz](weights.npz) | Trained real-data multinomial coefficients and scaling |
| [Agnuxo_SpectralBCI_Track02.zip](Agnuxo_SpectralBCI_Track02.zip) | Exact file accepted and scored by the official Codabench portal |
| [train_public.py](train_public.py) | Deterministic train / validation / held-out evaluation and ZIP packaging |
| [audit.py](audit.py) | Independent tests on the exact packaged entry |
| [training_results.json](training_results.json) | Original raw split assignments, model comparison and measured results |
| [independent_20261031.json](independent_20261031.json) | Audited results and uncertainty estimates |
| [kit_compat.py](kit_compat.py) | Explicit documented Windows/NeuralBench 0.4.0 compatibility layer |
| [docs/](docs/) | Official scored warm-up screenshot |

## Reproduce

Use Python 3.11 or later, the public [organiser's starter kit](https://github.com/neural-interfaces26/2026-competition) pinned to commit **2b53aec91c111f8b5423b6674f096c48cc1a99af**, and the legitimately downloaded public [PROTEUS NEMAR nm000290](https://nemar.org/dataexplorer/detail?dataset_id=nm000290). Reproduction uses the *organiser's* train/validation/test split and preprocessing; no raw EEG data is redistributed in this repository.

First clone this repository and the organisers' starter kit, then install the dependencies using `python -m pip install -r requirements.txt` in a virtual environment. Download the public PROTEUS dataset through NeuralBench or the starter kit; set `BCI_KIT_DIR` to the top-level starter-kit directory and `BCI_DATA_DIR` to the directory holding `Dreyer2026Proteus`. Run `python train_public.py` to refit the model. `python audit.py` runs the independent package checks using the logged public test features; for auditing the original result, the checked-in JSON is the preserved raw output. The official ZIP can be uploaded directly without retraining.

The original development workstation used the public data directory `E:\Proyecto-concursos\NeuralTrack02\public_data`, a 6-CPU-thread local run, and did not use Server-IA or paid Codex API. All training code and the full verification history are preserved. To run a new offline test on a fresh machine, first prepare the organisers' study and then execute the training pipeline; the downloaded raw data must remain outside Git.

## Expected applications

This compact approach is suitable for research on subject-independent BCI decoding, transparent comparisons between neurophysiological band features and learned neural encoders, and low-compute experimentation. It is not a clinical or medical-device system.

## Observations and limitations

The public evaluation consists of only two held-out participants (eight official subject/session/context cells), so the interval reflects a limited sample. Subject shift remains substantial. The 0.38 result is specific to this public warm-up task and cannot be extrapolated to the hidden sealed phase or a leaderboard win. No unreleased data or paid pretrained network was used.

**Author:** Francisco Angulo de Lafuente · [Agnuxo1](https://github.com/Agnuxo1) · Spain. New source code: MIT License. NEMAR data and the organisers' kit retain their own terms.
