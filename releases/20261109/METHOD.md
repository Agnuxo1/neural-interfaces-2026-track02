# Spectral BCI Decoder — first sealed reference

Author: Francisco Angulo de Lafuente. ORCID: https://orcid.org/0009-0001-1634-7063.

## Results and evidence

This frozen, trained model completed the public warm-up as entry 1023163, with rounded balanced accuracy 0.38 on 31 October 2026. Its code and weights were recovered exactly. Three new CPU checks exercised the current, complete Solver and Objective classes; a separate synthetic control distinguished cell, session and pooled scoring. These are compatibility checks. This first sealed reference predicts neither a performance improvement nor a sealed score.

## Method and training data

The only training dataset was the public PROTEUS BCI Bordeaux release, NEMAR nm000290 v1.0.0, DOI https://doi.org/10.82901/nemar.nm000290 (Dreyer Pauline, Bourdil Manon, Bechon Loic and contributors). The organisers' current Scientific FAQ identifies this EEG-BIDS release as CC BY 4.0. No raw EEG, feature cache, subject identifier table or study labels are included. No other dataset, external pretrained model, cloud API or paid model was used.

The original public protocol used six training participants (1,521 windows), two validation participants (379 windows) and two mutually disjoint public test participants (755 windows). Four C values (0.005, 0.05, 0.5, 5.0) were compared only on validation participants; C = 0.005 was frozen before public test evaluation. Scaling and logistic coefficients were refitted on train plus validation (1,900 windows, eight participants). The two public test participants were excluded from fitting. The warm-up holdout does not reveal the sealed allocation.

For each four-second window with 41 EEG channels at 120 Hz, inference computes six Hann-tapered FFT bandpowers (4–8, 8–12, 12–16, 16–22, 22–30, 30–40 Hz), absolute log power and channel-relative log power (492 features), then frozen scaling and a class-balanced multinomial linear decision rule. Classes 0, 1 and 2 follow the official order. The model uses no subject or session metadata, labels, future windows, online fitting, network or cross-session state. NumPy inference accepts the supplied Torch tensor through a CPU copy.

## Compute and reproducibility

Original training ran on a six-thread CPU workstation, without GPU. The preserved log records 6.88 seconds elapsed for its script; previously downloaded public data and one-time environment acquisition are separate. This is not a complete resource profile. Inference requires 41 short FFTs plus a 492-by-3 linear map per window. A new CPU contract replay over 30 synthetic windows, including imports, took 1.738159192 seconds with 652,252 KiB process peak RSS. That is a compatibility measurement, not a hidden-cohort or A100 runtime estimate. The official 60-minute, one-A100 evaluation limit applies.

The original training kit was pinned to 2b53aec91c111f8b5423b6674f096c48cc1a99af with NeuralBench 0.4.0 and seed 20261031. The frozen Solver and current Objective/streaming contract were additionally exercised with real Benchopt 1.10.0 against official kit 7160f93ae5e86249e7ee5ce0583e34ec7163c521, on synthetic fixtures only. No training or study-data replay occurred during recovery. Full worker-image/CLI replay and sealed generalization are unverified.

Original source, training program, numerical logs and public warm-up provenance: https://github.com/Agnuxo1/neural-interfaces-2026-track02 . Original two-file ZIP SHA-256: d5a1f2bef886fcd775b5a41338612f016087f0c6a4b321dc3ef33e27987de0cd. This archive preserves submission.py and weights.npz byte-for-byte and adds only this method, the MIT licence and citation metadata.

## Rights and limitations

Original code and trained weights are MIT-licensed; see LICENSE. Dataset creators, the competition kit and scientific software retain their own rights and licences. This is a research BCI decoder, without clinical or medical-device validation. The warm-up holdout had only two participants; its score and the synthetic checks establish no final rank, selection or prize.
