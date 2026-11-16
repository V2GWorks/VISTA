# 0001: Baseline anomaly model

- Status: Accepted
- Date: 2026-11-16
- Deciders: @Ray3R3, @samoteng

## Context

VISTA needs a first model to anchor the pipeline and the evaluation harness. Real inspection lines rarely have many labelled defect images; they have plenty of images of good parts. MVTec AD reflects this: training sets contain only defect-free images, and defects appear only at test time.

So the baseline must:

1. Train on normal images only (unsupervised / one-class).
2. Produce a pixel-level score map, not just an image label, so we can localise defects and evaluate segmentation (pixel AUROC, PRO).
3. Be reproducible on a laptop GPU or CPU in reasonable time.
4. Carry a licence compatible with an Apache-2.0 codebase.

## Options considered

| Option | Fits 1 | Fits 2 | Notes |
|---|---|---|---|
| **PatchCore** (Roth et al., CVPR 2022) | Yes | Yes | Memory bank of pretrained CNN patch features; no training loop, only feature extraction and coreset subsampling. Strong published results on MVTec AD. |
| PaDiM | Yes | Yes | Gaussian per patch position; simpler but generally weaker than PatchCore and assumes aligned images. |
| EfficientAD | Yes | Yes | Faster inference, but a student–teacher training loop adds tuning work we don't need for a baseline. |
| Supervised detector/segmenter (e.g. YOLO-seg) | No | Yes | Needs labelled defects for training, which contradicts the setting. Ultralytics models are AGPL-3.0. |

## Decision

Use **PatchCore via [anomalib](https://github.com/open-edge-platform/anomalib)** (Apache-2.0) with a WideResNet-50 backbone as the baseline, starting on the `bottle` and `carpet` categories (one object, one texture).

## Consequences

- No training loop to debug first: the evaluation harness can be built against a model that is known to work on this benchmark.
- Inference cost and memory grow with the memory bank. The coreset sampling ratio is the main knob and will be recorded in every experiment config.
- Scores are distances in feature space, not probabilities. `InspectionResult.image_score` requires `[0, 1]`, so we need a normalisation step fitted on held-out good images; the threshold is chosen there, not on the test set.
- Later models (EfficientAD, a supervised head once we have labels) are compared against this baseline using the same harness and data splits.
