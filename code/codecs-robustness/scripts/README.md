# Attack & Codec CI Framework

## Overview

This repository contains **GitLab CI helper scripts and configuration snippets** for building, training and evaluating:

* **Adversarial attack containers** (for image/video quality metrics)
* **Codec/metric containers** (e.g. VMAF, custom IQA/VQA models)
* **GitLab pipelines** that orchestrate multi-stage *build → train → test* jobs

The scripts are written in Bash and rely on Docker-in-Docker runners with NVIDIA GPUs. They expect the usual CI variables (`CI_PROJECT_DIR`, `CI_JOB_NAME`, `GML_SHARED`, `LAUNCH_ID`, etc.) to be provided by GitLab.


## Folder Conventions

| Path                | What it contains                                                                 |
| ------------------- | -------------------------------------------------------------------------------- |
| `attacks/<method>/` | Source code, `Dockerfile`, `run.py`, `train.py`, etc. for each attack method.    |
| `codecs/<metric>/`  | Source of the metric or codec implementation, usually with its own `Dockerfile`. |
| `subjects/`         | External libraries pulled as sub‑modules (e.g., **IQA‑PyTorch**).                |
| `scripts/`          | All helper Bash scripts (this folder).                                           |


## Key files

| Path                                  | Purpose & highlights                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/attack-build.sh`             | Builds an **attack Docker image**. Dynamically injects optional defences (DiffPure, DISCO, CADV) and extra Python deps, then pushes to the `$NEW_CI_REGISTRY`.      |
| `scripts/attack-generate-pipeline.sh` | Emits per-attack **`pipeline-<method>.yml` snippets** with *build / train / test* stages, tuned batch-sizes and dependency graph.                                   |
| `scripts/attack-init.sh`              | Common helpers: parses `$CI_JOB_NAME`, decides whether a method is *trainable*, *multimetric*, etc., and sets the final image tag.                                  |
| `scripts/attack-test.sh`              | Runs an attack inside the built image. Handles four cases: pure inference, trainable UAPs, multimetric ensembles, and video metrics; also dumps attacked datasets.  |
| `scripts/codec-build.sh`              | Builds a **metric/codec Docker image**. Optionally fetches the IQA-PyTorch submodule, then pushes the image.                                                        |
| `scripts/codec-init.sh`               | Minimal helper that logs in to the registry and exports `$IMAGE` for codec jobs.                                                                                    |
| `scripts/codec-test.sh`               | Smoke-tests a codec image by piping a reference & distorted video through the containerised metric (handles no-ref and full-ref cases).                             |
| `scripts/codecs.txt`                  | Line-delimited list of codec/metric names that the pipeline iterates over.                                                                                          |
| `scripts/runner.yml`                  | Shared GitLab CI template (extends `.common`, defines tags & caching). *Content not included in this export.*                                                       |


## Typical CI flow

1. **Build stage**
   *GitLab job* → `attack-build.sh` or `codec-build.sh` → Push image to registry.

2. **Train stage** (optional)
   For trainable attacks (e.g. UAPs), `attack-test.sh` is invoked with `PARAM_TRAIN=1`.

3. **Test stage**
   `attack-test.sh` (attacks) or `codec-test.sh` (metrics) run inside the freshly built image, saving logs, CSVs and dumps as job artifacts.

Generated artifacts (e.g. `.csv`, `.log`, `dumps.zip`) are then available for downstream evaluation or manual download.


## Adding a new attack or metric

1. **Create a directory** under `attacks/<method>` or `codecs/<metric>` with a `Dockerfile`, `run.py`, etc.
2. **Register the name** in `scripts/methods.txt` or `scripts/codecs.txt`.
3. If the method is trainable, append it to `scripts/trainable_methods.txt`.
4. Commit & push—the pipeline generator will include it automatically.

---
