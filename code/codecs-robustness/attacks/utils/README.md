## Summary 
This subfolder contains util functions and variables.

---

### `loss_functions.py`

A mini-library of differentiable loss terms for image-compression and FTDA experiments.

* **Color-space utility** – `process_colorspace()` converts tensors between RGB and *YCbCr*, automatically handling the special JPEG-AI case where data live in `[0 … 255]`.
* **Loss collection**

  * Noise-matching: `added_noises_loss()` / `added_noises_loss_Y()`
  * Reconstruction MSE: `reconstr_loss()` / `reconstr_loss_Y()` and `src_reconstr_loss_Y()`
  * FTDA baselines: `ftda_default_loss()` / `ftda_default_loss_Y()`
  * Perceptual variants: `ftda_msssim_loss()` and `reconstruction_msssim_loss()` (multiscale SSIM)
  * Rate term: `bpp_increase_loss()`
  * **Experimental** focus loss: `pointwise_added_noises_loss()` applies a Gaussian-blur mask around a chosen pixel.
* **Registry** – `loss_name_2_func` lets training scripts pick a loss by string key.

> *Tip*: all MSE-style functions return **negative values** so that *maximising* the objective increases quality—mention this quirk elsewhere in the docs to avoid confusion.

---

### `codec_metrics.py`

Helpers for aggregating quality-metric results stored in a `pandas.DataFrame`.

* **Delta scores** – quantify attack impact (with defence applied)

  ```text
  FR:  fr_delta_score()      # clear-vs-attacked reconstruction
  NR:  nr_delta_score()
  ```
* **Defence effectiveness** – extra deltas comparing *with* vs *without* defensive preprocessing

  ```text
  FR:  fr_defence_delta_score()
  NR:  nr_defence_delta_score()
  ```
* **Baseline checks**

  * `mean_fr_clear_attacked()` – mean FR metric between *clear* and *attacked* images.
  * `delta_nr_clear_attacked()` – NR difference on originals (no reconstruction).
* **Convenience wrapper** – `calc_scores_codec()` loops over all registered metrics, prints each statistic, and returns a table (`DataFrame`) ready for logging or LaTeX export.

  * Built-in dictionaries `fr_2_lower_better` and `nr_2_lower_better` tell each scorer whether “lower-is-better,” automatically flipping signs where needed.

---

### `color_transforms_255.py`

Utility wrappers around **PyTorch** tensors for fast, differentiable colour-space conversion in the *0 … 255* range.

| Function             | Purpose                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `_rgb_to_y()`        | Core helper that computes the luminance channel **Y** from R G B tensors.                                |
| `rgb_to_ycbcr_255()` | Convert an RGB image to full-range **YCbCr**. Returns a 3-channel tensor in the same shape as the input. |
| `rgb_to_y_255()`     | Extract *only* the Y (luma) channel from RGB. Handy for Y-only metrics.                                  |
| `ycbcr_to_rgb_255()` | Inverse transform back to RGB; clamps output to `[0, 255]`.                                              |

---

### `defence_scoring_methods.py`

A collection of helpers that **quantify how well a defence recovers image-quality metrics after an attack**.
All routines operate on a `pandas.DataFrame` whose columns follow the naming pattern used in earlier scripts (`clear`, `attacked`, `defended-clear`, `defended-attacked`, plus SSIM/PSNR columns).

| Group                                | Function(s)                                           | What it measures                                                                         |
| ------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Relative / absolute quality loss** | `robust_rel_gain`, `robust_abs_gain`                  | Δ between pristine images and **defended-attacked** outputs.                             |
|                                      | `both_defended_rel_gain`, `both_defended_abs_gain`    | Δ when *both* clear and attacked images were run through the defence.                    |
|                                      | `nonpurified_rel_gain`, `nonpurified_abs_gain`        | Baseline Δ for **unprotected** attacks (no defence).                                     |
| **Per-pair similarity**              | `defence_similarity_score`                            | Combined SSIM + PSNR of defended-attacked vs. clear images.                              |
|                                      | `defence_clear_similarity_score`                      | Same but on defended-clear vs. clear (should be *high* if defence is “non-destructive”). |
| **Rank-correlation checks (SROCC)**  | `robust_attacked_srocc_mos`, `robust_clear_srocc_mos` | Correlate metric values *after defence* with human MOS collected on the clear originals. |
|                                      | `clear_srocc_mos`, `attacked_srocc_mos`               | Correlate *raw* metric scores (no defence) with MOS.                                     |
|                                      | `robust_clear_srocc_clear`                            | Correlate metric before/after defence on clear images (checks monotonicity).             |


* **`calc_scores_defence(df, metric_range=1)`** iterates over this registry, prints each statistic for quick CLI inspection, and returns a tidy `DataFrame` (`score`, `value`) ready for logging, CSV export, or LaTeX tables.

---

### `evaluate.py`

A grab-bag of helpers for **video I/O, metric prediction, and quick JPEG evaluation**.

| Component                                                                     | What it does                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`predict(img1, img2=None, model, device='cpu')`**                           | Normalises NumPy/`uint8` or RGB tensors to **N×C×H×W float32**, ships them to `device`, and runs the given *PyTorch* model. <br/>• FR metrics pass both images (`img1`, `img2`); NR metrics pass only one.                                                               |
| **`Encoder`**                                                                 | Thin wrapper around **PyAV** for one-shot RGB → video encoding |
| **`eval_encoded_video(model, encoded_path, orig_path, is_fr, batch_size=1)`** | Streams two videos in sync (via `iter_images`/`get_batch`), feeds them to `model`, and **yields** per-batch metrics:`(metric, idx_start, idx_end, psnr, ssim, mse, linf, mae)`                                                                                      |
| **`compress(img, q)`**                                                        | Fast JPEG re-encoding with OpenCV (`q` = 0–100). Returns either NumPy `float32` or Torch tensor in `[0,1]`.                                                                                                                                                              |
| **`jpeg_generator(img_iter, quality_list)`**                                  | Generator that yields *(original, jpeg)* pairs for each quality factor supplied.                                                                                                                                                                                         |
| **`write_log(path, dataset_name, mean_time_ms, preset=-1)`**                  | Appends one line to a CSV log (`test_dataset, mean_time_ms, attack_preset`). Creates the header automatically.                                                                                                                                                           |
| **`create_tensor(video_iter, device)`**                                       | Yields per-frame 4-D tensors (`1×3×H×W`) ready for batched inference.                                                                                                                                                                                                    |

---

### `fgsm_evaluate.py`

This is the **master script** that drives an end-to-end benchmark of a *learned image/video codec* under adversarial attacks and optional defences.

1. **Metric helpers**

   * `calc_frs()` – batch-compute full-reference metrics (MSE/MAE/PSNR/SSIM/L∞/MS-SSIM/VMAF).
   * `calc_nrs()` – query NR-IQA models such as *NIQE*.

2. **Attack handling**

   * `apply_attack()` switches the codec to `train()` for gradient access, launches the callback with user-defined hyper-parameters (loaded from JSON/CSV), and returns the perturbed tensor plus wall-time.

3. **Codec evaluation**

   * `create_row()` runs **defended** and **undefended** versions of the codec on *clean* and *attacked* images, then assembles one result row with:

     * bit-per-pixel stats (raw + “real”);
     * FR / NR metrics for every pairwise comparison needed;
     * reference JPEG2000 baselines at equal *PSNR* and equal *BPP*;
     * optional PNG dumps for qualitative inspection.

4. **Dataset loop** – `run()` iterates over a folder of images (single-frame for now), crops/resizes to multiples of 64, launches the attack, calls `create_row()`, and streams rows into a Pandas `DataFrame`.

5. **CLI entry point** – `test_main()` parses arguments (device, codec name, datasets, attack/defence presets, logging paths…), wires up models via `importlib`, and executes the full pipeline.

   * Results:

     * per-dataset raw CSV (`<codec>_<dataset>_test.csv`)
     * per-dataset score log (`<dataset>_log.csv`)
     * aggregate “total” log + raw CSV
     * *optional* parallel run for a **main** (reference) codec when evaluating JPEG-AI.

#### Notable implementation details / comments

* **JPEG-AI branch** – the script handles 0–255 *YCbCr* tensors and converts them back to 0–1 RGB for metric calculation (`is_jpegai` flag).
* **Config-driven** – codec hyper-parameters come from `src/config.json`; attack/defence presets from JSON or CSV make sweeping experiments reproducible.
* **Timing stats** – average inference time (`mean_time`) and attack time are appended to the score tables for quick throughput checks.

---

### `metrics.py`

Tiny wrapper layer that standardises all **quality-metric calls** used in the pipeline.

| Function / Class           | Role                                                                                  |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `PSNR`               | Peak-signal-to-noise ratio via **skimage**.                                           |
| `SSIM`               | Mean structural similarity (multi-frame friendly).                                    |
| `MSE`, `MAE`, `L_inf_dist` | Classic pixel-wise errors.                                                            | 
| `MSSSIM`             | Multiscale SSIM using **pytorch-msssim** (GPU-friendly).                              |
| `vmaf`          | **VMAF** via FFmpeg/`libvmaf`. Saves temp PNGs, runs subprocess, parses the JSON log. |
| `niqe`             | Thin `torch.nn.Module` that wraps **piq**’s NIQE (`lower_better=True`).               |

> **Tip**: all helpers return **Python scalars** (except `vmaf`, which returns a 0-D tensor) so they can drop straight into NumPy/Pandas without type juggling.

---

### `read_dataset.py`

Utilities for loading still images **and** video frames into NumPy / PyTorch, plus a lightweight `Dataset` wrapper.

| Item                                     | Purpose                                                                                                                                                                                                                                                    | 
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `to_torch(x, device='cpu')`              | Convert an `H×W×3` or `N×H×W×3` NumPy array → **`N×3×H×W` float32 tensor**.                                                                                                                                                                                |       |
| `to_numpy(x)`                            | Inverse transform: 4-D Torch tensor → NumPy in NHWC layout.                                                                                                                                                                                                |       |
| `get_batch(video_iter, batch_size)`      | Pulls the next `batch_size` elements from a generator returned by `iter_images()` and *keeps frames from the same video together*. Returns:<br/>`images, video_name, first_fn, first_path, updated_iter, is_video`.                                        |       |
| `iter_images(path)`                      | Yield `(image, frame_id, video_name, abs_path, is_video)`. Uses **PyAV** for decoding video.                                                                                       |       |
| `center_crop(img)`                       | 256×256 crop around the centre (resizes smaller images first).                                                                                                                                                                                             |       |
| `MyCustomDataset(path_gt, device='cpu')` | Simple `torch.utils.data.Dataset` that:<br/>• collects all files in `path_gt`,<br/>• shuffles every epoch,<br/>• returns a centre-cropped, normalised tensor in `[0,1]` on the requested device.<br/>Includes a `next_data()` helper for manual iteration. |       |

---

### `traditional_reference_codec.py`

Quick wrappers around **Glymur** to benchmark *JPEG 2000* against your learned codecs.

| Function                                                  | Job                                                                                                                                 |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `jpeg2k_compress(src, dump_path, target_quality, device)` | Encodes each RGB frame at a **given PSNR** (one target per image). Returns the decoded tensor ∈ \[0 … 1] and a list of actual BPPs. |
| `jpeg2k_compress_fix_bpp(src, dump_path, bpp, device)`    | Encodes at a **fixed bit-per-pixel**. Uses `cratios = 24 / bpp`. Same outputs as above.                                             |
