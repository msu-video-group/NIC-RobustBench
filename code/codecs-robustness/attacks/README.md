# Attack Methods

This directory collects adversarial attack implementations used in the codec robustness benchmark.  Each
subfolder corresponds to a single attack and contains a `run.py` file exposing an
`attack` function.  The function receives an image tensor produced by a codec together with the
codec model and returns a modified tensor that should lead to a drop in metric scores while
respecting a distortion constraint.

## Folder Structure

```bash
.
├── attack_A/
│ └── run.py
├── attack_B/
│ └── run.py
.
.
.
└── attack_X/
│ └── run.py
```
## Directory overview

- **ifgsm** – iterative FGSM attack with L_inf constraint.
- **pgd-ifgsm** – variant of IFGSM using random initialization (PGD style).
- **ftda** – Fast Threshold-Constrained Distortion Attack.
- **ftda-randn-init** – FTDA starting from a random perturbation.
- **ftda-linf** – FTDA variant constrained in L_inf norm.
- **madc** – gradient attack with orthogonal projection of the update direction.
- **madc-randn-init** – same as `madc` but initialized with random noise.
- **madc-linf** – L_inf version of the MADC attack.
- **madc-norm** – MADC variant that uses a normalized gradient.
- **ssah** – spatial–spectral attack relying on wavelet decomposition.
- **ssah-randn-init** – `ssah` with additional random initialization.
- **cadv** – colourization based attack driven by a pretrained colour network.
- **korhonen-et-al** – attack using spatial activity maps as proposed by Korhonen et al.
- **random-noise** – baseline that adds clipped Gaussian noise.
- **mad-mix** – experimental attack mixing MADC-like steps without orthogonalization and steps with a normalized gradient.
- **noattack** – helper script computing metric ranges without applying an attack.
- **utils/** – common utilities (loss functions, metrics, evaluation helpers).  Notable files:
  - `fgsm_evaluate.py` – runner executing attacks and gathering metrics;
  - `codec_losses.py` – set of loss functions for optimisation;
  - `codec_scoring_methods.py` – functions to compute benchmark scores;
  - `evaluate.py`, `read_dataset.py` – dataset helpers and codec wrappers.
- **ci.yml** – GitLab CI configuration describing pipelines for each attack.

All attack functions share the same basic signature:

```python
def attack(compress_image, model=None, device='cpu', is_jpegai=False,
           loss_func=None, loss_func_name='undefined', **params):
    ...
```

`compress_image` contains the codec output in `[0,1]` range (or `[0,255]` for JPEGAI models).
The function should return a tensor of the same shape clipped to the valid range.
