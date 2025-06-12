## Overview

This directory contains one subfolder per neural image codec implementation. A folder is named after the codec (for example `cheng2020-attn-4` or `jpegai-v61-bop-b0002`). Each codec folder encapsulates:

- A **Dockerfile** that installs all codec-specific dependencies.
- A **src/** directory containing the `CodecModel` wrapper around the actual codec implementation.

---

## Folder Structure

```bash
.
├── codec_A/
│ ├── Dockerfile
│ └── src/
│ └── model.py
├── codec_B/
│ ├── Dockerfile
│ └── src/
│ └── model.py
.
.
.
└── codec_X/
│ ├── Dockerfile
│ └── src/
│ └── model.py
```

For each version of codec and target bitrate there are separate subfolder. Each subfolder has the following structure:
- **`Codec_name/`**  
  Top-level folder named after the codec (e.g. `cdc`, `bmshj2018`, etc.).

- **`Dockerfile`**  
  Defines a container image with:
  - System libraries and Python packages required by this specific codec.
  - Weights downloading and initialisation.

- **`src/model.py`**  
  - Contains a class `CodecModel` that wraps the low-level codec implementation.
  - Exposes a uniform interface with one essential funtion `forward()`, which encodes input image.
  - Allows passing codec-specific parameters at initialization.

---

## `CodecModel` Wrapper

Each codec may have its own hyperparameters (e.g. quality levels, network architecture flags, rate–distortion tradeoffs). To keep a consistent API:

```python
from src.codec_model import CodecModel

# Example usage
model = CodecModel(
    quality=5,        # codec-specific parameter
    use_entropy=True  # codec-specific flag
)
compressed = model(input_image)
```

Each codec exposes a `model.py` module documenting the `CodecModel` wrapper.  Most wrappers rely on the [CompressAI](https://github.com/InterDigitalInc/CompressAI) library but can also include original research code shipped with the codec.
