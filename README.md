# SuperMimic

A [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/)
implementation for training world models and policies to mimic motion data, built on [mjlab](../mjlab_research).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install dependencies:

```bash
cd supermimic
uv venv --python 3.13
uv add --editable ../mjlab_research --config-settings editable_mode=compat
```

## Usage

Train a world model:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run python src/supermimic/scripts/train.py \
  --runner.run_name <description>
```
