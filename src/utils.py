"""Utility functions: logging, random seed, RNG state, and torch.load wrapper."""

import random
from datetime import datetime

import numpy as np
import torch


def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def set_seed(seed=42):
    log(f"Setting random seed: seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda_random_state_all"] = None
    return state


def set_rng_state(state):
    if state is None:
        return
    try:
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "numpy_random_state" in state:
            np.random.set_state(state["numpy_random_state"])
        if "torch_random_state" in state:
            torch.set_rng_state(state["torch_random_state"])
        cuda_rng_state = state.get("torch_cuda_random_state_all")
        if torch.cuda.is_available() and cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
    except Exception as exc:
        log(f"[Warning] Failed to restore RNG state, continuing. Reason: {exc}")


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)