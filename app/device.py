"""Pick where local models run. Same code on a GPU laptop and a CPU-only server."""

import logging
from functools import cache

from app.config import get_settings

log = logging.getLogger(__name__)


@cache
def get_device() -> str:
    wanted = get_settings().device.lower()
    if wanted != "auto":
        return wanted
    try:
        import torch

        if torch.cuda.is_available():
            log.info("using CUDA: %s", torch.cuda.get_device_name(0))
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def model_kwargs() -> dict:
    """fp16 on GPU halves memory and roughly doubles throughput; CPU stays fp32."""
    if get_device().startswith("cuda"):
        import torch

        return {"torch_dtype": torch.float16}
    return {}
