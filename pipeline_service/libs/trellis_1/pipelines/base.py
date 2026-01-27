import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            if model is not None:
                model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        import os
        import json
        import sys
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
            print(f"[Trellis] Loading pipeline from local path: {path}", file=sys.stderr, flush=True)
        else:
            from huggingface_hub import hf_hub_download
            print(f"[Trellis] Downloading pipeline config from HuggingFace: {path}...", file=sys.stderr, flush=True)
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        model_keys = list(args['models'].keys())
        print(f"[Trellis] Loading {len(model_keys)} models: {', '.join(model_keys)}", file=sys.stderr, flush=True)
        _models = {}
        for i, (k, v) in enumerate(args['models'].items(), 1):
            print(f"[Trellis] Loading model {i}/{len(model_keys)}: {k} ({v})", file=sys.stderr, flush=True)
            _models[k] = models.from_pretrained(f"{path}/{v}")
        print(f"[Trellis] All models loaded successfully", file=sys.stderr, flush=True)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    def to(self, device: torch.device) -> None:
        for m in self.models.values():
            if m is not None:
                m.to(device) 

    @property
    def device(self) -> torch.device:
        for m in self.models.values():
            if hasattr(m, 'device'):
                return m.device
        for m in self.models.values():
            if hasattr(m, 'parameters'):
                return next(m.parameters()).device
        raise RuntimeError("No device found.")

    

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
