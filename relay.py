from __future__ import annotations
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from diffusers import schedulers  # type: ignore
from diffusers.pipelines import StableDiffusionPipeline  # type: ignore
from loguru import logger
from omegaconf import MISSING
from ranzen import implements
from ranzen.hydra import Option, Relay
import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm  # type: ignore
import wandb

__all__ = ["StableDiffusionRelay"]


class StableDiffusionModel(Enum):
    V1_1 = "CompVis/stable-diffusion-v1-1"
    V1_2 = "CompVis/stable-diffusion-v1-2"
    V1_3 = "CompVis/stable-diffusion-v1-3"
    V1_4 = "CompVis/stable-diffusion-v1-4"


@dataclass
class WandbConf:
    _target_: str = "wandb.init"
    name: Optional[str] = None
    mode: str = "online"
    id: Optional[str] = None
    anonymous: Optional[bool] = None
    project: Optional[str] = "stable-diffusion"
    group: Optional[str] = None
    entity: Optional[str] = "predictive-analytics-lab"
    tags: Optional[List[str]] = None
    reinit: bool = False
    job_type: Optional[str] = None
    resume: Optional[str] = None
    dir: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class SamplerConf:
    _target_: str = "relay.SamplerConf"
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0


def _clean_up_dict(obj: Any) -> Any:
    """Convert enums to strings and filter out _target_."""
    if isinstance(obj, MutableMapping):
        return {key: _clean_up_dict(value) for key, value in obj.items() if key != "_target_"}
    elif isinstance(obj, Enum):
        return str(f"{obj.name}")
    return obj


class Dtype(Enum):
    F16 = torch.float16
    F32 = torch.float32
    BF16 = torch.bfloat16


@dataclass
class StableDiffusionRelay(Relay):
    wandb: Union[
        wandb.sdk.wandb_run.Run,  # type: ignore
        wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
        None,
    ] = MISSING
    sampler: SamplerConf = MISSING
    scheduler: Union[
        schedulers.DDIMScheduler,
        schedulers.LMSDiscreteScheduler,
        schedulers.PNDMScheduler,
    ] = MISSING
    prompt: Union[str, List[str]] = MISSING
    model: StableDiffusionModel = StableDiffusionModel.V1_4
    device: Union[int, Literal["cpu"]] = 0
    cache_dir: str = ".model_cache"
    dtype: Dtype = Dtype.F16
    seed: Optional[int] = None
    repeats: int = 1
    batch_size: int = 1
    revision: Optional[str] = None
    local_files_only: bool = True
    use_auth_token: bool = False

    @classmethod
    @implements(Relay)
    def with_hydra(
        cls,
        root: Path | str,
        *,
        scheduler: list[Option],
        clear_cache: bool = False,
        instantiate_recursively: bool = True,
    ) -> None:
        configs = dict(scheduler=scheduler, wandb=[WandbConf], sampler=[SamplerConf])
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: dict[str, Any]) -> None:
        logger.info(f"Run config: {_clean_up_dict(raw_config)}")
        wandb.config.update(raw_config)
        logger.info(
            f"Loading pretrained model '{self.model.value}' with cache directory "
            f"'{Path(self.cache_dir).resolve()}'; will attempt to download the model if no "
            "existing download is found (this requires a Hugging Face access token)."
        )
        device = torch.device(self.device)
        logger.info(f"Running on device '{device}'.")
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(  # type: ignore
            self.model.value,
            scheduler=self.scheduler,
            torch_dtype=self.dtype.value,
            use_auth_token=self.use_auth_token,
            cache_dir=self.cache_dir,
            revision=self.revision,
            local_files_only=self.local_files_only,
        ).to(device)

        generator = (
            None if self.seed is None else torch.Generator(device=device).manual_seed(self.seed)
        )
        prompt_ls = [self.prompt] if isinstance(self.prompt, str) else self.prompt
        prompt_str = "\n- ".join(prompt_ls)
        if self.repeats > 1:
            # Tile the prompts by ``repeats`` by interleaving, such that all repeats of a single
            # prompt are contiguous.
            prompt_ls = list(chain.from_iterable(zip(*(prompt_ls for _ in range(self.repeats)))))
        # Group the prompts into batches.
        batches = [
            prompt_ls[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range((len(prompt_ls) + self.batch_size - 1) // self.batch_size)
        ]
        with tqdm(total=len(batches), desc="Sampling batches", leave=True) as pbar:
            with autocast("cuda"):
                for batch in batches:
                    output = pipe(
                        prompt=batch,
                        **asdict(self.sampler),
                        generator=generator,
                        output_type="pil",  # type: ignore
                    )
                    images = output["sample"]
                    images_wandb = [
                        wandb.Image(image, caption=prompt) for image, prompt in zip(images, batch)
                    ]
                    wandb.log({"samples": images_wandb})
                    pbar.update()
