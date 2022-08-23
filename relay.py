from __future__ import annotations
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

from diffusers import schedulers  # type: ignore
from diffusers.pipelines import StableDiffusionPipeline  # type: ignore
from loguru import logger
from omegaconf import MISSING
from ranzen import implements
from ranzen.hydra import Option, Relay
import torch
from torch.amp.autocast_mode import autocast
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
    prompt: Union[str, Tuple[str]] = MISSING
    model: StableDiffusionModel = StableDiffusionModel.V1_4
    device: Union[int, Literal["cpu"]] = 0
    cache_dir: str = ".model_cache"
    dtype: Dtype = Dtype.F16

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
        logger.info(f"Run config:\n'{_clean_up_dict(raw_config)}'")
        wandb.config.update(raw_config)
        logger.info(
            f"Loading pretrained model '{self.model.value}' using cache directory "
            f"'{Path(self.cache_dir).resolve()}'; will attempt to download the model if no "
            "existing download is found."
        )
        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(  # type: ignore
            self.model.value,
            scheduler=self.scheduler,
            torch_dtype=self.dtype.value,
            use_auth_token=True,
            cache_dir=self.cache_dir,
        )
        device = torch.device(self.device)
        pipe.to(device)
        logger.info(f"Using device '{device}'.")

        prompt_str = self.prompt if isinstance(self.prompt, str) else "\n".join(self.prompt)
        logger.info(f"Beginning text-to-image sampling with text prompt(s):\n{prompt_str}")
        with autocast("cuda"):
            output = pipe(
                prompt=list(self.prompt),
                **asdict(self.sampler),
                output_type="pil",  # type: ignore
            )
            images = output["sample"][0]  # type: ignore
            if not isinstance(images, list):
                images = [images]
            images_wandb = [
                wandb.Image(image, caption=prompt) for image, prompt in zip(images, self.prompt)
            ]
        logger.info(f"Finished sampling; logging images to wandb.")
        wandb.log({"generated_images": images_wandb})
