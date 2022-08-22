from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import torch
import wandb
from diffusers import schedulers  # type: ignore
from diffusers.pipelines import StableDiffusionPipeline  # type: ignore
from loguru import logger
from torch.amp.autocast_mode import autocast

from ranzen import implements
from ranzen.hydra import Option, Relay

__all__ = ["StableDiffusionRelay"]


class SdCheckpoint(Enum):
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
class SamplingConf:
    _target_: str = "relay.SamplingConf"
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0


@dataclass
class StableDiffusionRelay(Relay):
    wandb: Union[
        wandb.sdk.wandb_run.Run,  # type: ignore
        wandb.sdk.lib.disabled.RunDisabled,  # type: ignore
        None,
    ]
    sampling: SamplingConf
    scheduler: Union[
        schedulers.DDIMScheduler,
        schedulers.LMSDiscreteScheduler,
        schedulers.PNDMScheduler,
    ]
    prompt: Tuple[str] = ("a photo of an astronaut riding a horse on mars",)
    model: SdCheckpoint = SdCheckpoint.V1_4
    device: Union[int, Literal["cpu"]] = 0

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
        configs = dict(scheduler=scheduler, wandb=[WandbConf], sampling=[SamplingConf])
        super().with_hydra(
            root=root,
            clear_cache=clear_cache,
            instantiate_recursively=instantiate_recursively,
            **configs,
        )

    @implements(Relay)
    def run(self, raw_config: dict[str, Any]) -> None:
        logger.info(f"Current working directory: '{os.getcwd()}'")
        wandb.config.update(raw_config)
        pipe = StableDiffusionPipeline.from_pretrained(  # type: ignore
            self.model.value,
            scheduler=self.scheduler,
            use_auth_token=True,
        )
        assert isinstance(pipe, StableDiffusionPipeline)

        pipe.to(torch.device(self.device))
        with autocast("cuda"):
            output = pipe(
                prompt=list(self.prompt),
                **asdict(self.sampling),
                output_type="pil",  # type: ignore
            )
            images = output["sample"][0]  # type: ignore
            images_wandb = [
                wandb.Image(image, caption=prompt)
                for image, prompt in zip(images, self.prompt)
            ]
        wandb.log({"generated_images": images_wandb})
