
from dataclasses import dataclass, field
from omegaconf import MISSING
from relay import SamplingConf
from relay import SdCheckpoint
from typing import Any
from typing import Tuple
from typing import Union


@dataclass
class StableDiffusionRelayConf:
    _target_: str = "relay.StableDiffusionRelay"
    wandb: Any = MISSING  # Optional[Union[Run, RunDisabled]]
    sampling: SamplingConf = MISSING
    scheduler: Any = MISSING  # Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler]
    prompt: Tuple[str] = ('a photo of an astronaut riding a horse on mars',)
    model: SdCheckpoint = SdCheckpoint.V1_4
    device: Union[int, str] = 0
