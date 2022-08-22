
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PNDMSchedulerConf:
    _target_: str = "diffusers.schedulers.scheduling_pndm.PNDMScheduler"
    num_train_timesteps: Any = 1000
    beta_start: Any = 0.0001
    beta_end: Any = 0.02
    beta_schedule: Any = "linear"
    tensor_format: Any = "pt"
    skip_prk_steps: Any = False
