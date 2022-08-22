
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DDIMSchedulerConf:
    _target_: str = "diffusers.schedulers.scheduling_ddim.DDIMScheduler"
    num_train_timesteps: Any = 1000
    beta_start: Any = 0.0001
    beta_end: Any = 0.02
    beta_schedule: Any = "linear"
    trained_betas: Any = None
    timestep_values: Any = None
    clip_sample: Any = True
    set_alpha_to_one: Any = True
    tensor_format: Any = "pt"
