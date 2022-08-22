
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LMSDiscreteSchedulerConf:
    _target_: str = "diffusers.schedulers.scheduling_lms_discrete.LMSDiscreteScheduler"
    num_train_timesteps: Any = 1000
    beta_start: Any = 0.0001
    beta_end: Any = 0.02
    beta_schedule: Any = "linear"
    trained_betas: Any = None
    timestep_values: Any = None
    tensor_format: Any = "pt"
