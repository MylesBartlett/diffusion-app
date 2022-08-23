from __future__ import annotations

from diffusers import schedulers  # type: ignore
from ranzen.hydra import Option

from relay import StableDiffusionRelay


def main() -> None:
    scheduler_ops = [
        Option(name="ddim", class_=schedulers.DDIMScheduler),
        Option(name="lms", class_=schedulers.LMSDiscreteScheduler),
        Option(name="pndms", class_=schedulers.PNDMScheduler),
    ]
    StableDiffusionRelay.with_hydra(
        root="conf",
        clear_cache=True,
        instantiate_recursively=True,
        scheduler=scheduler_ops,
    )


if __name__ == "__main__":
    main()
