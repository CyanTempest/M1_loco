"""Mac training entrypoint; learning and checkpointing remain owned by UniLab."""

import argparse
import json
import platform

import torch

from .config import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", choices=("flat", "mixed"), default="flat")
    parser.add_argument("--profile", choices=("basic", "b2w-full"), default="basic")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--checkpoint")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process. Run in a native Mac terminal, or select --device cpu."
        )
    torch.set_num_threads(args.torch_threads)
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "learner_device": args.device,
                "physics_device": "cpu",
                "wheel_kp": 0,
                "wheel_kd": 1,
                "terrain": args.terrain,
                "profile": args.profile,
            }
        ),
        flush=True,
    )
    cfg = configuration(
        args.terrain,
        num_envs=args.num_envs,
        iterations=args.iterations,
        device=args.device,
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        profile=args.profile,
    )
    from unilab.scripts.train_rsl_rl import main as run

    run(cfg)


if __name__ == "__main__":
    main()
