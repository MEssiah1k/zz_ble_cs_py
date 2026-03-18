"""CLI entrypoint for Bluetooth Channel Sounding simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments import exp_v0, exp_v1, exp_v2, exp_v3, exp_v4, exp_v5, exp_v6


EXPERIMENTS = {
    "v0": exp_v0.run,
    "v1": exp_v1.run,
    "v2": exp_v2.run,
    "v3": exp_v3.run,
    "v4": exp_v4.run,
    "v5": exp_v5.run,
    "v6": exp_v6.run,
}


def load_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bluetooth Channel Sounding simulation runner")
    parser.add_argument("--version", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else Path("configs") / f"{args.version}.json"
    config = load_config(config_path)
    if args.seed is not None:
        config["random_seed"] = args.seed
    config["version_name"] = args.version
    summary = EXPERIMENTS[args.version](config, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
