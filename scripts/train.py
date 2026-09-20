"""Trains one fold of the 2.5D study classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch

from knee.dataset import build_data
from knee.model import Model2p5D
from knee.train import TrainConfig, fit, pick_device, seed_all, write_history, write_oof


def parse_args(argv=None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list, or None for sys.argv.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, default=None, help="validation fold")
    parser.add_argument("--out-dir", default=None, help="where the checkpoint and OOF go")
    parser.add_argument("--overrides", default="{}",
                        help="JSON dict of config fields, same keys as TRAIN_OVERRIDES")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Builds the run config from the defaults, TRAIN_OVERRIDES and the command line.

    Args:
        args: The parsed arguments.

    Returns:
        The config, with the command line winning over the environment.
    """
    overrides = json.loads(args.overrides)
    if args.fold is not None:
        overrides["FOLD"] = args.fold
    if args.out_dir is not None:
        overrides["OUT_DIR"] = args.out_dir
    return TrainConfig.from_env(overrides)


def main() -> None:
    """Runs one fold end to end and writes the checkpoint, OOF and history."""
    cfg = build_config(parse_args())
    seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    device = pick_device()
    print("torch", torch.__version__, "| timm", timm.__version__, "| device", device)

    out_dir = Path(cfg.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, valid_dl = build_data(cfg)
    model = Model2p5D(cfg.as_dict()).to(device)
    print(cfg.BACKBONE, "features:", model.backbone.num_features,
          "| params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6),
          "| trainable: %.1fM" % (sum(p.numel() for p in model.parameters()
                                      if p.requires_grad) / 1e6))

    result = fit(cfg, model, train_dl, valid_dl, out_dir)
    write_oof(cfg, result, valid_dl.dataset.df, out_dir)
    write_history(cfg, result, out_dir)


if __name__ == "__main__":
    main()
