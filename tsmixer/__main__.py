"""CLI entry point for package."""
import argparse
import os

import fn2argparse
import tensorflow as tf

from tsmixer import forecast, train
from tsmixer.utils import config


def create_parser():
    """Create the argparser for training and forecasting."""
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="cmd")
    fn2argparse.convert(train.train, sub_parsers.add_parser("train"))
    forecast_parser = sub_parsers.add_parser("forecast")
    forecast_parser.add_argument("--seed", default=train.train.__kwdefaults__["seed"])
    forecast_parser.add_argument(
        "--out_dir", default=train.train.__kwdefaults__["out_dir"]
    )
    forecast_parser.add_argument(
        "--forecast_len",
        type=int,
        default=forecast.forecast.__kwdefaults__["forecast_len"],
    )
    forecast_parser.add_argument(
        "--num_forecasts",
        type=int,
        default=forecast.forecast.__kwdefaults__["num_forecasts"],
    )
    return parser


def main():
    """Parse the args and run the command."""
    args = create_parser().parse_args()
    cmd = args.cmd
    delattr(args, "cmd")
    tf.keras.utils.set_random_seed(args.seed)
    if cmd == "train":
        os.makedirs(args.out_dir, exist_ok=True)
        train.train(
            **vars(config.save(args, os.path.join(args.out_dir, "config.yaml")))
        )
    elif cmd == "forecast":
        forecast.forecast(
            **(config.load(os.path.join(args.out_dir, "config.yaml")) | vars(args))
        )


if __name__ == "__main__":
    main()
