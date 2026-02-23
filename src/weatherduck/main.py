import argparse

from loguru import logger

from .configs import autoregressive_experiment_factory, singlestep_experiment_factory


def main() -> None:
    """Run a single-step or autoregressive experiment.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Run a WeatherDuck experiment.")
    parser.add_argument(
        "--mode",
        choices=("single", "autoregressive"),
        default="autoregressive",
        help="Select single-step or autoregressive experiment.",
    )
    args = parser.parse_args()

    exp = (
        singlestep_experiment_factory()
        if args.mode == "single"
        else autoregressive_experiment_factory()
    )
    with logger.catch(reraise=True):
        exp.run()
