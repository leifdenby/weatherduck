from loguru import logger

from .configs import autoregressive_experiment_factory


def main():
    exp = autoregressive_experiment_factory()
    with logger.catch(reraise=True):
        exp.run()
