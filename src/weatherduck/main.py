from .weatherduck import autoregressive_experiment_factory
from loguru import logger


def main():
    exp = autoregressive_experiment_factory()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        with logger.catch(reraise=True):
            exp.run()