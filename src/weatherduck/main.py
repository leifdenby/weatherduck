from .weatherduck import experiment_factory

def main():
    exp = experiment_factory()
    exp.run()