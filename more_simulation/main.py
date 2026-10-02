from argparse import ArgumentParser

from more_simulation.plotting import plot_results
from more_simulation.simulation import Simulation


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("rpp_workspace")
    parser.add_argument("--rpp-configuration", default=None)
    arguments = parser.parse_args()
    simulation = Simulation(
        rpp_workspace=arguments.rpp_workspace,
        rpp_configuration=arguments.rpp_configuration,
    )
    simulation.run()
    plot_results(simulation.results)


if __name__ == "__main__":
    main()
