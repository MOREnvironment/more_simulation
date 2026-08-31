from more_simulation.plotting import plot_results
from more_simulation.simulation import Simulation


def main():
    simulation = Simulation()
    simulation.run()
    plot_results(simulation.results)


if __name__ == "__main__":
    main()
