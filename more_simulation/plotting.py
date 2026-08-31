import matplotlib.pyplot as plt


def plot_results(results, show: bool = True):
    """Plot state and output histories for every simulated vessel."""
    figures = []
    for vessel_index, result in enumerate(results):
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        time = result["time"]

        for state_index in range(result["states"].shape[1]):
            axes[0].plot(
                time,
                result["states"][:, state_index],
                label=f"State {state_index}",
            )
        axes[0].set_ylabel("State")
        axes[0].grid(True)
        if result["states"].shape[1]:
            axes[0].legend()

        for output_index in range(result["outputs"].shape[1]):
            axes[1].plot(
                time,
                result["outputs"][:, output_index],
                label=f"Output {output_index}",
            )
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Output")
        axes[1].grid(True)
        if result["outputs"].shape[1]:
            axes[1].legend()

        figure.suptitle(f"Vessel {vessel_index + 1}")
        figure.tight_layout()
        figures.append(figure)

    if show:
        plt.show()

    return figures
