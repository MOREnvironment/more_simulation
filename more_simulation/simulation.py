import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import casadi as ca
import numpy as np
from more_common.casadi_graph import RppCasadiGraph
from rpp_plugin_types.more_dynamics import VehicleModel3D
from rpp_py.context_builder import ComponentContextBuilder
from rpp_py.data_manager import DataManager


class Simulation:
    COMPONENTS = {
        "vessels": "List[more_dynamics::VehicleModel3D]",
    }

    def __init__(
        self,
        delta_t: float = 0.1,
        duration: float = 35.0,
        script_path: Path | None = None,
    ):
        if delta_t <= 0.0:
            raise ValueError("delta_t must be greater than zero")
        if duration < 0.0:
            raise ValueError("duration cannot be negative")

        self.delta_t = delta_t
        self.duration = duration
        self.results = []

        data_manager = DataManager()
        data_manager.load_script_description = self._load_script_description
        self.context_builder = ComponentContextBuilder(
            data_manager=data_manager
        )
        script_path = (script_path or Path(__file__)).resolve()
        if self._has_source_workspace(script_path):
            self.rpp_context = self.context_builder.build_from_script(
                str(script_path)
            )
        else:
            description = self._installed_script_description()
            self.rpp_context = (
                self.context_builder.build_from_script_description(
                    str(description)
                )
            )
        self.rpp_context.initialize()
        self.vessels: List[VehicleModel3D] = self.rpp_context.get_component(
            "vessels"
        )

    @staticmethod
    def _load_script_description(script_path: str):
        """Load legacy and named-configuration script descriptions."""
        with Path(script_path).open(encoding="utf-8") as description_file:
            description = json.load(description_file)

        components = description.get("Components")
        configurations = description.get("Configurations")
        if components is None and configurations:
            active = description.get("ActiveConfiguration")
            if active not in configurations:
                active = next(iter(configurations))
            components = configurations[active].get("Components", {})

        return SimpleNamespace(
            components=components or {},
            spec=description.get("Spec", {}),
        )

    def run(self):
        """Run every configured vessel and retain each simulation result."""
        num_steps = int(self.duration / self.delta_t)
        self.results = []

        for vessel_index, vessel in enumerate(self.vessels):
            graph = RppCasadiGraph(vessel.graph())
            inputs = ca.DM.zeros(num_steps + 1, graph.num_inputs)
            self.results.append(
                self._simulate(graph, inputs, num_steps, vessel_index)
            )

        return self.results

    def _simulate(
        self,
        graph: RppCasadiGraph,
        inputs: ca.DM,
        num_steps: int,
        vessel_index: int,
    ):
        initial_conditions = self._extract_initial_conditions(graph.payload)
        state_symbol = graph.step.sx_in(0)
        parameter_symbol = graph.step.sx_in(1)
        ode = {
            "x": state_symbol,
            "p": parameter_symbol,
            "ode": graph.step(state_symbol, parameter_symbol),
        }
        integrator = ca.integrator(
            f"vessel_sim_{vessel_index}",
            "cvodes",
            ode,
            0,
            self.delta_t,
            {"abstol": 1e-8, "reltol": 1e-6},
        )

        state = ca.DM(initial_conditions)
        time = np.arange(num_steps + 1, dtype=float) * self.delta_t
        states = np.zeros((num_steps + 1, initial_conditions.shape[0]))
        outputs = np.zeros((num_steps + 1, graph.num_outputs))
        states[0, :] = state.full().flatten()
        outputs[0, :] = graph.output(state, inputs[0, :]).full().flatten()

        for step in range(num_steps):
            control_input = inputs[step, :]
            integration_result = integrator(x0=state, p=control_input)
            state = integration_result["xf"].full().flatten()
            states[step + 1, :] = state
            outputs[step + 1, :] = graph.output(
                state, control_input
            ).full().flatten()

        return {
            "time": time,
            "states": states,
            "outputs": outputs,
            "inputs": np.asarray(inputs),
        }

    @staticmethod
    def _extract_initial_conditions(graph: VehicleModel3D.CasadyPayload):
        initial_conditions = []
        for state_description in graph.stateDescription:
            if state_description.ic:
                initial_conditions.extend(state_description.ic)
            else:
                initial_conditions.extend([0.0] * state_description.size)
        return ca.DM(initial_conditions)

    @staticmethod
    def _installed_script_description() -> Path:
        from ament_index_python.packages import get_package_share_directory

        package_share = Path(get_package_share_directory("more_simulation"))
        return (
            package_share
            / ".rppws"
            / "script_descriptions"
            / "simulation.json"
        )

    @staticmethod
    def _has_source_workspace(script_path: Path) -> bool:
        return any(
            (parent / ".rppws").is_dir() for parent in script_path.parents
        )
