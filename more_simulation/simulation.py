from pathlib import Path
from typing import List

import casadi as ca
import numpy as np
from rpp_py.data_manager import DataManager
DataManager() # triggers sourcing of rpp_plugin_types
from more_common.casadi_graph import RppCasadiGraph
from rpp_plugin_types.more_dynamics import VehicleModel3D
from rpp_py.context_builder import ComponentContextBuilder


class Simulation:
    COMPONENTS = {
        "vessels": "List[more_dynamics::VehicleModel3D]",
    }
    RPP_SCRIPT_LIBRARY = "more_simulation"
    RPP_SCRIPT_NAME = "simulation"

    def __init__(
        self,
        delta_t: float = 0.1,
        duration: float = 35.0,
        rpp_workspace: Path | str | None = None,
        rpp_configuration: str | None = None,
    ):
        if delta_t <= 0.0:
            raise ValueError("delta_t must be greater than zero")
        if duration < 0.0:
            raise ValueError("duration cannot be negative")
        if rpp_workspace is None or not str(rpp_workspace).strip():
            raise ValueError("rpp_workspace must be specified")

        self.delta_t = delta_t
        self.duration = duration
        self.results = []

        data_manager = DataManager(workspace_path=rpp_workspace)
        self.context_builder = ComponentContextBuilder(
            data_manager=data_manager
        )
        self.rpp_context = self.context_builder.build_script_from_library(
            self.RPP_SCRIPT_LIBRARY,
            self.RPP_SCRIPT_NAME,
            configuration=rpp_configuration,
        )
        self.rpp_context.initialize()
        self.vessels: List[VehicleModel3D] = self.rpp_context.get_component(
            "vessels"
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
