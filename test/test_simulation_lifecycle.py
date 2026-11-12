from unittest.mock import Mock, call

import pytest

from more_simulation import simulation as module
from rpp_py.lifecycle import CallbackResult, DeactivationResult, LifecycleError


@pytest.fixture
def context(monkeypatch):
    context = Mock()
    context.configure.return_value = CallbackResult()
    context.activate.return_value = CallbackResult()
    context.deactivate.return_value = DeactivationResult()
    context.cleanup.return_value = CallbackResult()
    context.get_component.return_value = [object()]
    monkeypatch.setattr(module, "DataManager", Mock())
    builder = Mock()
    builder.build_script_from_library.return_value = context
    monkeypatch.setattr(module, "ComponentContextBuilder", Mock(return_value=builder))
    return context


def test_configure_activate_and_close_in_order(context):
    simulation = module.Simulation(rpp_workspace="workspace")
    assert simulation.vessels == context.get_component.return_value
    simulation.close()
    simulation.close()

    assert context.mock_calls == [
        call.configure(), call.activate(), call.get_component("vessels"),
        call.deactivate(), call.cleanup(), call.close(),
    ]


@pytest.mark.parametrize("operation", ["configure", "activate"])
def test_startup_failure_releases_context(context, operation):
    getattr(context, operation).return_value = CallbackResult.failure("cannot start")

    with pytest.raises(LifecycleError, match="cannot start"):
        module.Simulation(rpp_workspace="workspace")

    if operation == "configure":
        context.activate.assert_not_called()
    context.get_component.assert_not_called()
    assert context.mock_calls[-3:] == [
        call.deactivate(), call.cleanup(), call.close()]


def test_component_lookup_failure_releases_context(context):
    context.get_component.side_effect = RuntimeError("missing vessels")

    with pytest.raises(RuntimeError, match="missing vessels"):
        module.Simulation(rpp_workspace="workspace")

    context.cleanup.assert_called_once()
    context.close.assert_called_once()


@pytest.mark.parametrize("operation", ["deactivate", "cleanup"])
def test_teardown_failure_still_releases_context(context, operation):
    simulation = module.Simulation(rpp_workspace="workspace")
    getattr(context, operation).side_effect = RuntimeError("cannot stop")

    with pytest.raises(RuntimeError, match="cannot stop"):
        simulation.close()

    context.cleanup.assert_called_once()
    context.close.assert_called_once()
