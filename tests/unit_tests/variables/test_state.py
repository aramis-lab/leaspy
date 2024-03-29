import pytest
import re
import torch
from leaspy.variables.state import State
from leaspy.variables.dag import VariablesDAG
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    IndividualLatentVariable,
    LinkedVariable,
    DataVariable,
    PopulationLatentVariable,
    NamedVariables,
)
from leaspy.variables.specs import LatentVariableInitType


def _assert_empty_state(state: State):
    """Assert provided state is empty."""
    assert len(state) == 0
    assert state.auto_fork_type is None
    assert len(state.tracked_variables) == 0
    assert state._values == {}  # noqa
    assert state._last_fork is None  # noqa


@pytest.fixture
def state() -> State:
    """Return a State instance for testing purposes."""
    return State(
        VariablesDAG.from_dict(
            NamedVariables(
                {
                    "mean": Hyperparameter(100.0),
                    "scale": Hyperparameter(0.1),
                    "x": PopulationLatentVariable(Normal("mean", "scale")),
                    "t": DataVariable(),
                    "model": LinkedVariable(lambda *, x, t: x * t),
                }
            )
        )
    )


def test_empty_state():
    """Test that a State with raw instantiation is empty."""
    state = State(VariablesDAG.from_dict({}))
    _assert_empty_state(state)
    state.track_variable("foo")
    _assert_empty_state(state)
    state.clear()
    _assert_empty_state(state)


def _assert_state_in_initial_state(state: State):
    assert len(state) == 8
    assert state.dag.sorted_variables_names == (
        "mean",
        "nll_regul_ind_sum_ind",
        "scale",
        "t",
        "x",
        "nll_regul_ind_sum",
        "model",
        "nll_regul_x",
    )
    assert state["mean"] == torch.tensor(100)
    assert state["scale"] == torch.tensor(0.1)
    assert state.auto_fork_type is None
    assert len(state.tracked_variables) == 0
    assert state._values == {  # noqa
        "mean": torch.tensor(100),
        "nll_regul_ind_sum_ind": None,
        "scale": torch.tensor(0.1000),
        "t": None,
        "x": None,
        "nll_regul_ind_sum": None,
        "model": None,
        "nll_regul_x": None,
    }
    assert state.dag.direct_ancestors == {
        "mean": frozenset(),
        "scale": frozenset(),
        "x": frozenset(),
        "nll_regul_x": frozenset({"mean", "scale", "x"}),
        "t": frozenset(),
        "model": frozenset({"t", "x"}),
        "nll_regul_ind_sum_ind": frozenset(),
        "nll_regul_ind_sum": frozenset({"nll_regul_ind_sum_ind"}),
    }
    assert state._last_fork is None  # noqa


def test_state_variable_tracking(state):
    """Test that tracking and un-tracking variables works as expected."""
    _assert_state_in_initial_state(state)
    state.track_variable("foo")
    assert len(state.tracked_variables) == 0
    state.track_variable("mean")
    assert state.tracked_variables == {"mean"}
    state.track_variable("mean")
    assert state.tracked_variables == {"mean"}
    state.track_variable("model")
    assert state.tracked_variables == {"mean", "model"}
    state.untrack_variable("foo")
    assert state.tracked_variables == {"mean", "model"}
    state.untrack_variable("model")
    assert state.tracked_variables == {"mean"}
    state.track_variables(("model", "x"))
    assert state.tracked_variables == {"x", "mean", "model"}
    state.untrack_variables(("x", "mean", "model"))
    assert len(state.tracked_variables) == 0


def test_state_precompute_all_error(state):
    """Test that an error is raised when trying to compute the state with an unset independent variable."""
    from leaspy.exceptions import LeaspyInputError

    with pytest.raises(
        LeaspyInputError,
        match=re.escape(
            "'t' is an independent variable which is required to proceed"
        ),
    ):
        state.precompute_all()


def test_state_cannot_put_independant_variable(state):
    """Test that an error is raised when trying to set a value for an independent variable."""
    from leaspy.exceptions import LeaspyInputError

    with pytest.raises(
        LeaspyInputError,
        match=re.escape("'scale' is not intended to be set"),
    ):
        state.put("scale", 2)


def test_state_precompute_all(state):
    """Test the precompute all method of the state."""
    state.put("t", torch.tensor(2))
    state.put("x", torch.tensor(1000))
    state.precompute_all()

    assert state["x"] == torch.tensor(1000)
    assert state["mean"] == torch.tensor(100)
    assert state["nll_regul_ind_sum_ind"] == torch.tensor(0)
    assert state["scale"] == torch.tensor(0.1)
    assert state["t"] == torch.tensor(2)
    assert state["nll_regul_ind_sum"] == torch.tensor(0)
    assert state["model"] == torch.tensor(2000)


@pytest.mark.parametrize("method", LatentVariableInitType)
def test_state_precompute_with_put_population_latent_variables(state, method):
    """Test the precompute all method of the state."""
    torch.manual_seed(42)
    state.put("t", torch.tensor(2))
    state.put_population_latent_variables(method)
    state.precompute_all()

    assert torch.allclose(
        state["x"],
        torch.tensor(100.0337 if method == LatentVariableInitType.PRIOR_SAMPLES else 100.0)
    )
    assert state["mean"] == torch.tensor(100)
    assert state["nll_regul_ind_sum_ind"] == torch.tensor(0)
    assert state["scale"] == torch.tensor(0.1)
    assert state["t"] == torch.tensor(2)
    assert state["nll_regul_ind_sum"] == torch.tensor(0)
    assert torch.allclose(
        state["model"],
        torch.tensor(200.0673 if method == LatentVariableInitType.PRIOR_SAMPLES else 200.0)
    )
