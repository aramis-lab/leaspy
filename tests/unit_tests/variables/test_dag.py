import re
import pytest


def test_empty_dag():
    """Tests empty DAG creation."""
    from leaspy.variables.dag import VariablesDAG

    d = VariablesDAG.from_dict({})

    assert d.variables == {}
    assert d.direct_ancestors == {}
    assert d.direct_children == {}
    assert d.sorted_variables_names == ()
    assert d.sorted_children == {}
    assert d.sorted_ancestors == {}
    assert d.sorted_variables_by_type == {}
    assert d.individual_variable_names == ()
    a, b = d.compute_topological_order_and_path_matrix({}, {})
    assert a == ()
    assert len(b) == 0
    with pytest.raises(KeyError, match="foo"):
        _ = d["foo"]


def test_dag_error_with_single_variable():
    """Assert that an error is raised when trying to create a DAG with a single node."""
    from leaspy.variables.dag import VariablesDAG
    from leaspy.variables.specs import IndepVariable
    from leaspy.exceptions import LeaspyInputError

    with pytest.raises(
        LeaspyInputError,
        match=re.escape("There are some variables left alone: {'x'}"),
    ):
        VariablesDAG.from_dict({"x": IndepVariable()})


def test_basic_dag_with_two_nodes():
    """Tests with a DAG containing 2 nodes on 2 levels."""
    from leaspy.variables.dag import VariablesDAG
    from leaspy.variables.specs import IndepVariable, LinkedVariable

    d = VariablesDAG.from_dict(
        {
            "x": IndepVariable(),
            "y": LinkedVariable(lambda *, x: -x),
        }
    )

    assert len(d.variables) == 2
    assert d.direct_ancestors == {'x': frozenset(), 'y': frozenset({'x'})}
    assert d.direct_children == {'x': frozenset({'y'}), 'y': frozenset()}
    assert d.sorted_variables_names == ("x", "y")
    assert d.sorted_children == {'x': ('y',), 'y': ()}
    assert d.sorted_ancestors == {'x': (), 'y': ('x',)}
    assert len(d.sorted_variables_by_type) == 2
    assert set(d.sorted_variables_by_type.keys()) == {IndepVariable, LinkedVariable}
    assert "x" in d.sorted_variables_by_type[IndepVariable]
    assert "y" not in d.sorted_variables_by_type[IndepVariable]
    assert "x" not in d.sorted_variables_by_type[LinkedVariable]
    assert "y" in d.sorted_variables_by_type[LinkedVariable]
    assert d.individual_variable_names == ()
    a, b = d.compute_topological_order_and_path_matrix({}, {})
    assert a == ()
    assert len(b) == 0


def test_advanced_dag_with_seven_nodes():
    """Tests with a DAG containing 7 nodes on 3 levels."""
    from leaspy.variables.dag import VariablesDAG
    from leaspy.variables.specs import IndepVariable, LinkedVariable, IndividualLatentVariable
    from leaspy.variables.distributions import Normal

    d = VariablesDAG.from_dict(
        {
            "V": IndepVariable(),
            "X": IndividualLatentVariable(Normal),
            "Y": IndepVariable(),
            "Z": IndividualLatentVariable(Normal),
            "X_x_Y": LinkedVariable(lambda *, X, Y: X * Y),
            "Y_x_Z": LinkedVariable(lambda *, Y, Z: Y * Z),
            "model": LinkedVariable(lambda *, X_x_Y, Y_x_Z, V: X_x_Y + Y_x_Z + V),
        }
    )

    assert len(d.variables) == 7
    assert d.direct_ancestors == {
        "V": frozenset(),
        "X": frozenset(),
        "Y": frozenset(),
        "Z": frozenset(),
        "X_x_Y": frozenset({"X", "Y"}),
        "Y_x_Z": frozenset({"Y", "Z"}),
        "model": frozenset({"V", "X_x_Y", "Y_x_Z"}),
    }
    assert d.direct_children == {
        "Z": frozenset({"Y_x_Z"}),
        "model": frozenset(),
        "X_x_Y": frozenset({"model"}),
        "Y": frozenset({"X_x_Y", "Y_x_Z"}),
        "X": frozenset({"X_x_Y"}),
        "Y_x_Z": frozenset({"model"}),
        "V": frozenset({"model"}),
    }
    assert d.sorted_variables_names == ('V', 'X', 'Y', 'Z', 'X_x_Y', 'Y_x_Z', 'model')
    assert d.sorted_children == {
        "V": ("model",),
        "X": ("X_x_Y", "model"),
        "Y": ("X_x_Y", "Y_x_Z", "model"),
        "Z": ("Y_x_Z", "model"),
        "X_x_Y": ("model",),
        "Y_x_Z": ("model",),
        "model": (),
    }
    assert d.sorted_ancestors == {
        "V": (),
        "X": (),
        "Y": (),
        "Z": (),
        "X_x_Y": ("X", "Y"),
        "Y_x_Z": ("Y", "Z"),
        "model": ("V", "X", "Y", "Z", "X_x_Y", "Y_x_Z"),
    }
    assert len(d.sorted_variables_by_type) == 3
    assert set(d.sorted_variables_by_type.keys()) == {IndepVariable, IndividualLatentVariable, LinkedVariable}
    for variable in ("V", "Y"):
        assert variable in d.sorted_variables_by_type[IndepVariable]
        assert variable not in d.sorted_variables_by_type[LinkedVariable]
        assert variable not in d.sorted_variables_by_type[IndividualLatentVariable]
    for variable in ("X", "Z"):
        assert variable not in d.sorted_variables_by_type[IndepVariable]
        assert variable not in d.sorted_variables_by_type[LinkedVariable]
        assert variable in d.sorted_variables_by_type[IndividualLatentVariable]
    for variable in ("X_x_Y", "Y_x_Z", "model"):
        assert variable not in d.sorted_variables_by_type[IndepVariable]
        assert variable in d.sorted_variables_by_type[LinkedVariable]
        assert variable not in d.sorted_variables_by_type[IndividualLatentVariable]
    assert d.individual_variable_names == ("X", "Z")
    a, b = d.compute_topological_order_and_path_matrix({}, {})
    assert a == ()
    assert len(b) == 0


def test_dag_with_multiple_components():
    """Tests with a DAG containing 2 connected components."""
    from leaspy.variables.dag import VariablesDAG
    from leaspy.variables.specs import IndepVariable, LinkedVariable

    d = VariablesDAG.from_dict(
        {
            "A": IndepVariable(),
            "B": IndepVariable(),
            "C": IndepVariable(),
            "D": IndepVariable(),
            "E": IndepVariable(),
            "model1": LinkedVariable(lambda *, A, B: A * B),
            "C_plus_D": LinkedVariable(lambda *, C, D: C + D),
            "model2": LinkedVariable(lambda *, C_plus_D, E: C_plus_D * E),
        }
    )

    assert len(d.variables) == 8
    assert d.direct_ancestors == {
        "A": frozenset(),
        "B": frozenset(),
        "C": frozenset(),
        "D": frozenset(),
        "E": frozenset(),
        "model1": frozenset({"A", "B"}),
        "C_plus_D": frozenset({"C", "D"}),
        "model2": frozenset({"C_plus_D", "E"}),
    }
    assert d.direct_children == {
        "B": frozenset({"model1"}),
        "C_plus_D": frozenset({"model2"}),
        "model1": frozenset(),
        "D": frozenset({"C_plus_D"}),
        "A": frozenset({"model1"}),
        "E": frozenset({"model2"}),
        "model2": frozenset(),
        "C": frozenset({"C_plus_D"}),
    }
    assert d.sorted_variables_names == ('A', 'B', 'C', 'D', 'E', 'model1', 'C_plus_D', 'model2')
    assert d.sorted_children == {
        "A": ("model1",),
        "B": ("model1",),
        "C": ("C_plus_D", "model2"),
        "D": ("C_plus_D", "model2"),
        "E": ("model2",),
        "model1": (),
        "C_plus_D": ("model2",),
        "model2": (),
    }
    assert d.sorted_ancestors == {
        "A": (),
        "B": (),
        "C": (),
        "D": (),
        "E": (),
        "model1": ("A", "B"),
        "C_plus_D": ("C", "D"),
        "model2": ("C", "D", "E", "C_plus_D"),
    }
    assert len(d.sorted_variables_by_type) == 2
    assert set(d.sorted_variables_by_type.keys()) == {IndepVariable, LinkedVariable}
    for variable in ("A", "B", "C", "D", "E"):
        assert variable in d.sorted_variables_by_type[IndepVariable]
        assert variable not in d.sorted_variables_by_type[LinkedVariable]
    for variable in ("C_plus_D", "model1", "model2"):
        assert variable not in d.sorted_variables_by_type[IndepVariable]
        assert variable in d.sorted_variables_by_type[LinkedVariable]
    assert d.individual_variable_names == ()
    a, b = d.compute_topological_order_and_path_matrix({}, {})
    assert a == ()
    assert len(b) == 0


def test_dag_error_with_a_single_node_component():
    """Test that an error is raised when one component is made of a single node."""
    from leaspy.variables.dag import VariablesDAG
    from leaspy.variables.specs import IndepVariable, LinkedVariable
    from leaspy.exceptions import LeaspyInputError

    with pytest.raises(
        LeaspyInputError,
        match=re.escape("There are some variables left alone: {'C'}"),
    ):
        VariablesDAG.from_dict(
            {
                "A": IndepVariable(),
                "B": IndepVariable(),
                "C": IndepVariable(),
                "model1": LinkedVariable(lambda *, A, B: A * B),
            }
        )
