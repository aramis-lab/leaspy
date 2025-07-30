"""This module defines the `VariablesDAG` class used to represent the relationships between the variables of a model."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import (
    Mapping as TMapping,
)
from typing import Type

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.filtered_mapping_proxy import FilteredMappingProxy

from .specs import (
    IndividualLatentVariable,
    VariableInterface,
    VariableName,
    VariablesToFrozenSet,
)

__all__ = ["VariablesDAG"]


@dataclass(frozen=True)
class VariablesDAG(Mapping):
    """
    Directed acyclic graph of symbolic variables used in a leaspy model with efficient topologically sorted bidirectional access.

    Parameters
    ----------
    variables : :class:`~typing.Mapping` [ :class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.VariableInterface`]
        A dictionary mapping variable names to their specification objects (e.g., `IndepVariable`, `LinkedVariable`, etc.).
    direct_ancestors : :class:`~leaspy.variables.specs.VariablesToFrozenSet`
        A dictionary mapping each variable name to the set of variable names it directly depends on (i.e., its parents).
        Use :meth:`~leaspy.variables.dag.VariablesDAG.from_dict()` class method to reach the natural dependencies of linked variables.

    Notes
    -----

    Internally, this class precomputes:

    - ``direct_children``: inverse of `direct_ancestors`, mapping each node to its immediate dependents.

    - ``sorted_variables_names``: a topological ordering of variable names (roots to leaves).
        
    - ``sorted_children``: all transitive children of a node in topological order.
        
    - ``sorted_ancestors``: all transitive ancestors of a node in topological order.
        
    - ``sorted_variables_by_type``: variables grouped and ordered by their Python class/type.

    In general the VariablesDAG is not a tree because the graph may not be totally connected and may have multiple roots.
    It is not a multi-tree either since there may be multiple directed paths between two nodes - e.g. `logistic_model = f[g, b(g), ...]`.
    However, we do assume that there is no cycle in the graph (not checked currently), which is equivalent to be topologically sortable.

    In order to improve the efficiency of the algorithms, a node-wise sorted children and ancestors mappings are computed.
    These mappings are useful to:

    - perform computations and caching of intermediate variable dependencies
    - quickly reset all dependent nodes upon a modification

    Finally, we do not store children nor ancestors in a specific node class to avoid cross-references in such nodes.

    To Do
    -----
    - pre-compute roots (no ancestors) and leaves (no children) as well?
    - stratify variables dictionary per variable class?

    References
    ----------
    https://en.wikipedia.org/wiki/Directed_acyclic_graph#Computational_problems

    Examples
    --------
    >>> from leaspy.variables import VariablesDAG
    >>> from leaspy.variables.specs import IndepVariable, LinkedVariable
    >>> d_vars = {
        "x": IndepVariable(),
        "y": LinkedVariable(lambda *, x: -x),
    }
    >>> dag = VariablesDAG.from_dict(d_vars)
    """

    variables: TMapping[VariableName, VariableInterface]
    """Dictionary of variable names to their corresponding variable specification objects."""

    direct_ancestors: VariablesToFrozenSet = field(repr=False)
    """Mapping of variable names to their direct ancestors."""

    # pre-computed data that only depend on frozen attributes of dataclass
    direct_children: VariablesToFrozenSet = field(init=False, repr=False, compare=False)
    """Mapping of variable names to their direct children."""

    sorted_variables_names: tuple[VariableName, ...] = field(
        init=False, repr=False, compare=False
    )
    """Tuple of variable names sorted in topological order (from roots to leaves)."""
    # path_matrix: torch.Tensor = field(init=False, repr=False, compare=False)  # big and useless?
    sorted_children: TMapping[VariableName, tuple[VariableName, ...]] = field(
        init=False, repr=False, compare=False
    )
    """Mapping of all children of a given variable in a topological order."""

    sorted_ancestors: TMapping[VariableName, tuple[VariableName, ...]] = field(
        init=False, repr=False, compare=False
    )
    """Mapping of all ancestor of a given variable in a topological order."""

    sorted_variables_by_type: TMapping[
        # a better type hint would be nice here: Type[cls] -> Mapping[VariableName, cls]
        Type[VariableInterface],
        TMapping[VariableName, VariableInterface],
    ] = field(init=False, repr=False, compare=False)
    """The sorted variables, but stratified per variable type, to easily access them."""

    @classmethod
    def from_dict(cls, input_dictionary: TMapping[VariableName, VariableInterface]):
        """
        Instantiate a new DAG of variables from a dictionary of variables.

        This method is using :class:`~leaspy.variables.specs.LinkedVariable` dependencies as direct ancestors.

        Parameters
        ----------
        input_dictionary : :class:`~typing.Mapping` [:class:`~leaspy.variables.specs.VariableName` , :class:`~leaspy.variables.specs.VariableInterface`]
            The dictionary to use to create the DAG.

        Returns
        -------
        :class:`VariablesDAG`
            A new instance of `VariablesDAG` with the computed direct ancestor relationships.

        """
        direct_ancestors = {
            variable_name: variable.get_ancestors_names()
            for variable_name, variable in input_dictionary.items()
        }
        return cls(input_dictionary, direct_ancestors=direct_ancestors)

    def __post_init__(self):
        """
        Post-initialization routine to compute and cache derived attributes for the DAG.
        """
        nodes = self._check_consistency_of_nodes()
        children = self._compute_direct_children(nodes)
        (
            sorted_variables_names,
            sorted_children,
            sorted_ancestors,
        ) = self._compute_topological_orders(children)
        d_types = self._stratify_variables(sorted_variables_names)
        # Cache all those pre-computations values while keeping a "frozen" dataclass
        object.__setattr__(self, "direct_children", children)
        object.__setattr__(self, "sorted_variables_names", sorted_variables_names)
        # object.__setattr__(self, "path_matrix", path_matrix)
        object.__setattr__(self, "sorted_children", sorted_children)
        object.__setattr__(self, "sorted_ancestors", sorted_ancestors)
        object.__setattr__(self, "sorted_variables_by_type", d_types)

    def _check_consistency_of_nodes(self) -> frozenset[VariableName]:
        """
        Ensure internal consistency between variables and their declared ancestors.

        Returns
        -------
        :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]
            The set of all variable names (i.e. DAG nodes) in the graph.

        Raises
        ------
        :exc:`ValueError`
            If there is a mismatch between the declared variable names and the keys in `direct_ancestors`.

        """
        nodes = frozenset(self.variables.keys())
        if nodes != self.direct_ancestors.keys():
            raise ValueError("Inconsistent nodes in dictionary of ancestors edges")
        self._raise_if_bad_nodes_in_edges(
            self.direct_ancestors, nodes, what="ancestors"
        )
        return nodes

    def _compute_direct_children(
        self, nodes: frozenset[VariableName]
    ) -> VariablesToFrozenSet:
        """
        Compute children for efficient bidirectional access.
        
        Parameters
        ----------
        nodes : :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]
            The set of all variable names in the DAG.

        Returns
        -------
        :class:`~leaspy.variables.specs.VariablesToFrozenSet`
            A dictionary mapping each variable name to a frozenset of its direct children.

        Raises
        ------
        :exc:`ValueError`
            If a node appears isolated—i.e., not listed as an ancestor or child in either mapping.
        """
        children = defaultdict(set)
        for child_name, set_ancestors in self.direct_ancestors.items():
            for ancestor in set_ancestors:
                children[ancestor].add(child_name)
        children = {var_name: frozenset(children[var_name]) for var_name in nodes}
        self._raise_if_left_alone_nodes(children, self.direct_ancestors)
        return children

    def _compute_topological_orders(
        self,
        children: VariablesToFrozenSet,
    ) -> tuple[
        tuple[VariableName, ...],
        TMapping[VariableName, tuple[VariableName, ...]],
        TMapping[VariableName, tuple[VariableName, ...]],
    ]:
        """
        Compute the topological order of variables and their transitive relationships.

        Parameters
        ----------
        children : :class:`~leaspy.variables.specs.VariablesToFrozenSet`
            Mapping from each variable to its direct children.

        Returns
        -------
        sorted_variables_names : :obj:`tuple`[:class:`~leaspy.variables.specs.VariableName`, ...]
            A tuple of variable names sorted in topological order (from roots to leaves).
        sorted_children : :class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :obj:`tuple`:class:`~leaspy.variables.specs.VariableName`, ...]]
            A dictionary mapping each variable to a tuple of its transitive children,
            ordered according to the topological sort.
        sorted_ancestors : :class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :obj:`tuple`[:class:`~leaspy.variables.specs.VariableName`, ...]]
            A dictionary mapping each variable to a tuple of its transitive ancestors,
            ordered according to the topological sort.
        """
        (
            sorted_variables_names,
            path_matrix,
        ) = self.compute_topological_order_and_path_matrix(
            children, self.direct_ancestors
        )
        sorted_children, sorted_ancestors = self.compute_sorted_children_and_ancestors(
            sorted_variables_names, path_matrix
        )
        return sorted_variables_names, sorted_children, sorted_ancestors

    def _stratify_variables(
        self,
        sorted_variables_names: tuple[VariableName, ...],
    ) -> TMapping[Type[VariableInterface], TMapping[VariableName, VariableInterface]]:
        """
        Stratification of variables, per variable type.

        Parameters
        ----------
        sorted_variables_names : :obj:`tuple`[:class:`~leaspy.variables.specs.VariableName`, ...]
            A topologically sorted tuple of variable names, typically from roots to leaves.

        Returns
        -------
        :class:`~typing.Mapping`[:obj:`Type`[:class:`~leaspy.variables.specs.VariableInterface`], :class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :class:`~leaspy.variables.specs.VariableInterface`]]
            A dictionary mapping each variable class/type to a filtered, read-only mapping of
            variable names to variable instances of that type.
        """
        d_types = defaultdict(list)
        for var_name in sorted_variables_names:
            d_types[type(self.variables[var_name])].append(var_name)
        return {
            var_type: FilteredMappingProxy(self.variables, subset=tuple(l_vars_type))
            for var_type, l_vars_type in d_types.items()
        }

    def __iter__(self):
        """
        Iterates on keys in topological order (.keys(), .values() and .items() methods are automatically provided by :class:`~typing.Mapping`).
        """
        return iter(self.sorted_variables_names)

    def __len__(self) -> int:
        """Get number of nodes."""
        return len(self.variables)

    def __getitem__(self, variable_name: VariableName) -> VariableInterface:
        """Get the variable specifications."""
        return self.variables[variable_name]

    @staticmethod
    def _raise_if_bad_nodes_in_edges(
        d_edges: VariablesToFrozenSet,
        s_nodes: frozenset[VariableName],
        *,
        what: str,
    ) -> None:
        """
        Validate that all referenced nodes in edge mappings are known and that there are no self-references.

        Parameters
        ----------
        d_edges : :class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]]
            A dictionary mapping each node to a frozenset of connected nodes (e.g., ancestors or children).
        s_nodes : :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]
            The set of valid nodes expected in the graph.
        what : :obj:`str`
            A descriptive label for the type of edges being checked (e.g., "ancestors", "children"),
            used in error messages.

        Raises
        ------
        :exc:`LeaspyInputError`
            If there are unknown nodes referenced in the edges, or if any self-loops are detected.
        """
        pooled_nodes_from_edges = set().union(*d_edges.values())
        unknown_nodes = pooled_nodes_from_edges.difference(s_nodes)
        if len(unknown_nodes):
            raise LeaspyInputError(
                f"Those {what} variables are unknown: {unknown_nodes}"
            )
        self_loops = {n for n, s_connected in d_edges.items() if n in s_connected}
        if len(self_loops):
            raise LeaspyInputError(f"Those variables have self {what}: {self_loops}")

    @staticmethod
    def _raise_if_left_alone_nodes(
        d_children: VariablesToFrozenSet,
        d_ancestors: VariablesToFrozenSet,
    ) -> None:
        """
        Raise an error if any node has neither children nor ancestors, indicating an isolated node.
        
        Parameters
        ----------
        d_children : :class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]]
            Mapping from each variable to its direct children.

        d_ancestors ::class:`~typing.Mapping`[:class:`~leaspy.variables.specs.VariableName`, :obj:`frozenset`[:class:`~leaspy.variables.specs.VariableName`]]
            Mapping from each variable to its direct ancestors.

        Raises
        ------
        :exc:`LeaspyInputError`
            If any variables are detected that have no children and no ancestors, i.e., are left completely alone.
        """
        s_left_alone = {
            var_name
            for var_name, s_ancestors in d_children.items()
            if len(s_ancestors) == 0 and len(d_ancestors[var_name]) == 0
        }
        if len(s_left_alone):
            raise LeaspyInputError(
                f"There are some variables left alone: {s_left_alone}"
            )

    @staticmethod
    def compute_topological_order_and_path_matrix(
        direct_children: VariablesToFrozenSet,
        direct_ancestors: VariablesToFrozenSet,
    ) -> tuple[tuple[VariableName, ...], torch.Tensor]:
        """P
        roduce a topological sorting of the DAG.

        This relies on a modified Kahn's algorithm to produce a topological sorting of DAG,
        and the corresponding path matrix as a by-product.

        Parameters
        ----------
        direct_children : :class:`~leaspy.variables.specs.VariablesToFrozenSet`
            Mapping from each variable to its direct children.
        direct_ancestors : :class:`~leaspy.variables.specs.VariablesToFrozenSet`
            Mapping from each variable to its direct ancestors.

        Returns
        -------
        sorted_nodes : :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, ...]
            Nodes in a topological order.
        path_matrix : :class:`torch.Tensor` [:obj:`bool`]
            Boolean triangle superior (strict) matrix indicating whether
            there is a (directed) path between nodes.

        Notes
        -----
        The algorithm has linear complexity with the O(number of edges + number of nodes).
        Input nodes are sorted by name in order to have fully reproducible output
        of the initial order of nodes and edges.
        (Thus renaming nodes may change the output, due to non-uniqueness of topological order)
        """
        nodes = sorted(direct_ancestors.keys())
        if set(nodes) != direct_children.keys():
            raise ValueError(
                "The nodes in provided 'direct_ancestors' do not match "
                "the nodes in provided 'direct_children'."
            )
        n_nodes = len(nodes)
        ix_nodes = {n: i for i, n in enumerate(nodes)}
        # copy of direct_ancestors & direct_children, with fixed order of nodes
        direct_ancestors_ = {n: direct_ancestors[n] for n in nodes}
        direct_children_ = {n: sorted(direct_children[n]) for n in nodes}
        # from roots (no ancestors) to leaves (no children)
        sorted_nodes: tuple[VarName, ...] = ()
        # indices of matrix correspond to `ix_nodes` until topological order is found
        path_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.bool)
        q_roots = SimpleQueue()
        for n, s_ancestors in direct_ancestors_.items():
            if len(s_ancestors) == 0:
                q_roots.put(n)
        while not q_roots.empty():
            n = q_roots.get()
            sorted_nodes += (n,)
            i = ix_nodes[n]
            for m in direct_children_[n]:
                j = ix_nodes[m]
                path_matrix[:, j] |= path_matrix[:, i]
                path_matrix[i, j] = True
                # drop edge (of the local copy of edges); no need to drop in `direct_children_`
                direct_ancestors_[m] = direct_ancestors_[m].difference({n})
                if len(direct_ancestors_[m]) == 0:
                    q_roots.put(m)
        if set(sorted_nodes) != set(nodes):
            raise ValueError("Input graph is not a DAG")
        # reorder elements of path matrix before returning it
        ix_sorted_nodes = [ix_nodes[n] for n in sorted_nodes]
        path_matrix = path_matrix[ix_sorted_nodes, :][:, ix_sorted_nodes]
        if not torch.equal(path_matrix, path_matrix.triu(1)):
            raise ValueError(
                f"Input graph is not a DAG: sorted path matrix = {path_matrix}"
            )
        return sorted_nodes, path_matrix

    @staticmethod
    def compute_sorted_children_and_ancestors(
        sorted_nodes: tuple[VariableName, ...],
        path_matrix: torch.Tensor,
    ) -> tuple[
        dict[VariableName, tuple[VariableName, ...]],
        dict[VariableName, tuple[VariableName, ...]],
    ]:
        """
        Produce node-wise topologically sorted children and ancestors from provided nodes.

        Parameters
        ----------
        sorted_nodes : :obj:`tuple` of :class:`~leaspy.variables.specs.VariableName`
            The sorted nodes.
        path_matrix : :class:`torch.Tensor`
            A binary 2D tensor where `path_matrix[i, j] == 1` indicates a path from node `i` to node `j`.
            Assumes the row/column indices correspond to the order of `sorted_nodes`.

        Returns
        -------
        sorted_children : :obj:`dict` [:class:`~leaspy.variables.specs.VariableName`, :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, ...]]
            The sorted children.
        sorted_ancestors : :obj:`dict` [:class:`~leaspy.variables.specs.VariableName`, :obj:`tuple` [:class:`~leaspy.variables.specs.VariableName`, ...]]
            The sorted ancestors.
        """
        sorted_children = {
            node: tuple(
                sorted_nodes[j]
                for j in path_matrix[idx_node, :].nonzero(as_tuple=False).squeeze(dim=1)
            )
            for idx_node, node in enumerate(sorted_nodes)
        }
        sorted_ancestors = {
            node: tuple(
                sorted_nodes[i]
                for i in path_matrix[:, idx_node].nonzero(as_tuple=False).squeeze(dim=1)
            )
            for idx_node, node in enumerate(sorted_nodes)
        }
        return sorted_children, sorted_ancestors

    @property
    def individual_variable_names(self) -> tuple[VariableName, ...]:
        """
        Returns a tuple of variable names corresponding to the individual variables.

        Returns
        -------
        :obj:`tuple` of :class:`~leaspy.variables.specs.VariableName` :
            The individual variable names.
        """
        try:
            return tuple(self.sorted_variables_by_type[IndividualLatentVariable].keys())
        except KeyError:
            return ()
