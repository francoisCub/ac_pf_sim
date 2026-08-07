# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# Modified by University of Liège in 2026. 
# Original DC power flow problem example changed to AC power flow.


import copy
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from energnn.converter import Converter, ElementsConverter
from energnn.graph import Graph, GraphShape, GraphStructure, JaxBackend, NumpyBackend, collate_graphs
from energnn.problem import ProblemBatch, ProblemLoader, Problem


class _LineElementsConverter(ElementsConverter):
    """Extracts line parameters: conductance (g) and susceptance (b)."""

    def _get_table(self, *, G: np.ndarray, B: np.ndarray, **kwargs) -> pd.DataFrame:
        rows, cols = np.nonzero(np.triu(B, k=1))
        return pd.DataFrame({
            "from": rows, 
            "to": cols, 
            "g": G[rows, cols], 
            "b": B[rows, cols]
        })

class _BusElementsConverter(ElementsConverter):
    """Extracts bus injections: active power (p) and reactive power (q)."""

    def _get_table(self, *, P: np.ndarray, Q: np.ndarray, V_set: np.ndarray, is_pq: np.ndarray, is_pv: np.ndarray, is_slack: np.ndarray, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            "id": np.arange(P.shape[0]), 
            "p": P, 
            "q": Q,
            "v_set": V_set,
            "is_pq":is_pq,
            "is_pv": is_pv,
            "is_slack": is_slack,
        })


class _OracleBusElementsConverter(ElementsConverter):
    """Extracts ACPF solution: voltage magnitude (v) and phase angle (theta)."""

    def _get_table(self, *, P: np.ndarray, Q: np.ndarray, V: np.ndarray, theta: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "p": P, 
            "q": Q,
            "v": V, 
            "theta": theta
        })


class ACSystemContextConverter(Converter):
    """Converts a AC system ``(G, B, P, Q)`` into a context :class:`energnn.graph.Graph`."""

    def __init__(self):
        self.elements_converter_dict = {
            "line": _LineElementsConverter(port_list=["from", "to"], feature_list=["g", "b"]),
            "bus": _BusElementsConverter(port_list=["id"], feature_list=["p", "q", "v_set", "is_pq", "is_pv", "is_slack"]),
        }


class ACSystemOracleConverter(Converter):
    """Converts the solution vector V and ``theta`` into an oracle :class:`energnn.graph.Graph`.

    Oracles only carry features: their hyper-edges have no ports, hence no address registry.
    """

    def __init__(self):
        self.elements_converter_dict = {
            "bus": _OracleBusElementsConverter(port_list=None, feature_list=["p", "q", "v", "theta"]),
        }


AC_SYSTEM_CONTEXT_STRUCTURE = ACSystemContextConverter().get_structure()
AC_SYSTEM_DECISION_STRUCTURE = ACSystemOracleConverter().get_structure()


class ACSystemProblemBatch(ProblemBatch):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return AC_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return AC_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def save(self, *, path: str) -> None:
        pass


class ACSystemProblem(Problem):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return AC_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return AC_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[float, dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array))
        return float(objective), {}

    def save(self, *, path: str) -> None:
        pass


def _generate_sparse_AC_SYSTEM(n, m, vmin=0.9, vmax=1.1, thetamin=-np.pi/6, thetamax=np.pi/6, rmin=0.01, rmax=0.1, xmin=0.1, xmax=1.0, shunt_conductance=0.1):
    """Generates sparse matrix G, B and vectors P, Q and theta such that Kirchoff laws are respected for a AC network."""
    # Ensure connectivity by building a spanning tree first
    B = np.zeros((n, n))
    G = np.zeros((n, n))
    nodes = np.arange(n)
    np.random.shuffle(nodes)
    # Spanning tree
    for i in range(n - 1):
        u, v = nodes[i], nodes[i+1]
        r, x = np.random.uniform(rmin, rmax), np.random.uniform(xmin, xmax)
        denom = r**2+x**2
        g, b = r/denom, x/denom
        G[u, v] = G[v, u] = -g
        B[u, v] = B[v, u] = b

    # Add remaining m - (n-1) edges among the still-free upper-triangular pairs
    iu, ju = np.triu_indices(n, k=1)
    free = np.flatnonzero(B[iu, ju] == 0)
    n_extra = min(m - (n - 1), free.size)
    if n_extra > 0:
        idxs = np.random.choice(free, n_extra, replace=False)
        for idx in idxs:
            u, v = iu[idx], ju[idx]
            r, x = np.random.uniform(rmin, rmax), np.random.uniform(xmin, xmax)
            denom = r**2+x**2
            g, b = r/denom, x/denom
            G[u, v] = G[v, u] = -g
            B[u, v] = B[v, u] = b

    np.fill_diagonal(G, -G.sum(axis=1))
    np.fill_diagonal(B, -B.sum(axis=1))
    Y = G + 1j * B
    Y += np.eye(n) * shunt_conductance

    V_mag = np.random.uniform(vmin, vmax, n)
    theta = np.random.uniform(thetamin, thetamax, n)
    V_complex = V_mag * np.exp(1j * theta)

    I_complex = Y @ V_complex
    S_complex = V_complex * np.conj(I_complex)

    P = np.real(S_complex)
    Q = np.imag(S_complex)

    is_slack = np.zeros(n, dtype=int)
    is_pv = np.zeros(n, dtype=int)
    is_pq = np.zeros(n, dtype=int)

    # slack bus
    slack_idx = np.argmax(P)
    is_slack[slack_idx] = 1

    # PV buses
    pv_mask = (P > 0) & (np.arange(n) != slack_idx)
    is_pv[pv_mask] = 1

    # PQ buses
    is_pq = 1 - (is_slack + is_pv)

    # Center and rescale targets
    V_mag = (V_mag - (vmax+vmin) / 2 ) / (vmax-vmin)
    theta = (theta - (thetamax+thetamin) / 2) / (thetamax-thetamin)

    return G, B, P, Q, V_mag, theta, is_pq, is_pv, is_slack

class ACSystemProblemGenerator:
    __test__ = False
    """Generates random sparse linear systems."""

    def __init__(self, *, seed: int = 0, n_max: int = 32):

        self.seed = seed
        self.n_max = n_max

        self.context_converter = ACSystemContextConverter()
        self.oracle_converter = ACSystemOracleConverter()

        np.random.seed(seed)

    def generate_problem(self, backend: JaxBackend | NumpyBackend | None = None) -> ACSystemProblem:
        # The converters build graphs on a numpy backend: their shapes vary from one problem to
        # the next, and building them directly in jax would trigger one XLA compilation per new shape.
        if backend is None:
            backend = JaxBackend()
        n = np.random.randint(2, self.n_max + 1)
        m = np.random.randint(n - 1, 3 * n)
        G, B, P, Q, V, theta, is_pq, is_pv, is_slack = _generate_sparse_AC_SYSTEM(n, m)

        context = self.context_converter(G=G, B=B, P=P, Q=Q, V_set=V, is_pq=is_pq, is_pv=is_pv, is_slack=is_slack)
        oracle = self.oracle_converter(P=P, Q=Q, V=V, theta=theta)

        if isinstance(backend, NumpyBackend):
            return ACSystemProblem(context=context, oracle=oracle)
        return ACSystemProblem(context=context.to_backend(backend), oracle=oracle.to_backend(backend))

    def generate_problem_batch(self, batch_size: int = 8) -> ACSystemProblemBatch:

        context_list, oracle_list = [], []

        numpy_backend = NumpyBackend()
        for _ in range(batch_size):
            problem = self.generate_problem(backend=numpy_backend)
            context = problem.context
            oracle = problem.oracle
            context_list.append(context)
            oracle_list.append(oracle)

        max_context_shape = GraphShape(
            backend=numpy_backend,
            hyper_edge_sets={
                "line": np.array(self.n_max * 3),
                "bus": np.array(self.n_max),
            },
            addresses=np.array(self.n_max),
        )
        # Oracles carry no ports, hence an empty address registry.
        max_oracle_shape = GraphShape(
            backend=numpy_backend, hyper_edge_sets={"bus": np.array(self.n_max)}, addresses=np.array(0)
        )

        # Padding and collating are done in numpy (variable shapes are free there); the padded
        # batch has a fixed shape, so the final conversion to jax compiles only once.
        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs(context_list).to_backend(JaxBackend())
        oracle_batch = collate_graphs(oracle_list).to_backend(JaxBackend())

        return ACSystemProblemBatch(context=context_batch, oracle=oracle_batch)


class ACSystemProblemLoader(ProblemLoader):
    __test__ = False

    def __init__(
        self,
        seed: int = 0,
        dataset_size: int = 32,
        batch_size: int = 8,
        n_max: int = 4,
        shuffle: bool = False,
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.n_max = n_max
        self.shuffle = shuffle
        self.len = dataset_size
        self.current_step = 0

        self.generator = ACSystemProblemGenerator(seed=seed, n_max=n_max)
        # The loader resets its RNG at each epoch, so every epoch regenerates the exact same
        # batches: they are generated once and cached.
        self._batch_cache: list[ACSystemProblemBatch] = []

    @property
    def decision_structure(self) -> GraphStructure:
        return AC_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return AC_SYSTEM_CONTEXT_STRUCTURE

    def __iter__(self):
        self.current_step = 0
        np.random.seed(self.seed)
        return self

    def __next__(self) -> ACSystemProblemBatch:
        if self.current_step >= self.len:
            raise StopIteration
        batch_start = self.current_step
        batch_end = min(self.current_step + self.batch_size, self.len)
        self.current_step = batch_end
        n_batch = batch_end - batch_start
        batch_index = batch_start // self.batch_size
        if batch_index < len(self._batch_cache):
            return self._batch_cache[batch_index]
        batch = self.generator.generate_problem_batch(batch_size=n_batch)
        self._batch_cache.append(batch)
        return batch

    def __len__(self):
        return max(self.dataset_size // self.batch_size, 1)
