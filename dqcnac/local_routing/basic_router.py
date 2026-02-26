"""Basic router class module."""

import logging
from typing import Dict, List, Tuple, Union

import rustworkx as rx
from qiskit.circuit import Qubit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.exceptions import TranspilerError

from ..nonlocal_gate_scheduling import CatDisEntInst, CatEntInst, TeleportInst

logger = logging.getLogger(__name__)


class BasicRouter:
    """
    Perform routing of a DAGCircuit with remote gates onto a `coupling_map` adding swap gates.
    When a cz is does not comply with the coupling map,
    insert one or more swaps in front (on the shortest path) to make it compatible.
    """

    def __init__(
        self,
        coupling_map: CouplingMap,
        ebits: List[int],
        initial_layout: Layout,
        network_to_local: Dict[str, Dict[int, int]],
        reg_to_node: Dict[Qubit, str],
        couplings: Dict[str, CouplingMap],
    ):
        """
        BasicSwap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented the coupling map of the network.
            ebits (list): list of ebits in the coupling map
            initial_layout (Layout): initial layout
            network_to_local (dict): dictionary mapping qubits in the network to qubits on the nodes
            reg_to_node (dict): dictionary mapping circuit register to nodes
            couplings (dict): dictionary mapping nodes to their coupling map
        """
        self.coupling_map = coupling_map
        self.ebits = ebits
        # to keep track of last utilized ebit/ first free beit
        self.timed_ebits = {ebit: 0 for ebit in self.ebits}
        self.initial_layout = initial_layout
        self.network_to_local = network_to_local
        self.reg_to_node = reg_to_node
        self.couplings = couplings
        self.node_to_dist = {node: dict() for node in self.couplings}
        for node in self.network_to_local:
            for d, l in self.network_to_local[node].items():
                self.node_to_dist[node].update({l: d})

    def is_ebit(self, q: Union[int, Qubit], layout) -> bool:
        """
        Check if a qubit is an ebit

        Args:
            q: qubit to check
            layout: layout of the circuit

        Returns:
            True if qubit is an ebit, False otherwise
        """
        if isinstance(q, int):
            return q in self.ebits
        else:
            return layout[q] in self.ebits

    def dijkstra_shortest_path(
        self, coupling: CouplingMap, initial: int, end: int
    ) -> List[int]:
        """
        Use dijkstra algorithm to compute the shortest path between two nodes in a graph

        Args:
            coupling: the graph
            initial: the source node
            end: the target node

        Returns:
            the shortest path
        """

        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()

        edges = list(coupling.graph.edge_list())
        weights = {
            edge: coupling.graph.get_edge_data_by_index(i)
            for i, edge in enumerate(edges)
        }
        while current_node != end:
            visited.add(current_node)
            destinations = coupling.graph.neighbors(current_node)

            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = weights[(current_node, next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {
                node: shortest_paths[node]
                for node in shortest_paths
                if node not in visited
            }

            if not next_destinations:
                return "Route Not Possible"
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        return path

    def shortest_undirected_path(
        self, coupling_map: CouplingMap, physical_qubit1: int, physical_qubit2: int
    ) -> List[int]:
        """
        Compute the shortest undirected path between two physical qubits

        Args:
            coupling_map: the coupling map
            physical_qubit1: the first physical qubit
            physical_qubit2: the second physical qubit

        Returns:
            the path from physical qubit1 to physical qubit2
        """

        return self.dijkstra_shortest_path(
            coupling_map, physical_qubit1, physical_qubit2
        )

    def distance(self, local_map: CouplingMap, node: str, p1: int, p2: int) -> int:
        """
        Compute the distance between two qubits in a coupling map

        Args:
            local_map: the coupling map
            node: the network node to which the qubits pertains
            p1: the first qubit
            p2:  the second qubit

        Returns:
            the distance between the qubits
        """

        index_map = {}
        for q in local_map.graph.nodes():
            index_map[q] = len(index_map)
        distance_matrix = rx.digraph_distance_matrix(
            local_map.graph, as_undirected=True
        )
        return int(
            distance_matrix[
                index_map[self.network_to_local[node][p1]],
                index_map[self.network_to_local[node][p2]],
            ]
        )

    @staticmethod
    def multi_qubit_ops(dag: DAGCircuit) -> List[DAGNode]:
        """
        Get a list of 2 qubit operations. Ignore directives like snapshot and barrier.

        Args:
            dag: the circuit

        Returns:
            a list of two qubit gates in the circuit
        """

        ops = []
        for node in dag.op_nodes(include_directives=False):
            if len(node.qargs) >= 2:
                ops.append(node)
        return ops

    def run(self, dag: DAGCircuit) -> Tuple[DAGCircuit, Layout]:
        """
        Inserts swap ro route a circuit on a coupling map

        Args:
            dag: the circuit

        Returns:
            the routed circuit and the final mapping
        """

        new_dag = dag.copy_empty_like()

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic router runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError(
                "The layout does not match the amount of qubits in the DAG"
            )

        canonical_register = dag.qregs["q"]
        trivial_layout = self.initial_layout
        current_layout = trivial_layout.copy()

        reg_map = {q: q for q in canonical_register}

        for layer in dag.serial_layers():
            subdag = layer["graph"]

            for gate in self.multi_qubit_ops(subdag):
                if isinstance(gate.op, TeleportInst):

                    logger.debug(f"TELEPORT: {gate.op} {gate.qargs}")
                    logger.debug(f"CURRENT LAYOUT: {current_layout}")

                    physical_q0 = current_layout[gate.qargs[0]]
                    physical_q1 = current_layout[gate.qargs[1]]
                    physical_q2 = current_layout[gate.qargs[-2]]
                    physical_q3 = current_layout[gate.qargs[-1]]

                    paths = list()

                    middle = list()
                    if len(gate.qargs) > 4:
                        for qarg in gate.qargs[2:-2]:
                            middle.append(current_layout[qarg])
                        for ebit in middle:
                            self.timed_ebits[ebit] += 1

                    self.timed_ebits[physical_q1] += 1

                    self.timed_ebits[physical_q2] += 1

                    local_map1 = self.couplings[
                        self.reg_to_node[reg_map[gate.qargs[0]]]
                    ]
                    local_map2 = self.couplings[
                        self.reg_to_node[reg_map[gate.qargs[-1]]]
                    ]

                    # local routing between first qubit and first ebit
                    if (
                        self.distance(
                            local_map1,
                            self.reg_to_node[reg_map[gate.qargs[0]]],
                            physical_q0,
                            physical_q1,
                        )
                        > 1
                    ):
                        _path = self.shortest_undirected_path(
                            local_map1,
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q0],
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q1],
                        )
                        path = [
                            self.node_to_dist[self.reg_to_node[reg_map[gate.qargs[0]]]][
                                q
                            ]
                            for q in _path
                        ]
                        if path[0] == physical_q1:
                            path = path[::-1]

                        paths.append(path)

                        for swap in range(len(path) - 2):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                    # local routing between second ebit and second qubit
                    if (
                        self.distance(
                            local_map2,
                            self.reg_to_node[reg_map[gate.qargs[-1]]],
                            physical_q2,
                            physical_q3,
                        )
                        > 1
                    ):
                        _path = self.shortest_undirected_path(
                            local_map2,
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[-1]]]
                            ][physical_q2],
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[-1]]]
                            ][physical_q3],
                        )

                        path = [
                            self.node_to_dist[
                                self.reg_to_node[reg_map[gate.qargs[-1]]]
                            ][q]
                            for q in _path
                        ]
                        if path[0] == physical_q2:
                            path = path[::-1]

                        paths.append(path)

                        for swap in range(len(path) - 2):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                    # update current_layout
                    for path in paths:
                        for swap in range(len(path) - 2):
                            (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            ) = (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            )
                            current_layout.swap(path[swap], path[swap + 1])

                    new_dag.apply_operation_back(
                        gate.op, qargs=[reg_map[qarg] for qarg in gate.qargs]
                    )

                elif isinstance(gate.op, CatEntInst):
                    physical_q0 = current_layout[gate.qargs[0]]
                    physical_q1 = current_layout[gate.qargs[1]]
                    physical_q2 = current_layout[gate.qargs[-1]]

                    logger.debug(f"CATENT: {gate.op} {gate.qargs}")
                    logger.debug(f"CURRENT LAYOUT: {current_layout}")

                    paths = list()

                    middle = list()

                    if len(gate.qargs) > 3:
                        for qarg in gate.qargs[2:-1]:
                            middle.append(current_layout[qarg])
                        for ebit in middle:
                            self.timed_ebits[ebit] += 1

                    self.timed_ebits[physical_q1] += 1

                    self.timed_ebits[physical_q2] += 1

                    local_map1 = self.couplings[
                        self.reg_to_node[reg_map[gate.qargs[0]]]
                    ]

                    try:
                        self.distance(
                            local_map1,
                            self.reg_to_node[reg_map[gate.qargs[0]]],
                            physical_q0,
                            physical_q1,
                        )
                    except Exception as e:
                        logger.error(f"{local_map1.get_edges()}")
                        logger.error(e)

                    # local routing between first qubit and first ebit
                    if (
                        self.distance(
                            local_map1,
                            self.reg_to_node[reg_map[gate.qargs[0]]],
                            physical_q0,
                            physical_q1,
                        )
                        > 1
                    ):
                        _path = self.shortest_undirected_path(
                            local_map1,
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q0],
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q1],
                        )
                        path = [
                            self.node_to_dist[self.reg_to_node[reg_map[gate.qargs[0]]]][
                                q
                            ]
                            for q in _path
                        ]

                        if path[0] == physical_q1:
                            path = path[::-1]

                        paths.append(path)

                        for swap in range(len(path) - 2):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                    # update current_layout
                    for path in paths:
                        for swap in range(len(path) - 2):
                            (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            ) = (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            )
                            current_layout.swap(path[swap], path[swap + 1])

                    new_dag.apply_operation_back(
                        gate.op, qargs=[reg_map[qarg] for qarg in gate.qargs]
                    )

                    if (
                        current_layout[gate.qargs[1]]
                        != self.initial_layout[gate.qargs[1]]
                    ):
                        physical_q0 = current_layout[gate.qargs[1]]
                        physical_q1 = self.initial_layout[gate.qargs[1]]
                        local_map = self.couplings[
                            self.reg_to_node[reg_map[gate.qargs[1]]]
                        ]

                        _path = self.shortest_undirected_path(
                            local_map,
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[1]]]
                            ][physical_q0],
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[1]]]
                            ][physical_q1],
                        )
                        path = [
                            self.node_to_dist[self.reg_to_node[reg_map[gate.qargs[1]]]][
                                q
                            ]
                            for q in _path
                        ]

                        if path[0] == physical_q1:
                            path = path[::-1]
                        paths.append(path)

                        for swap in range(len(path) - 1):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                        self.timed_ebits[self.initial_layout[gate.qargs[1]]] += 1

                        # update current_layout
                        for path in paths:
                            for swap in range(len(path) - 1):
                                (
                                    reg_map[current_layout[path[swap]]],
                                    reg_map[current_layout[path[swap + 1]]],
                                ) = (
                                    reg_map[current_layout[path[swap]]],
                                    reg_map[current_layout[path[swap + 1]]],
                                )
                                current_layout.swap(path[swap], path[swap + 1])

                elif isinstance(gate.op, CatDisEntInst):
                    logger.debug(f"DISENT: {gate.op} {gate.qargs}")
                    logger.debug(f"CURRENT LAYOUT: {current_layout}")

                    new_dag.apply_operation_back(
                        gate.op, qargs=[reg_map[qarg] for qarg in gate.qargs]
                    )

                    paths = list()

                    # if ebit has been moved from original position
                    if (
                        current_layout[gate.qargs[0]]
                        != self.initial_layout[gate.qargs[0]]
                    ):
                        physical_q0 = current_layout[gate.qargs[0]]
                        physical_q1 = self.initial_layout[gate.qargs[0]]
                        local_map = self.couplings[
                            self.reg_to_node[reg_map[gate.qargs[0]]]
                        ]

                        _path = self.shortest_undirected_path(
                            local_map,
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q0],
                            self.network_to_local[
                                self.reg_to_node[reg_map[gate.qargs[0]]]
                            ][physical_q1],
                        )
                        path = [
                            self.node_to_dist[self.reg_to_node[reg_map[gate.qargs[0]]]][
                                q
                            ]
                            for q in _path
                        ]

                        if path[0] == physical_q1:
                            path = path[::-1]
                        paths.append(path)

                        for swap in range(len(path) - 1):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                        self.timed_ebits[self.initial_layout[gate.qargs[0]]] += 1

                        # update current_layout
                        for path in paths:
                            for swap in range(len(path) - 1):
                                (
                                    reg_map[current_layout[path[swap]]],
                                    reg_map[current_layout[path[swap + 1]]],
                                ) = (
                                    reg_map[current_layout[path[swap]]],
                                    reg_map[current_layout[path[swap + 1]]],
                                )
                                current_layout.swap(path[swap], path[swap + 1])

                else:
                    qargs = list(gate.qargs)
                    logger.debug(f"GATE: {gate.op} {gate.qargs}")
                    logger.debug(f"CURRENT LAYOUT: {current_layout}")

                    physical_q0 = current_layout[qargs[0]]
                    physical_q1 = current_layout[qargs[1]]

                    paths = list()

                    local_map = self.couplings[self.reg_to_node[reg_map[qargs[0]]]]

                    # local routing between qubits
                    if (
                        self.distance(
                            local_map,
                            self.reg_to_node[reg_map[qargs[0]]],
                            physical_q0,
                            physical_q1,
                        )
                        != 1
                    ):
                        _path = self.shortest_undirected_path(
                            local_map,
                            self.network_to_local[self.reg_to_node[reg_map[qargs[0]]]][
                                physical_q0
                            ],
                            self.network_to_local[self.reg_to_node[reg_map[qargs[0]]]][
                                physical_q1
                            ],
                        )
                        path = [
                            self.node_to_dist[self.reg_to_node[reg_map[qargs[0]]]][q]
                            for q in _path
                        ]

                        if path[0] in self.ebits and path[-1] not in self.ebits:
                            path = path[::-1]

                        paths.append(path)

                        for swap in range(len(path) - 2):
                            connected_wire_1 = path[swap]
                            connected_wire_2 = path[swap + 1]

                            qubit_1 = current_layout[connected_wire_1]
                            qubit_2 = current_layout[connected_wire_2]

                            # create the swap operation
                            new_dag.apply_operation_back(
                                SwapGate(), qargs=[qubit_1, qubit_2], cargs=[]
                            )

                        # update current_layout
                        for swap in range(len(path) - 2):
                            (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            ) = (
                                reg_map[current_layout[path[swap]]],
                                reg_map[current_layout[path[swap + 1]]],
                            )
                            current_layout.swap(path[swap], path[swap + 1])

                    new_dag.apply_operation_back(
                        gate.op, qargs=[reg_map[qarg] for qarg in gate.qargs]
                    )

        for wire in list(new_dag.idle_wires()):
            new_dag._remove_idle_wire(wire)

        return new_dag, current_layout
