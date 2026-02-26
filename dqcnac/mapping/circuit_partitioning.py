"""Circuit partitioning with metis."""

from typing import Any, Optional, Union

import numpy as np
import pymetis as mt
from networkx import Graph
from pymetis import Options
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit

from ..network import Network


def equal_partitions(graph: Graph, n_parts: int) -> dict[int, set]:
    """
    Use pymetis to compute equal partitions.

    Args:
        graph: a graph representing the circuit
        n_parts: the number of target partitions

    Returns:
        A dictionary of partitions to qubits
    """

    adjncy = []
    xadj = [0]
    eweights = []

    for node in graph.nodes:
        # xadj.append(len(adjncy))
        for neighbor in graph.neighbors(node):
            adjncy.append(neighbor)
            eweights.append(graph.get_edge_data(node, neighbor)["weight"])
        xadj.append(len(adjncy))

    n_cuts, membership = mt.part_graph(
        n_parts, xadj=xadj, adjncy=adjncy, eweights=eweights, options=Options(ufactor=1)
    )

    partitions = dict()

    for i in range(n_parts):
        partitions[i] = set(np.argwhere(np.array(membership) == i).ravel())

    return partitions


def circuit_partitioning(
    network: Network,
    circuit: Union[QuantumCircuit, DAGCircuit],
    graph: Optional[Graph] = None,
    use_pre_pass=False,
    max_gates_in_a_group: int = 1,
) -> dict[int, int]:
    """
    Compute unequal partitions of *circuit* over the *network* to minimize communication between different nodes

    Args:
        network: the network
        circuit: the circuit to partition
        graph: a weighted graph representing the circuit
        use_pre_pass: a boolean, it should be True if using pre-pass optimizations
        max_gates_in_a_group: the maximum number of gates to consider when computing the partitions

    Returns:
        A dictionary of qubits to partitions
    """

    if graph is None:
        if isinstance(circuit, QuantumCircuit):
            circuit = circuit_to_dag(circuit)

        # cc = check_commutations(circuit)

        circuit_graph = Graph()

        nodes_to_qubits = dict()
        qubits_to_nodes = dict()

        for qubit in circuit.qubits:
            node_idx = len(nodes_to_qubits)
            nodes_to_qubits[node_idx] = qubit
            qubits_to_nodes[qubit] = node_idx
            circuit_graph.add_node(node_idx)

        edges: dict[tuple[int, int], int] = {}

        if use_pre_pass:
            edges_with_count = _EdgesWithCount(max_gates_in_a_group)
            for op in circuit.topological_op_nodes():
                if len(op.qargs) > 1:
                    edges_with_count.input_edges(
                        qubits_to_nodes[op.qargs[0]], qubits_to_nodes[op.qargs[1]]
                    )
            edges_with_count.final_update()
            edges = edges_with_count.edges

        else:
            for op in circuit.topological_op_nodes():
                if len(op.qargs) > 1:
                    edge_nodes = (
                        qubits_to_nodes[op.qargs[0]],
                        qubits_to_nodes[op.qargs[1]],
                    )
                    if edge_nodes in edges:
                        edges[edge_nodes] += 1
                    elif (edge_nodes[1], edge_nodes[0]) in edges:
                        edges[(edge_nodes[1], edge_nodes[0])] += 1
                    else:
                        edges[edge_nodes] = 1

        for edge in edges:
            circuit_graph.add_edge(edge[0], edge[1], weight=edges[edge])
    else:
        circuit_graph = graph

    n_parts = len(network.nodes)

    eq_partitions = equal_partitions(circuit_graph, n_parts)

    qpus = list(network.nodes.keys())
    partition_to_qpu = {partition: qpus[partition] for partition in eq_partitions}

    qubit_to_partition: dict[Optional, Any] = {
        node: None for node in circuit_graph.nodes
    }

    for partition in eq_partitions:
        for node in eq_partitions[partition]:
            qubit_to_partition[node] = partition

    # Try to improve initial solution to exploit less nodes and more qubits per node
    qubits = set(circuit_graph.nodes)

    while len(qubits) != 0:
        move_gains: dict[Optional, Any] = {qubit: None for qubit in qubits}

        for qubit in qubits:
            local_cost = 0
            partition_number = qubit_to_partition[qubit]
            partition_set = eq_partitions[partition_number]
            partition_deviation = (
                len(partition_set)
                - network.nodes[partition_to_qpu[partition_number]].mem_cap
            )
            local_cost += sum(
                [
                    circuit_graph.get_edge_data(qubit, neigh)["weight"]
                    / (2**partition_deviation)
                    for neigh in circuit_graph.neighbors(qubit)
                    if qubit_to_partition[neigh] == partition_number
                ]
            )
            move_gain = {}

            for partition in eq_partitions:
                remote_cost = 0
                if partition != qubit_to_partition[qubit]:
                    remote_cost += sum(
                        [
                            circuit_graph.get_edge_data(qubit, remote)["weight"]
                            for remote in circuit_graph.neighbors(qubit)
                            if qubit_to_partition[remote] == partition
                        ]
                    )

                move_gain[partition] = remote_cost - local_cost

            best_move = (None, 0)

            for move in move_gain:
                if (
                    move_gain[move] > 0
                    and len(eq_partitions[move])
                    < network.nodes[partition_to_qpu[move]].mem_cap
                ):
                    if best_move is None or move_gain[move] > best_move[1]:
                        best_move = (move, move_gain[move])
            move_gains[qubit] = best_move

        selected_move = max(move_gains.keys(), key=lambda x: move_gains[x][1])

        if move_gains[selected_move][0] is not None:
            eq_partitions[qubit_to_partition[selected_move]].remove(selected_move)
            eq_partitions[move_gains[selected_move][0]].add(selected_move)
            qubit_to_partition[selected_move] = move_gains[selected_move][0]
            qubits.remove(selected_move)
        else:
            break

    # look if there are still too many qubits in a node
    qubits_in_excess = set()

    for ind_part, partition in eq_partitions.items():
        while len(partition) > network.nodes[partition_to_qpu[ind_part]].mem_cap:
            qubits_in_excess.add(partition.pop())
    if qubits_in_excess:
        for ind_part, partition in eq_partitions.items():
            while (
                len(partition) < network.nodes[partition_to_qpu[ind_part]].mem_cap
                and qubits_in_excess
            ):
                partition.add(qubits_in_excess.pop())
        for partition in eq_partitions:
            for node in eq_partitions[partition]:
                qubit_to_partition[node] = partition

    if graph is not None:
        return eq_partitions
    else:
        return {
            q: partition_to_qpu[qubit_to_partition[qubits_to_nodes[q]]]
            for q in qubits_to_nodes
        }


class _EdgesWithCount:
    """Internal class used to more easily count the edges when grouping is active"""

    def __init__(self, max_gates_in_a_group):
        """
        Initializes the class

        Args:
            max_gates_in_a_group (int): the max number of gates allowed in a group. If less than 1, is ignored
        """
        self._last_edges_used: dict[tuple[int, int], int] = {}
        self._edge_connected_with: dict[int, int] = {}
        self.edges: dict[tuple[int, int], int] = {}
        self._max_gates_in_a_group = max_gates_in_a_group

    def _set_edge(self, edge_start: int, edge_end: int, count: int):
        """Sets an edge to a starting count"""
        self._last_edges_used[(edge_start, edge_end)] = count
        self._edge_connected_with[edge_start] = edge_end
        self._edge_connected_with[edge_end] = edge_start

    def _increase_count(self, edge_start: int, edge_end: int, amount: int):
        """Increases an edge count by amount"""
        if (edge_start, edge_end) in self._last_edges_used:
            self._last_edges_used[(edge_start, edge_end)] += amount
        else:
            self._last_edges_used[(edge_end, edge_start)] += amount

    def _check_last_edge_used(self, edge_start: int, edge_end: int):
        """Checks if an edge was used recently"""
        return (
            edge_start in self._edge_connected_with
            and self._edge_connected_with[edge_start] == edge_end
        )

    def _update_edges(self, edge_start: int, edge_end: int):
        """Updates the final edge count relative to the input edge"""
        if (edge_start, edge_end) in self.edges:
            self.edges[(edge_start, edge_end)] = (
                self.edges.setdefault((edge_start, edge_end), 0) + 1
            )
        else:
            self.edges[(edge_end, edge_start)] = (
                self.edges.setdefault((edge_end, edge_start), 0) + 1
            )

    def _clean_and_update_edge(self, edge_side: int):
        """Cleans the temporary count and updates the final edge count relative to the input edge"""
        if edge_side in self._edge_connected_with:
            edge_to_remove = self._edge_connected_with.pop(edge_side)
            del self._edge_connected_with[edge_to_remove]
            if (edge_side, edge_to_remove) in self._last_edges_used:
                del self._last_edges_used[(edge_side, edge_to_remove)]
            else:
                del self._last_edges_used[(edge_to_remove, edge_side)]

            self._update_edges(edge_side, edge_to_remove)

    def _amount_of_repetitions(self, edge_start: int, edge_end: int):
        """Counts the amount of repetitions of the input edge"""
        if (edge_start, edge_end) in self._last_edges_used:
            return self._last_edges_used[(edge_start, edge_end)]
        else:
            return self._last_edges_used[(edge_end, edge_start)]

    def input_edges(self, edge_start, edge_end):
        """
        Performs all the operations that need to be performed on the input edge

        Args:
            edge_start (int): the start of the input edge
            edge_end (int): the end of the input edge
        """
        if self._check_last_edge_used(edge_start, edge_end):
            self._increase_count(edge_start, edge_end, 1)
            if (
                self._amount_of_repetitions(edge_start, edge_end)
                == self._max_gates_in_a_group
            ):
                # if max_gates_in_a_group == 0, then it never enters
                self._clean_and_update_edge(edge_start)
                self._clean_and_update_edge(edge_end)
        else:
            self._clean_and_update_edge(edge_start)
            self._clean_and_update_edge(edge_end)
            self._set_edge(edge_start, edge_end, 1)

    def final_update(self):
        """Updates the final edge count from the temporary counts still not finalized"""
        for edge in self._last_edges_used:
            self.edges[edge] = self.edges.setdefault(edge, 0) + 1
