"""Network node class module."""

from qiskit.transpiler import CouplingMap


class Node:
    """
    The Node class representing a node in the network.
    """

    def __init__(
        self,
        name: str,
        num_qubits: int,
        coupling_list: list[tuple],
        ebits: list[int],
        cap: int,
    ):
        """
        Initialize the Node class.

        Args:
            name: The name of the node.
            num_qubits: The total number of qubits available in the node.
            coupling_list: List of tuples representing coupling between qubits.
            ebits: List of qubits that are in error.
            cap: The total capacity of the node.
        """

        self.name = name
        self.mem_cap = num_qubits
        self.active_qubits = 0
        self.ebits = ebits
        self.qubits: list[int] = [i for i in range(cap) if i not in self.ebits]
        self.connections = {}

        self.coupling_map = CouplingMap(couplinglist=coupling_list)
