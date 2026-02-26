"""Statistics module for distributed quantum circuit analysis."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from ..nonlocal_gate_scheduling import CatEntInst, TeleportInst


def get_num_epr(circuit: QuantumCircuit) -> int:
    """
    Compute the number of EPR pairs in a distributed quantum circuit.

    Args:
        circuit: The quantum circuit to analyze.

    Returns:
        The number of EPR pairs in the circuit.
    """
    num_eprs = 0
    for node in circuit_to_dag(circuit).topological_op_nodes():
        if isinstance(node.op, CatEntInst):
            num_eprs += len(node.qargs) // 2
        elif isinstance(node.op, TeleportInst):
            num_eprs += (len(node.qargs) - 2) // 2

    return num_eprs
