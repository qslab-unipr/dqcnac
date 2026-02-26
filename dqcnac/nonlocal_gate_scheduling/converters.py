"""Circuits converters module."""

import logging

from qiskit.circuit import Measure
from qiskit.circuit.library import (
    Barrier,
    CZGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    U1Gate,
    U2Gate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.dagcircuit import DAGCircuit

from .compilation_state import State
from .nonlocal_operations import CatDisEntInst, CatEntInst, TeleportInst

logger = logging.getLogger(__name__)


def state_to_dag(state: State) -> DAGCircuit:
    """
    Convert a compilation state to a DAGCircuit representation.

    Args:
        state: The compilation state to convert.

    Returns:
        The DAGCircuit representation of the compilation state.
    """
    dag_circuit = DAGCircuit()

    for item in sorted(state.executed, key=lambda x: x[0]):
        for qubit in item[1].qargs:
            if qubit._register.name not in dag_circuit.qregs:
                dag_circuit.add_qreg(qubit._register)
        for clbit in item[1].cargs:
            if clbit._register.name not in dag_circuit.cregs:
                dag_circuit.add_creg(clbit._register)
        cargs = []

        match item[1].op.name:
            case "u1":
                op = U1Gate(theta=item[1].op.params[0])
            case "u2":
                op = U2Gate(phi=item[1].op.params[0], lam=item[1].op.params[1])
            case "u3":
                op = U3Gate(
                    theta=item[1].op.params[0],
                    phi=item[1].op.params[1],
                    lam=item[1].op.params[2],
                )
            case "rx":
                op = RXGate(theta=item[1].op.params[0])
            case "ry":
                op = RYGate(theta=item[1].op.params[0])
            case "rz":
                op = RZGate(phi=item[1].op.params[0])
            case "x":
                op = XGate()
            case "y":
                op = YGate()
            case "z":
                op = ZGate()
            case "h":
                op = HGate()
            case "cz":
                op = CZGate()
            case "CatEnt":
                op = CatEntInst()
            case "CatDisEnt":
                op = CatDisEntInst()
            case "Teleport":
                op = TeleportInst()
            case "barrier":
                op = Barrier(num_qubits=item[1].op.num_qubits)
            case "measure":
                op = Measure()
                cargs = item[1].cargs

        dag_circuit.apply_operation_back(op, qargs=item[1].qargs, cargs=cargs)

    return dag_circuit
