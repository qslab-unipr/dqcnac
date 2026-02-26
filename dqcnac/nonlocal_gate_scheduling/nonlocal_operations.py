from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.dagcircuit.dagnode import DAGOpNode


class CatEntInst(Gate):
    """
    CatEnt gate for nonlocal quantum operations.
    """

    def __init__(self, label=None):
        super().__init__("CatEnt", 3, [], label)

    @property
    def _qasm_definition(self):
        qc = self._qc()
        qubit_to_qasm = {bit: f"p{i}" for i, bit in enumerate(qc.qubits)}
        return (
            "gate "
            + self.name.lower()
            + " "
            + ",".join(qubit_to_qasm[qubit] for qubit in qc.qubits)
            + ";\n"
        )

    def _qc(self):
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(Gate(self.name, self.num_qubits, []), q[:], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        return qc


class CatDisEntInst(Gate):
    """
    CatDisEnt gate for nonlocal quantum operations.
    """

    def __init__(self, label=None):
        super().__init__("CatDisEnt", 2, [], label)

    @property
    def _qasm_definition(self):
        qc = self._qc()
        qubit_to_qasm = {bit: f"p{i}" for i, bit in enumerate(qc.qubits)}
        return (
            "gate "
            + self.name.lower()
            + " "
            + ",".join(qubit_to_qasm[qubit] for qubit in qc.qubits)
            + ";\n"
        )

    def _qc(self):
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(Gate(self.name, self.num_qubits, []), q[:], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        return qc


class TeleportInst(Gate):
    """
    Teleport gate for nonlocal quantum operations.
    """

    def __init__(self, label=None):
        super().__init__("Teleport", 3, [], label)

    @property
    def _qasm_definition(self):
        qc = self._qc()
        qubit_to_qasm = {bit: f"p{i}" for i, bit in enumerate(qc.qubits)}
        return (
            "gate "
            + self.name.lower()
            + " "
            + ",".join(qubit_to_qasm[qubit] for qubit in qc.qubits)
            + ";\n"
        )

    def _qc(self):
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(Gate(self.name, self.num_qubits, []), q[:], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        return qc


class CatEnt(DAGOpNode):
    """
    CatEnt node for nonlocal quantum operations.
    """

    def __init__(self, qargs=(), cargs=()):
        super().__init__(CatEntInst(), qargs=qargs, cargs=cargs)


class CatDisEnt(DAGOpNode):
    """
    CatDisEnt node for nonlocal quantum operations.
    """

    def __init__(self, qargs=(), cargs=(), time=3):
        super().__init__(CatDisEntInst(), qargs=qargs, cargs=cargs)


class Teleport(DAGOpNode):
    """
    Teleport node for nonlocal quantum operations.
    """

    def __init__(self, qargs=(), cargs=()):
        super().__init__(TeleportInst(), qargs=qargs, cargs=cargs)
