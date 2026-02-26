"""Gate grouping module."""

from typing import Dict, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Clbit, Instruction, Qubit

from ..network import Network


class _ControlWithInstr:
    """Class used internally to have a memory connected to a control qubit"""

    def __init__(self, num_qubits: int, qubits_list: Sequence[Qubit]):
        """Initializes the class"""
        self.controlled = False
        self.target = ""
        self.working_memory = []
        self.num_qubits = num_qubits
        self.qubits = qubits_list

    def check_if_controlled(self) -> bool:
        """Checks if the qubit is being used as control"""
        return self.controlled

    def set_control(self, target: str, instruction: CircuitInstruction) -> None:
        """Sets the qubit as controlled"""
        self.target = target
        self.working_memory.append(instruction)
        self.controlled = True

    def add_control(self, instruction: CircuitInstruction) -> None:
        """Adds instruction to a qubit used as control"""
        self.working_memory.append(instruction)

    def get_target(self):
        """Returns the target qubit"""
        return self.target

    def get_control_as_instruction(self) -> Instruction:
        """Returns the gates connected to the qubit as a single instruction"""
        circuit = QuantumCircuit(12)
        for instruction in self.working_memory:
            circuit.append(instruction)
        instruction_output = circuit.to_instruction()
        return instruction_output

    def get_control(self) -> list:
        """Returns the gates connected to the qubit as a list"""
        return self.working_memory

    def reset_control(self) -> None:
        """Resets the qubit to its initial state"""
        self.controlled = False
        self.target = ""
        self.working_memory = []


class _Target:
    """Class used internally to have all the information when a qubit is targeted in one place."""

    def __init__(self):
        """Initializes the class"""
        self.control_reg = ""
        self.control_ind: int = -1
        self.targeted = False

    def check_if_targeted(self) -> bool:
        """Checks if the qubit is being targeted"""
        return self.targeted

    def set_target(self, control_reg: str, control_ind: int):
        """Sets the qubit as being targeted by an input qubit"""
        self.control_reg = control_reg
        self.control_ind = control_ind

    def get_control(self) -> tuple[str, int]:
        """Returns the gate connected to the qubit"""
        return self.control_reg, self.control_ind


class _QPUWithControls:
    """Class used internally to hold all the information about a QPU and its connections"""

    def __init__(
        self,
        id_qpu: str,
        num_qubits: int,
        num_ebits: int,
        connected_qpus: list[str],
        qubits_list: list[int],
    ):
        """Initializes the class"""
        self.num_qubits = num_qubits
        self.qubits_list = qubits_list
        self._id_qpu: str = id_qpu
        self._controlled: dict[int, _ControlWithInstr] = {
            qubit: _ControlWithInstr(num_qubits, qubits_list) for qubit in qubits_list
        }
        self._targeted: dict[int, _Target] = {qubit: _Target() for qubit in qubits_list}
        self._ebits_as_control: list[Optional[str]] = [None] * num_ebits
        self._ebits_as_target: list[Optional[str]] = [None] * num_ebits
        self._connected_qpus: list[str] = connected_qpus

    def control_ebits_used(self) -> int:
        """Returns the ebits currently used as control"""
        return len(self._ebits_as_control) - self._ebits_as_control.count(None)

    def ebits_left(self) -> int:
        """Returns the remaining ebits"""
        return self._ebits_as_control.count(None)

    def _occupy_ebit_as_control(self, target):
        """Occupies an ebit connected to a target qubit"""
        if self.ebits_left():
            ind = self._ebits_as_control.index(None)
            self._ebits_as_control[ind] = target

    def _free_ebit_as_control(self, target):
        """Frees an ebit connected to a target qubit"""
        position = self._ebits_as_control.index(target)
        self._ebits_as_control[position] = None

    def check_if_targeting(self, target):
        """Checks if the ebit is being targeted"""
        return target in self._ebits_as_control

    def qpu_that_targets_this(self):
        """Returns the qpu that targets this one"""
        return self._ebits_as_target[0]

    def find_qubit_targeting(self, reg_target) -> int:
        """Returns the index of the qubit targeted in the register reg_target, or -1 if not found"""
        ind_t = -1
        for ind in self.qubits_list:
            if ind in self._controlled and (
                self._controlled[ind].get_target() == reg_target
            ):
                ind_t = ind
                break
        if ind_t == -1:
            raise ValueError("no qubit targeting found")
        return ind_t

    def is_targeted(self, position) -> bool:
        """Returns true if the qubit in position is being targeted"""
        return self._targeted[position].check_if_targeted()

    def is_targeted_from(self, position) -> tuple[str, int]:
        """Returns the control from which the qubit in position is being targeted"""
        return self._targeted[position].get_control()

    def set_targeted(self, position, control_reg, control_ind):
        """Sets the qubit as being targeted by an input qubit"""
        self._targeted[position].set_target(control_reg, control_ind)

    def is_controlled(self, position):
        """Returns true if the qubit is being used as control"""
        return self._controlled[position].check_if_controlled()

    def get_target(self, position):
        """Returns the target qubit of the qubit in position"""
        return self._controlled[position].get_target()

    def set_controlled(
        self, position: int, target, instruction: CircuitInstruction
    ) -> None:
        """Sets the qubit as control of the input instruction, with relative target"""
        self._controlled[position].set_control(target, instruction)
        self._occupy_ebit_as_control(target)

    def _add_controlled(self, position: int, instruction: CircuitInstruction) -> None:
        """Adds the instruction to the controlled qubit"""
        self._controlled[position].add_control(instruction)

    def _depth_of_circuit_of_controlled(self, position: int) -> int:
        """Returns the depth of the circuit relative to the controlled qubit"""
        return len(self._controlled[position].get_control())

    def get_controlled(self, position) -> list:
        """Returns the instruction relative to the controlled qubit"""
        return self._controlled[position].get_control()

    def reset_controlled(self, position):
        """Resets the controlled qubit"""
        target = self._controlled[position].get_target()
        self._controlled[position].reset_control()
        self._free_ebit_as_control(target)


class GateGrouping(object):
    """This class checks for gates that could share an EPR pair and groups them together"""

    def __init__(
        self,
        circuit: QuantumCircuit,
        nodes_mapping: Dict[Qubit, str],
        network: Network,
        max_gates_in_a_group: int = 0,
        check_swaps: bool = False,
    ):
        """
        Initializes the class with the parameters given

        Args:
            circuit (QuantumCircuit): Quantum circuit to be compiled
            nodes_mapping (dict): Mapping from qubit to nodes
            network (Network): The network configuration
            max_gates_in_a_group (int): The maximum number of gates that can be in a single group
            check_swaps (bool): Whether to check for swaps between gates
        """
        self.qubits_list: list[Qubit] = circuit.qubits
        self.clbits_list: list[Clbit] = circuit.clbits
        self.new_circuit = QuantumCircuit(self.qubits_list, self.clbits_list)
        self.data: list[Optional[CircuitInstruction]] = list(circuit.data) + [None]
        self.network: Network = network
        self.partitions = {node: [] for node in self.network.nodes.keys()}
        self.nodes_mapping = nodes_mapping
        self.current_gate = 0
        self.check_swaps = check_swaps
        self.max_non_local_gates = (
            max_gates_in_a_group if max_gates_in_a_group > 0 else 2**15
        )
        for q, p in self.nodes_mapping.items():
            self.partitions[p].append(q.index)

        self.qpus: dict[str, _QPUWithControls] = {
            node.name: _QPUWithControls(
                node.name,
                node.mem_cap,
                len(node.ebits),
                list(node.connections),
                self.partitions[node.name],
            )
            for node in self.network.nodes.values()
        }

    @staticmethod
    def local_op(qargs: Sequence, nodes_mapping) -> bool:
        """Returns True if the operation is local"""
        # qargs already mapped with mapping
        if len(qargs) == 1:
            return True
        return nodes_mapping[qargs[0]] == nodes_mapping[qargs[1]]

    def run(self) -> QuantumCircuit:
        """
        Returns:
            The circuit with the gates that could share an EPR pair grouped
        """
        while self.current_gate < len(self.data):  # last one is None
            # The idea is to check for every gate if it uses controlled qubits.
            # If it uses one, we need to check that the target is in the same qpu.
            # If both conditions are true, we can add the new gate to the multi-gate instructions.
            # If the gate does something to the control but does not target the same qpu, we must stop.
            # There is a multi-gate instruction for every qpu where the control can be.
            gate = self.data[self.current_gate]
            if gate is None:
                break
            if len(gate.qubits) == 1:
                reg = self.nodes_mapping[gate.qubits[0]]
                ind = gate.qubits[0].index

                if self.qpus[reg].is_controlled(ind):
                    if self.can_commute(gate, self.data[self.current_gate + 1]):
                        self.current_gate += 1
                        continue
                    self.add_to_circuit_and_reset(ind, reg)
                    self.add_current_gate_to_new_circuit()
                elif self.qpus[reg].is_targeted(ind):
                    reg_c, ind_c = self.qpus[reg].is_targeted_from(ind)
                    if gate.operation.name == "measure":
                        self.add_to_circuit_and_reset(ind_c, reg_c)
                        self.add_current_gate_to_new_circuit()
                    else:
                        self.add_gate_in_controlled(ind_c, reg_c)
                        self.reset_if_max_depth(reg_c, ind_c)
                else:
                    self.add_current_gate_to_new_circuit()
            elif self.local_op(gate.qubits, self.nodes_mapping):
                reg = self.nodes_mapping[gate.qubits[0]]
                ind0 = gate.qubits[0].index
                ind1 = gate.qubits[1].index

                match (
                    self.qpus[reg].is_controlled(ind0),
                    self.qpus[reg].is_controlled(ind1),
                    self.qpus[reg].is_targeted(ind0),
                    self.qpus[reg].is_targeted(ind1),
                ):
                    case (True, False, False, False):
                        # only first is controlled
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind0, reg)
                        self.add_current_gate_to_new_circuit()

                    case (False, True, False, False):
                        # only second is controlled
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind1, reg)
                        self.add_current_gate_to_new_circuit()

                    case (True, True, False, False):
                        # both are controlled
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind0, reg)
                        self.add_to_circuit_and_reset(ind1, reg)
                        self.add_current_gate_to_new_circuit()

                    case (True, False, False, True):
                        # first controlled, second targeted
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind0, reg)
                        self.add_gate_in_controlled(ind1, reg)
                        self.reset_if_max_depth(reg, ind1)

                    case (False, True, True, False):
                        # second controlled, first targeted
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind1, reg)
                        self.add_gate_in_controlled(ind0, reg)
                        self.reset_if_max_depth(reg, ind0)

                    case (False, False, True, True):
                        # both are targeted
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        reg_c, ind_c = self.qpus[reg].is_targeted_from(ind1)
                        self.add_to_circuit_and_reset(ind_c, reg_c)
                        reg_c0, ind_c0 = self.qpus[reg].is_targeted_from(ind0)
                        self.add_gate_in_controlled(ind_c0, reg_c0)
                        self.reset_if_max_depth(reg_c0, ind_c0)
                    case (False, False, True, False):
                        # only first is targeted
                        reg_c0, ind_c0 = self.qpus[reg].is_targeted_from(ind0)
                        self.add_gate_in_controlled(ind_c0, reg_c0)
                        self.reset_if_max_depth(reg_c0, ind_c0)
                    case (False, False, False, True):
                        # only second is targeted
                        reg_c1, ind_c1 = self.qpus[reg].is_targeted_from(ind1)
                        self.add_gate_in_controlled(ind_c1, reg_c1)
                        self.reset_if_max_depth(reg_c1, ind_c1)
                    case (False, False, False, False):
                        # both are neither targeted nor controlled
                        self.add_current_gate_to_new_circuit()
            else:  # non-local case
                reg0 = self.nodes_mapping[gate.qubits[0]]
                reg1 = self.nodes_mapping[gate.qubits[1]]
                ind0 = gate.qubits[0].index
                ind1 = gate.qubits[1].index
                if self.qpus[reg0].is_targeted(ind0):
                    if self.can_commute(gate, self.data[self.current_gate + 1]):
                        self.current_gate += 1
                        continue
                    reg_c, ind_c = self.qpus[reg0].is_targeted_from(ind0)
                    self.add_to_circuit_and_reset(ind_c, reg_c)

                if self.qpus[reg0].is_controlled(ind0):
                    if self.qpus[reg0].get_target(ind0) == reg1:
                        self.add_gate_in_controlled(ind0, reg0)
                        if self.qpus[reg1].is_controlled(
                            ind1
                        ):  # check if target is controlled
                            if self.can_commute(gate, self.data[self.current_gate + 1]):
                                self.current_gate += 1
                                continue
                            self.add_to_circuit_and_reset(ind1, reg1)
                            self.qpus[reg1].set_targeted(ind1, reg0, ind0)

                        if self.qpus[reg1].is_targeted(
                            ind1
                        ):  # check if target is targeted
                            if self.qpus[reg1].is_targeted_from(ind1) == (reg0, ind0):
                                continue  # everything has already been done
                            else:
                                if self.can_commute(
                                    gate, self.data[self.current_gate + 1]
                                ):
                                    self.current_gate += 1
                                    continue
                                reg2, ind2 = self.qpus[reg1].is_targeted_from(ind1)
                                self.add_to_circuit_and_reset(ind2, reg2)
                                self.qpus[reg1].set_targeted(ind1, reg0, ind0)
                        else:
                            self.qpus[reg1].set_targeted(ind1, reg0, ind0)
                        self.reset_if_max_depth(reg0, ind0)

                    else:
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        self.add_to_circuit_and_reset(ind0, reg0)
                        self.set_controlled_with_gate(ind0, reg1, reg0)
                        if self.qpus[reg1].is_targeted(ind1):
                            reg2, ind2 = self.qpus[reg1].is_targeted_from(ind1)
                            self.new_circuit.append(
                                self.as_instruction(
                                    self.qpus[reg2].get_controlled(ind2)
                                )
                            )
                            self.qpus[reg2].reset_controlled(ind2)
                            self.qpus[reg1].set_targeted(ind1, reg0, ind0)

                else:
                    if self.qpus[reg0].check_if_targeting(
                        reg1
                    ):  # if we're targeting a qpu we're already targeting
                        if self.can_commute(gate, self.data[self.current_gate + 1]):
                            self.current_gate += 1
                            continue
                        ind_t: int = self.qpus[reg0].find_qubit_targeting(reg1)
                        self.add_to_circuit_and_reset(ind_t, reg0)
                        self.set_controlled_with_gate(ind0, reg1, reg0)
                        self.qpus[reg1].set_targeted(ind1, reg0, ind0)

                    elif self.qpus[
                        reg0
                    ].ebits_left():  # if we're not targeting the same qpu, but we still have ebits left
                        self.set_controlled_with_gate(ind0, reg0, reg0)
                    else:  # if we do not have ebits left we must try to free them
                        if self.qpus[reg0].control_ebits_used() > 0:
                            for ind in self.qpus[reg0].qubits_list:
                                if self.qpus[reg0].is_controlled(ind):
                                    if self.can_commute(
                                        gate, self.data[self.current_gate + 1]
                                    ):
                                        self.current_gate += 1
                                        continue
                                    ind_c: int = ind
                                    self.add_to_circuit_and_reset(ind_c, reg0)
                                    self.set_controlled_with_gate(ind0, reg1, reg0)
                                    self.qpus[reg1].set_targeted(ind1, reg0, ind0)
                                    break
                        else:  # if we have all ebits occupied as target
                            if self.can_commute(gate, self.data[self.current_gate + 1]):
                                self.current_gate += 1
                                continue
                            reg_c: str = self.qpus[reg0].qpu_that_targets_this()
                            ind_c: int = self.qpus[reg_c].find_qubit_targeting(reg0)
                            self.add_to_circuit_and_reset(ind_c, reg_c)
                            self.set_controlled_with_gate(ind0, reg1, reg0)
                            self.qpus[reg1].set_targeted(ind1, reg0, ind0)
        for qpu in self.qpus.values():
            # final cleanup
            for qubit in qpu.qubits_list:
                if qpu.is_controlled(qubit):
                    self.new_circuit.append(
                        self.as_instruction(qpu.get_controlled(qubit)),
                        qargs=self.qubits_list,
                    )
                    qpu.reset_controlled(qubit)

        return self.new_circuit

    def add_gate_in_controlled(self, ind0: int, reg0: str):
        """
        Adds to the list of gates in the working memory that are using the given qubit as control

        Args:
            ind0 (int): the index of the qubit
            reg0 (str): the register of the qubit
        """
        self.qpus[reg0]._add_controlled(ind0, self.data.pop(self.current_gate))
        self.current_gate = 0

    def set_controlled_with_gate(self, ind0: int, reg1: str, reg0: str):
        """
        Sets the current gate in the working memory as using the given qubit as control

        Args:
            ind0 (int): the index of the controlled
            reg0 (str): the register of the controlled qubit
            reg1 (str): the register of the target qubit
        """
        self.qpus[reg0].set_controlled(ind0, reg1, self.data.pop(self.current_gate))
        self.current_gate = 0

    def add_current_gate_to_new_circuit(self):
        """Adds current gate to the new circuit"""
        self.new_circuit.append(self.data.pop(self.current_gate))
        self.current_gate = 0

    def as_instruction(self, working_memory: list[Instruction]) -> Instruction:
        """
        Transforms a list of instructions into a single Instruction

        Args:
            working_memory (list[Instruction]): The list of instructions to be transformed

        Returns:
            The transformed instruction
        """
        circuit = QuantumCircuit(self.qubits_list)
        for instruction in working_memory:
            circuit.append(instruction)
        instruction_output = circuit.to_instruction()
        return instruction_output

    def add_to_circuit_and_reset(self, ind: int, reg: str):
        """
        Adds the working memory connected to the given qubit to the circuit, and then resets it

        Args:
            ind (int): the index of the qubit
            reg (str): the register of the qubit
        """
        self.new_circuit.append(
            self.as_instruction(self.qpus[reg].get_controlled(ind)),
            qargs=self.qubits_list,
        )
        self.qpus[reg].reset_controlled(ind)

    def reset_if_max_depth(self, reg: str, ind: int):
        """
        Checks if the working memory is at max depth: if it is, it resets it

        Args:
            reg (str): the register of the qubit
            ind (int): the index of the qubit
        """
        if (
            self.qpus[reg]._depth_of_circuit_of_controlled(ind)
            >= self.max_non_local_gates
        ):
            self.add_to_circuit_and_reset(ind, reg)

    def can_commute(self, gate1: CircuitInstruction, gate2: CircuitInstruction) -> bool:
        """
        Checks if the given gates are swappable

        Args:
            gate1 (CircuitInstruction): the first gate
            gate2 (CircuitInstruction): the second gate
        Returns:
            True if the gates are swappable, False otherwise
        """
        # gate set: x, rx, z, rz, h, cz
        if not self.check_swaps:
            return False
        if gate1 is None or gate2 is None:
            return False
        if gate1.qubits[0] not in gate2.qubits:
            if len(gate1.qubits) == 1 or gate1.qubits[1] not in gate2.qubits:
                return True
        if gate1.operation.name in {"rz", "z"}:
            return gate2.operation.name in {"rz", "z", "cz"}
        elif gate1.operation.name == "cz":
            return gate2.operation.name in {"rz", "z", "cz"}
        else:
            return False
