"""QuantumCircuit to QoalaPrograms parser call module."""

from netqasm.lang.encoding import RegisterName
from netqasm.lang.instr.core import (
    BezInstruction,
    InitInstruction,
    JmpInstruction,
    LoadInstruction,
    MeasInstruction,
    SetInstruction,
    StoreInstruction,
)
from netqasm.lang.instr.vanilla import (
    CnotInstruction,
    CphaseInstruction,
    GateHInstruction,
    GateXInstruction,
    GateYInstruction,
    GateZInstruction,
    RotXInstruction,
    RotYInstruction,
    RotZInstruction,
)
from netqasm.lang.operand import Address, ArrayEntry, Immediate, Register, Template
from netqasm.lang.subroutine import Subroutine
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qoala.lang.hostlang import (
    AddCValueOp,
    AssignCValueOp,
    BasicBlock,
    BasicBlockType,
    BranchIfEqualOp,
    IqoalaSingleton,
    IqoalaTuple,
    ReceiveCMsgOp,
    ReturnResultOp,
    RunRequestOp,
    RunSubroutineOp,
    SendCMsgOp,
)
from qoala.lang.program import ProgramMeta, QoalaProgram
from qoala.lang.request import (
    CallbackType,
    EprRole,
    EprType,
    QoalaRequest,
    RequestRoutine,
    RequestVirtIdMapping,
)
from qoala.lang.routine import LocalRoutine, RoutineMetadata

from ..network import Network


class InstrToBlock:
    """The QuantumCircuit to QoalaPrograms parser class."""

    def __init__(
        self,
        compiled_circuit: QuantumCircuit,
        network_to_local: dict,
        network_config: Network,
        layout: Layout,
    ):
        """
        Initialize the parser

        Args:
            compiled_circuit: a distributed circuit
            network_to_local: a mapping from the circuit virtual qubits to physical qubits on network nodes
            network_config: the network configuration
            layout: the mapping of qubits
        """

        self.operations = []
        self.qubits_list = []
        self.layout = layout
        for inst in compiled_circuit.data:
            qubits = [network_to_local[qubit] for qubit in inst.qubits]
            self.operations.append(inst.operation)
            self.qubits_list.append(qubits)
        self.network_config: Network = network_config
        self.block_i: dict[str, int] = {node: 0 for node in self.network_config.nodes}
        self.nodes_with_blocks: dict[
            str, list[list[BasicBlock | LocalRoutine | RequestRoutine]]
        ] = {}
        self.meta, self.sockets = self._generate_meta()
        self.measured: list[tuple[str, int]] = []
        for node_name, node in self.network_config.nodes.items():
            self.nodes_with_blocks[node_name] = [[], [], []]

    def run(self) -> tuple[dict[str, QoalaProgram], list[tuple[str, int]]]:
        """
        Run the paser to parse a distributed DAGCircuit into QoalaPrograms

        Returns:
            the QoalaPrograms and a mapping of the measured qubits
        """
        self._prepare_qubits()
        self._from_instr_to_block()
        return self.from_instr_to_qoala()

    def _generate_meta(
        self,
    ) -> tuple[dict[str, ProgramMeta], dict[str, dict[str, int]]]:
        """
        Generate the metadata of a Qoala program

        Returns:
            the meta for each node and sockets info
        """
        metas: dict[str, ProgramMeta] = {}
        distances: dict = self.network_config.distances
        sockets: dict[str, dict[str, int]] = {}
        # look for nodes that have distance == 1
        for node, dists in distances.items():
            node_name = f"node{node}"
            sockets[node] = {}
            i = 0
            metas[node_name] = ProgramMeta(
                name="", parameters=[], csockets={}, epr_sockets={}
            )
            metas[node_name].name = node_name
            for connecting_node, dist in dists.items():
                if dist == 1:
                    metas[node_name].parameters.append(f"node{connecting_node}_id")
                    metas[node_name].epr_sockets[i] = f"node{connecting_node}"
                    metas[node_name].csockets[i] = f"node{connecting_node}"
                    sockets[node][connecting_node] = i
                    i += 1
        for node0 in self.network_config.nodes:
            for node1 in self.network_config.nodes:
                if (
                    node0 != node1
                    and f"node{node1}_id" not in metas[f"node{node0}"].parameters
                ):
                    metas[f"node{node0}"].parameters.append(f"node{node1}_id")
                    metas[f"node{node0}"].csockets[
                        len(metas[f"node{node0}"].csockets)
                    ] = f"node{node1}"
                    sockets[node0][node1] = len(metas[f"node{node0}"].csockets) - 1
        return metas, sockets

    def _prepare_qubits(self):
        """
        Create the routines to initialize qubits
        """
        for node_name, node in self.network_config.nodes.items():
            prepare_instructions: list[SetInstruction | InitInstruction] = []
            qbits_used_and_kept: list[int] = []
            for i in node.qubits:
                prepare_instruction = [
                    SetInstruction(reg=Register(RegisterName.Q, i), imm=Immediate(i)),
                    InitInstruction(reg=Register(RegisterName.Q, i)),
                ]
                prepare_instructions += prepare_instruction
                qbits_used_and_kept.append(i)
            prepare_subroutine = Subroutine(instructions=prepare_instructions)
            routine_name = f"prepare_qubits_routine{self.block_i[node_name]}"
            prepare_routine = LocalRoutine(
                name=routine_name,
                subroutine=prepare_subroutine,
                return_vars=[],
                metadata=RoutineMetadata(qbits_used_and_kept, qbits_used_and_kept),
            )
            ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
            block = BasicBlock(
                "b" + str(self.block_i[node_name]), BasicBlockType.QL, ops
            )
            self.nodes_with_blocks[node_name][0].append(block)
            self.nodes_with_blocks[node_name][1].append(prepare_routine)
            self.block_i[node_name] += 1

    def _from_instr_to_block(self):
        """
        Iterate over a list of gates to convert them into appropriate qoala routines
        """
        for operation, qubits in zip(self.operations, self.qubits_list):
            match operation.name:
                case "barrier":
                    continue
                case "h":
                    qubit: int = qubits[0][1]
                    node: str = qubits[0][0]
                    routine_name = f"h_routine{self.block_i[node]}"
                    h_routine = self.h_subroutine(routine_name, qubit)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(h_routine)
                    self.block_i[node] += 1
                case "x":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"x_routine{self.block_i[node]}"
                    x_routine = self.x_subroutine(routine_name, qubit)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(x_routine)
                    self.block_i[node] += 1
                case "y":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"y_routine{self.block_i[node]}"
                    y_routine = self.y_subroutine(routine_name, qubit)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(y_routine)
                    self.block_i[node] += 1
                case "z":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"z_routine{self.block_i[node]}"
                    z_routine = self.z_subroutine(routine_name, qubit)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(z_routine)
                    self.block_i[node] += 1
                case "rx":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"rx_routine{self.block_i[node]}"
                    rx_routine = self.rx_subroutine(
                        routine_name,
                        qubit,
                        operation.params["n"],
                        operation.params["d"],
                    )
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(rx_routine)
                    self.block_i[node] += 1
                case "ry":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"ry_routine{self.block_i[node]}"
                    ry_routine = self.ry_subroutine(
                        routine_name,
                        qubit,
                        operation.params["n"],
                        operation.params["d"],
                    )
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(ry_routine)
                    self.block_i[node] += 1
                case "rz":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"rz_routine{self.block_i[node]}"
                    rz_routine = self.rz_subroutine(
                        routine_name,
                        qubit,
                        operation.params["n"],
                        operation.params["d"],
                    )
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(rz_routine)
                    self.block_i[node] += 1
                case "cx":
                    qubit0 = qubits[0][1]
                    qubit1 = qubits[1][1]
                    node = qubits[0][0]
                    routine_name = f"cnot_routine{self.block_i[node]}"
                    cnot_routine = self.cx_subroutine(routine_name, qubit0, qubit1)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(cnot_routine)
                    self.block_i[node] += 1
                case "cz":
                    qubit0 = qubits[0][1]
                    qubit1 = qubits[1][1]
                    node = qubits[0][0]
                    routine_name = f"cz_routine{self.block_i[node]}"
                    cz_routine = self.cz_subroutine(routine_name, qubit0, qubit1)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(cz_routine)
                    self.block_i[node] += 1
                case "swap":
                    qubit0 = qubits[0][1]
                    qubit1 = qubits[1][1]
                    node = qubits[0][0]
                    routine_name = f"swap_routine{self.block_i[node]}"
                    swap_routine = self.swap_subroutine(routine_name, qubit0, qubit1)
                    ops = [RunSubroutineOp(None, IqoalaTuple([]), routine_name)]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(swap_routine)
                    self.block_i[node] += 1
                case "measure":
                    qubit = qubits[0][1]
                    node = qubits[0][0]
                    routine_name = f"measure_routine{self.block_i[node]}"
                    measure_routine = self.measure_subroutine(routine_name, qubit)
                    self.measured.append((node, qubit))
                    ops = [
                        RunSubroutineOp(
                            IqoalaTuple([f"r{qubit}"]), IqoalaTuple([]), routine_name
                        )
                    ]
                    block = BasicBlock(
                        "b" + str(self.block_i[node]), BasicBlockType.QL, ops
                    )
                    self.nodes_with_blocks[node][0].append(block)
                    self.nodes_with_blocks[node][1].append(measure_routine)
                    self.block_i[node] += 1
                case "catent":
                    qubit = qubits[0][1]
                    ebits = [q[1] for q in qubits[1:]]
                    nodes = [q[0] for q in qubits]

                    for i in range(1, len(ebits), 2):
                        sender_id = nodes[i]
                        receiver_id = nodes[i + 1]

                        # Generate EPR pairs between nodes
                        request_sender, request_receiver = self.generic_epr(
                            sender_id, receiver_id, ebits[i - 1], ebits[i]
                        )

                        op_request_create = [
                            RunRequestOp(None, IqoalaTuple([]), request_sender.name)
                        ]
                        op_request_receive = [
                            RunRequestOp(None, IqoalaTuple([]), request_receiver.name)
                        ]

                        block_create = BasicBlock(
                            "b" + str(self.block_i[sender_id]),
                            BasicBlockType.QC,
                            op_request_create,
                        )
                        block_receive = BasicBlock(
                            "b" + str(self.block_i[receiver_id]),
                            BasicBlockType.QC,
                            op_request_receive,
                        )

                        self.nodes_with_blocks[sender_id][0] += [block_create]
                        self.block_i[sender_id] += 1
                        self.nodes_with_blocks[receiver_id][0] += [block_receive]
                        self.block_i[receiver_id] += 1
                        self.nodes_with_blocks[sender_id][2] += [request_sender]
                        self.nodes_with_blocks[receiver_id][2] += [request_receiver]

                    # Execute entanglement swapping protocol if needed
                    if len(ebits) > 2:
                        self.ent_swap(nodes, ebits)

                    # Execute CatEntanglement protocol

                    # First node sends correction
                    catent_routine_sender = self.catent_routine_sender(
                        routine_name=f"catent_sender_routine{self.block_i[nodes[0]]}",
                        qubit=qubit,
                        ebit=ebits[0],
                    )
                    ops_s = [
                        RunSubroutineOp(
                            IqoalaTuple(["m0"]),
                            IqoalaTuple([]),
                            f"catent_sender_routine{self.block_i[nodes[0]]}",
                        )
                    ]
                    block_local_sender = BasicBlock(
                        "b" + str(self.block_i[nodes[0]]), BasicBlockType.QL, ops_s
                    )
                    self.block_i[nodes[0]] += 1

                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[nodes[0]][nodes[-1]],
                    )
                    send_msg = [
                        SendCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            value=IqoalaSingleton(name="m0"),
                        )
                    ]

                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[nodes[0]]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[nodes[0]] += 1
                    block_send_msg = BasicBlock(
                        "b" + str(self.block_i[nodes[0]]), BasicBlockType.CL, send_msg
                    )
                    self.block_i[nodes[0]] += 1
                    self.nodes_with_blocks[nodes[0]][0] += [
                        block_local_sender,
                        block_assign_csocket,
                        block_send_msg,
                    ]
                    self.nodes_with_blocks[nodes[0]][1] += [catent_routine_sender]

                    # Second node receives and performs correction
                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[nodes[-1]][nodes[0]],
                    )
                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[nodes[-1]]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[nodes[-1]] += 1
                    receive_msg = [
                        ReceiveCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            result=IqoalaSingleton(name="p0"),
                        )
                    ]
                    block_receive_msg = BasicBlock(
                        "b" + str(self.block_i[nodes[-1]]),
                        BasicBlockType.CC,
                        receive_msg,
                    )
                    self.block_i[nodes[-1]] += 1

                    catent_routine_receiver = self.catent_routine_receiver(
                        f"catent_receiver_routine{self.block_i[nodes[-1]]}", ebits[-1]
                    )
                    ops_r = [
                        RunSubroutineOp(
                            None,
                            IqoalaTuple(["p0"]),
                            f"catent_receiver_routine{self.block_i[nodes[-1]]}",
                        )
                    ]
                    block_local_receiver = BasicBlock(
                        "b" + str(self.block_i[nodes[-1]]), BasicBlockType.QL, ops_r
                    )
                    self.block_i[nodes[-1]] += 1

                    self.nodes_with_blocks[nodes[-1]][0] += [
                        block_assign_csocket,
                        block_receive_msg,
                        block_local_receiver,
                    ]
                    self.nodes_with_blocks[nodes[-1]][1] += [catent_routine_receiver]

                case "catdisent":
                    ebit = qubits[0][1]
                    qubit = qubits[1][1]
                    receiver_id = qubits[1][0]
                    sender_id = qubits[0][0]

                    # First node send corrections
                    catdisent_routine_sender = self.catdisent_routine_sender(
                        routine_name=f"catdisent_sender_routine{self.block_i[sender_id]}",
                        ebit=ebit,
                    )
                    ops_s = [
                        RunSubroutineOp(
                            IqoalaTuple(["m0"]),
                            IqoalaTuple([]),
                            f"catdisent_sender_routine{self.block_i[sender_id]}",
                        )
                    ]
                    block_local_sender = BasicBlock(
                        "b" + str(self.block_i[sender_id]), BasicBlockType.QL, ops_s
                    )
                    self.block_i[sender_id] += 1

                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[sender_id][receiver_id],
                    )
                    send_msg = [
                        SendCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            value=IqoalaSingleton(name="m0"),
                        )
                    ]

                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[sender_id]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[sender_id] += 1
                    block_send_msg = BasicBlock(
                        "b" + str(self.block_i[sender_id]), BasicBlockType.CL, send_msg
                    )
                    self.block_i[sender_id] += 1
                    self.nodes_with_blocks[sender_id][0] += [
                        block_local_sender,
                        block_assign_csocket,
                        block_send_msg,
                    ]
                    self.nodes_with_blocks[sender_id][1] += [catdisent_routine_sender]

                    # Second node receives and perform corrections
                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[receiver_id][sender_id],
                    )
                    recv_msg = [
                        ReceiveCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            result=IqoalaSingleton(name="p0"),
                        )
                    ]

                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[receiver_id]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[receiver_id] += 1
                    block_recv_msg = BasicBlock(
                        "b" + str(self.block_i[receiver_id]),
                        BasicBlockType.CC,
                        recv_msg,
                    )
                    self.block_i[receiver_id] += 1

                    catdisent_routine_receiver = self.catdisent_routine_receiver(
                        routine_name=f"catdisent_receiver_routine{self.block_i[receiver_id]}",
                        qubit=qubit,
                    )
                    ops_s = [
                        RunSubroutineOp(
                            None,
                            IqoalaTuple(["p0"]),
                            f"catdisent_receiver_routine{self.block_i[receiver_id]}",
                        )
                    ]
                    block_local_receiver = BasicBlock(
                        "b" + str(self.block_i[receiver_id]), BasicBlockType.QL, ops_s
                    )
                    self.block_i[receiver_id] += 1

                    self.nodes_with_blocks[receiver_id][0] += [
                        block_assign_csocket,
                        block_recv_msg,
                        block_local_receiver,
                    ]
                    self.nodes_with_blocks[receiver_id][1] += [
                        catdisent_routine_receiver
                    ]

                case "teleport":
                    qubit0 = qubits[0][1]
                    ebit0 = qubits[1][1]
                    ebit1 = qubits[2][1]
                    qubit1 = qubits[3][1]
                    sender_id = qubits[0][0]
                    receiver_id = qubits[-1][0]

                    # Create EPR pair
                    request_sender, request_receiver = self.generic_epr(
                        sender_id, receiver_id, ebit0, ebit1
                    )

                    op_request_create = [
                        RunRequestOp(
                            IqoalaTuple([]), IqoalaTuple([]), request_sender.name
                        )
                    ]
                    op_request_receiver = [
                        RunRequestOp(
                            IqoalaTuple([]), IqoalaTuple([]), request_receiver.name
                        )
                    ]

                    block_create = BasicBlock(
                        "b" + str(self.block_i), BasicBlockType.QC, op_request_create
                    )
                    self.block_i[sender_id] += 1
                    block_receive = BasicBlock(
                        "b" + str(self.block_i), BasicBlockType.QC, op_request_receiver
                    )
                    self.block_i[receiver_id] += 1

                    # First node sends corrections
                    teleport_routine_sender = self.teledata_routine_sender(
                        f"teleport_sender_routine{self.block_i[sender_id]}",
                        qubit0,
                        ebit0,
                    )
                    ops_s = [
                        RunSubroutineOp(
                            IqoalaTuple(["m0", "m1"]),
                            IqoalaTuple([]),
                            f"teleport_sender_routine{self.block_i[sender_id]}",
                        )
                    ]

                    block_local_sender = BasicBlock(
                        "b" + str(self.block_i[sender_id]), BasicBlockType.QL, ops_s
                    )
                    self.block_i[sender_id] += 1

                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[sender_id][receiver_id],
                    )
                    send_msg0 = [
                        SendCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            value=IqoalaSingleton(name="m0"),
                        )
                    ]
                    send_msg1 = [
                        SendCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            value=IqoalaSingleton(name="m1"),
                        )
                    ]

                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[sender_id]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[sender_id] += 1
                    block_send_msg0 = BasicBlock(
                        "b" + str(self.block_i[sender_id]), BasicBlockType.CL, send_msg0
                    )
                    self.block_i[sender_id] += 1
                    block_send_msg1 = BasicBlock(
                        "b" + str(self.block_i[sender_id]), BasicBlockType.CL, send_msg1
                    )
                    self.block_i[sender_id] += 1

                    self.nodes_with_blocks[sender_id][0] += [
                        block_create,
                        block_local_sender,
                        block_assign_csocket,
                        block_send_msg0,
                        block_send_msg1,
                    ]

                    # Second node receives and perform corrections
                    assign_csocket = AssignCValueOp(
                        result=IqoalaSingleton(name="csocket"),
                        value=self.sockets[receiver_id][sender_id],
                    )
                    recv_msg0 = [
                        ReceiveCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            result=IqoalaSingleton(name="p0"),
                        )
                    ]
                    recv_msg1 = [
                        ReceiveCMsgOp(
                            csocket=IqoalaSingleton(name="csocket"),
                            result=IqoalaSingleton(name="p1"),
                        )
                    ]

                    block_assign_csocket = BasicBlock(
                        "b" + str(self.block_i[receiver_id]),
                        BasicBlockType.CL,
                        [assign_csocket],
                    )
                    self.block_i[receiver_id] += 1
                    block_recv_msg0 = BasicBlock(
                        "b" + str(self.block_i[receiver_id]),
                        BasicBlockType.CC,
                        recv_msg0,
                    )
                    self.block_i[receiver_id] += 1
                    block_recv_msg1 = BasicBlock(
                        "b" + str(self.block_i[receiver_id]),
                        BasicBlockType.CC,
                        recv_msg1,
                    )
                    self.block_i[receiver_id] += 1

                    teleport_routine_receiver = self.teledata_routine_receiver(
                        f"teleport_receiver_routine{self.block_i[receiver_id]}",
                        ebit1,
                        qubit1,
                    )
                    ops_r = [
                        RunSubroutineOp(
                            None,
                            IqoalaTuple(["p0", "p1"]),
                            f"teleport_receiver_routine{self.block_i[receiver_id]}",
                        )
                    ]

                    block_local_receiver = BasicBlock(
                        "b" + str(self.block_i[receiver_id]), BasicBlockType.QL, ops_r
                    )
                    self.block_i[receiver_id] += 1

                    self.nodes_with_blocks[receiver_id][0] += [
                        block_receive,
                        block_assign_csocket,
                        block_recv_msg0,
                        block_recv_msg1,
                        block_local_receiver,
                    ]

                    self.nodes_with_blocks[sender_id][1] += [teleport_routine_sender]
                    self.nodes_with_blocks[receiver_id][1] += [
                        teleport_routine_receiver
                    ]

                    self.nodes_with_blocks[sender_id][2] += [request_sender]
                    self.nodes_with_blocks[receiver_id][2] += [request_receiver]

        if len(self.measured) != 0:
            returned = {}
            for node, measrued in self.measured:
                if node not in returned:
                    returned[node] = []
                returned[node].append(ReturnResultOp(IqoalaSingleton(f"r{measrued}")))

            for node in returned:
                block = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, returned[node]
                )
                self.nodes_with_blocks[node][0].append(block)
                self.block_i[node] += 1

    def from_instr_to_qoala(
        self,
    ) -> tuple[dict[str, QoalaProgram], list[tuple[str, int]]]:
        """
        Creates QoalaPrograms, one for each node, starting from a distributed circuit

        Returns:
            the QoalaPrograms and a mapping of the measured qubits
        """
        output_programs = {}
        for node_name, m in self.meta.items():
            blocks = self.nodes_with_blocks[node_name[4:]][0]
            routines = self.nodes_with_blocks[node_name[4:]][1]
            requests = self.nodes_with_blocks[node_name[4:]][2]
            output_programs[node_name] = QoalaProgram(
                m,
                blocks,
                {routine.name: routine for routine in routines},
                {request.name: request for request in requests},
            )

        return output_programs, self.measured

    def generic_epr(
        self, id_sender: str, id_receiver: str, ebit_sender: int, ebit_receiver: int
    ) -> tuple[RequestRoutine, RequestRoutine]:
        """
        Create the epr pair generation requests

        Args:
            id_sender: the sender id
            id_receiver: the receiver id
            ebit_sender: the sender bit
            ebit_receiver: the receiver ebit

        Returns:
            the qoala request routines
        """
        # TODO fidelity should be more realistic
        socket_sender = self.sockets[id_sender][id_receiver]
        socket_receiver = self.sockets[id_receiver][id_sender]
        req_sender = QoalaRequest(
            remote_id=Template(name=f"node{id_receiver}_id"),
            epr_socket_id=socket_sender,
            num_pairs=1,
            virt_ids=RequestVirtIdMapping.from_str(f"all {ebit_sender}"),
            timeout=1000,
            fidelity=1.0,
            typ=EprType.CREATE_KEEP,
            role=EprRole.CREATE,
            name=f"req_create{self.block_i[id_sender]}",
        )
        req_receiver = QoalaRequest(
            remote_id=Template(name=f"node{id_sender}_id"),
            epr_socket_id=socket_receiver,
            num_pairs=1,
            virt_ids=RequestVirtIdMapping.from_str(f"all {ebit_receiver}"),
            timeout=1000,
            fidelity=1.0,
            typ=EprType.CREATE_KEEP,
            role=EprRole.RECEIVE,
            name=f"req_receive{self.block_i[id_receiver]}",
        )
        return (
            RequestRoutine(
                name=f"req_create{self.block_i[id_sender]}",
                request=req_sender,
                return_vars=[],
                callback_type=CallbackType.WAIT_ALL,
                callback=None,
            ),
            RequestRoutine(
                name=f"req_receive{self.block_i[id_receiver]}",
                request=req_receiver,
                return_vars=[],
                callback_type=CallbackType.WAIT_ALL,
                callback=None,
            ),
        )

    @staticmethod
    def h_subroutine(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the h gate routine
        Args:
            routine_name: the routine name
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        h_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            GateHInstruction(reg=Register(RegisterName.Q, 0)),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(instructions=h_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def x_subroutine(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the x gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        x_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            GateXInstruction(reg=Register(RegisterName.Q, 0)),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=x_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def y_subroutine(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the y gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        y_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            GateYInstruction(reg=Register(RegisterName.Q, 0)),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=y_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def z_subroutine(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the z gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        z_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            GateZInstruction(reg=Register(RegisterName.Q, 0)),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=z_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def rx_subroutine(routine_name: str, qubit: int, n: int, d: int) -> LocalRoutine:
        """
        Create the rx gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used
            n: the exponent of pi
            d: the denominator of the pi fraction

        Returns:
            the qoala local routine
        """
        rx_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            RotXInstruction(
                reg=Register(RegisterName.Q, 0), imm0=Immediate(n), imm1=Immediate(d)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=rx_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def ry_subroutine(routine_name: str, qubit: int, n: int, d: int) -> LocalRoutine:
        """
        Create the ry gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used
            n: the exponent of pi
            d: the denominator of the pi fraction

        Returns:
            the qoala local routine
        """
        ry_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            RotYInstruction(
                reg=Register(RegisterName.Q, 0), imm0=Immediate(n), imm1=Immediate(d)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=ry_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def rz_subroutine(routine_name: str, qubit: int, n: int, d: int) -> LocalRoutine:
        """
        Create the rz gate routine

        Args:
            routine_name: the routine name
            qubit: the qubit used
            n: the exponent of pi
            d: the denominator of the pi fraction

        Returns:
            the qoala local routine
        """
        rz_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            RotZInstruction(
                reg=Register(RegisterName.Q, 0), imm0=Immediate(n), imm1=Immediate(d)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=rz_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def cx_subroutine(routine_name: str, control: int, target: int) -> LocalRoutine:
        """
        Create the cx gate local routine

        Args:
            routine_name: the routine name
            control: the control qubit
            target: the target qubit

        Returns:
            the qoala local routine
        """
        cx_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(control)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(target)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=cx_instructions, arguments=[]),
            [],
            RoutineMetadata([control, target], [control, target]),
        )

    @staticmethod
    def cz_subroutine(routine_name: str, qubit0: int, qubit1: int) -> LocalRoutine:
        """
        Create the cz gate routine

        Args:
            routine_name: the routine name
            qubit0: the first qubit
            qubit1: the second qubit

        Returns:
            the qoala local routine
        """
        cz_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit0)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(qubit1)),
            CphaseInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=cz_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit0, qubit1], [qubit0, qubit1]),
        )

    @staticmethod
    def swap_subroutine(routine_name: str, qubit0: int, qubit1: int) -> LocalRoutine:
        """
        Create the swap routine

        Args:
            routine_name: the routine name
            qubit0: the first qubit
            qubit1: the second qubit

        Returns:
            the qoala local routine
        """
        swap_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit0)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(qubit1)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 1), reg1=Register(RegisterName.Q, 0)
            ),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=swap_instructions, arguments=[]),
            [],
            RoutineMetadata([qubit0, qubit1], [qubit0, qubit1]),
        )

    @staticmethod
    def measure_subroutine(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the measurement routine

        Args:
            routine_name: the routine name
            qubit: the qubit measured

        Returns:
            the qoala local routine
        """
        measure_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.M, 0)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 0),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 0)),
            ),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(instructions=measure_instructions, arguments=[]),
            [f"m{qubit}"],
            RoutineMetadata([qubit], []),
        )

    @staticmethod
    def bsm_subroutine(routine_name: str, qubit0: int, qubit1: int) -> LocalRoutine:
        """
        Create the bell state measurement routine

        Args:
            routine_name: the routine name
            qubit0: the first qubit
            qubit1: the second qubit

        Returns:
            the qoala local routine
        """
        bsm_instructions = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit0)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(qubit1)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
            GateHInstruction(reg=Register(RegisterName.Q, 0)),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.M, 0)
            ),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 1), reg1=Register(RegisterName.M, 1)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 0),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 0)),
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 1),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 1)),
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=bsm_instructions, arguments=[]),
            ["m0", "m1"],
            RoutineMetadata([qubit0, qubit1], []),
        )

    @staticmethod
    def catent_routine_sender(routine_name: str, qubit: int, ebit: int) -> LocalRoutine:
        """
        Create the cat-ent routine for the sender node

        Args:
            routine_name: the routine name
            qubit: the qubit used
            ebit: the ebit used

        Returns:
            the qoala local routine
        """
        catent_instructions_sender = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(ebit)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 1), reg1=Register(RegisterName.M, 0)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 0),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 0)),
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=catent_instructions_sender, arguments=[]),
            ["m0"],
            RoutineMetadata([qubit, ebit], [qubit]),
        )

    @staticmethod
    def catent_routine_receiver(routine_name: str, ebit: int) -> LocalRoutine:
        """
        Create the cat-ent routine for the receiver node

        Args:
            routine_name: the routine name
            ebit: the ebit used

        Returns:
            the qoala local routine
        """
        catent_instructions_receiver = [
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(0)),
            LoadInstruction(
                reg=Register(RegisterName.C, 0),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 0)),
            ),
            # SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebit)),
            BezInstruction(reg=Register(RegisterName.C, 0), imm=Immediate(3)),
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebit)),
            GateXInstruction(reg=Register(RegisterName.Q, 0)),
            JmpInstruction(imm=Immediate(1)),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(instructions=catent_instructions_receiver, arguments=["p0"]),
            [],
            RoutineMetadata([ebit], [ebit]),
        )

    @staticmethod
    def catdisent_routine_sender(routine_name: str, ebit: int) -> LocalRoutine:
        """
        Create the cat-disent routine for the sender node, i.e. the receiver node of a previous cat-ent routine

        Args:
            routine_name: the routine name
            ebit: the ebit used

        Returns:
            the qoala local routine
        """
        catdisent_instructions_sender = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebit)),
            GateHInstruction(reg=Register(RegisterName.Q, 0)),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.M, 0)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 0),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 0)),
            ),
        ]
        return LocalRoutine(
            routine_name,
            Subroutine(instructions=catdisent_instructions_sender, arguments=[]),
            ["m0"],
            RoutineMetadata([ebit], []),
        )

    @staticmethod
    def catdisent_routine_receiver(routine_name: str, qubit: int) -> LocalRoutine:
        """
        Create the cat-disent routine for the receiver node, i.e. the sender node of a previous cat-ent routine

        Args:
            routine_name: the routine name
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        catdisent_instructions_receiver = [
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(0)),
            LoadInstruction(
                reg=Register(RegisterName.C, 0),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 0)),
            ),
            # SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            BezInstruction(reg=Register(RegisterName.C, 0), imm=Immediate(3)),
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            GateZInstruction(reg=Register(RegisterName.Q, 0)),
            JmpInstruction(imm=Immediate(1)),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(instructions=catdisent_instructions_receiver, arguments=["p0"]),
            [],
            RoutineMetadata([qubit], [qubit]),
        )

    @staticmethod
    def teledata_routine_sender(
        routine_name: str, qubit: int, ebit: int
    ) -> LocalRoutine:
        """
        Create the teledata routine for the sender node

        Args:
            routine_name: the routine name
            qubit: the qubit used
            ebit: the ebit used

        Returns:
            the qoala local routine
        """
        teleport_instructions_sender = [
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(qubit)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(ebit)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
            GateHInstruction(reg=Register(RegisterName.Q, 0)),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.M, 0)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 0),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 0)),
            ),
            MeasInstruction(
                reg0=Register(RegisterName.Q, 1), reg1=Register(RegisterName.M, 1)
            ),
            StoreInstruction(
                reg=Register(RegisterName.M, 1),
                entry=ArrayEntry(Address("output"), Register(RegisterName.R, 1)),
            ),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(instructions=teleport_instructions_sender, arguments=[]),
            ["m0", "m1"],
            RoutineMetadata([qubit, ebit], []),
        )

    @staticmethod
    def teledata_routine_receiver(
        routine_name: str, ebit: int, qubit: int
    ) -> LocalRoutine:
        """
        Create the teledata routine for the receiver node

        Args:
            routine_name: the routine name
            ebit: the ebit sued
            qubit: the qubit used

        Returns:
            the qoala local routine
        """
        teleport_instructions_receiver = [
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(0)),
            LoadInstruction(
                reg=Register(RegisterName.C, 0),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 0)),
            ),
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(1)),
            LoadInstruction(
                reg=Register(RegisterName.C, 1),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 1)),
            ),
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebit)),
            SetInstruction(reg=Register(RegisterName.Q, 1), imm=Immediate(qubit)),
            BezInstruction(reg=Register(RegisterName.C, 0), imm=Immediate(2)),
            GateXInstruction(reg=Register(RegisterName.Q, 0)),
            BezInstruction(reg=Register(RegisterName.C, 1), imm=Immediate(2)),
            GateZInstruction(reg=Register(RegisterName.Q, 0)),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 1), reg1=Register(RegisterName.Q, 0)
            ),
            CnotInstruction(
                reg0=Register(RegisterName.Q, 0), reg1=Register(RegisterName.Q, 1)
            ),
        ]

        return LocalRoutine(
            routine_name,
            Subroutine(
                instructions=teleport_instructions_receiver, arguments=["p0", "p1"]
            ),
            [],
            RoutineMetadata([ebit, qubit], [qubit]),
        )

    def ent_swap(self, nodes: list[str], ebits: list[int]):
        """
        Add entanglement swapping to the list of operation in a Qoala program

        Args:
            nodes: the list of nodes involved
            ebits: the list of ebits involved
        """
        # Measure intermediate ebits
        for i in range(1, len(ebits) - 1, 2):
            ebit0 = ebits[i]
            ebit1 = ebits[i + 1]
            node = nodes[i + 1]
            routine_name = f"bsm_routine{self.block_i[node]}"
            bsm_routine = self.bsm_subroutine(routine_name, ebit0, ebit1)
            ops = [
                RunSubroutineOp(
                    IqoalaTuple(["m0", "m1"]), IqoalaTuple([]), routine_name
                )
            ]
            block = BasicBlock("b" + str(self.block_i[node]), BasicBlockType.QL, ops)
            self.nodes_with_blocks[node][0].append(block)
            self.nodes_with_blocks[node][1].append(bsm_routine)
            self.block_i[node] += 1
        # Send corrections
        for i in range(0, len(ebits) - 2, 2):
            node = nodes[i + 2]

            if i < len(ebits) - 4:
                send_to = [nodes[i + 4], nodes[i + 4]]
            else:
                send_to = [nodes[0], nodes[-1]]

            if i != 0:
                receive_from = nodes[i - 1]

                # First parity
                assign_parity = AssignCValueOp(
                    result=IqoalaSingleton(name="parity0"), value=0
                )
                block_assign_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_parity]
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [block_assign_parity]

                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][receive_from],
                )
                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1
                receive_msg = [
                    ReceiveCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        result=IqoalaSingleton(name="p0"),
                    )
                ]
                block_receive_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CC, receive_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_receive_msg,
                ]

                check_parity = BranchIfEqualOp(
                    value0=IqoalaSingleton(name="p0"),
                    value1=IqoalaSingleton(name="m0"),
                    block_name=f"b{self.block_i[node] + 3}",
                )
                block_check_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [check_parity]
                )
                self.block_i[node] += 1

                assign_parity = AssignCValueOp(
                    result=IqoalaSingleton(name="plus0"), value=1
                )
                block_assign_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_parity]
                )
                self.block_i[node] += 1
                add_parity = AddCValueOp(
                    result=IqoalaSingleton(name="parity0"),
                    value0=IqoalaSingleton(name="p0"),
                    value1=IqoalaSingleton(name="plus0"),
                )
                block_add_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [add_parity]
                )
                self.block_i[node] += 1

                self.nodes_with_blocks[node][0] += [
                    block_check_parity,
                    block_assign_parity,
                    block_add_parity,
                ]

                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][send_to[0]],
                )
                send_msg = [
                    SendCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        value=IqoalaSingleton(name="parity0"),
                    )
                ]

                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1
                block_send_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, send_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_send_msg,
                ]

                # Second parity
                assign_parity = AssignCValueOp(
                    result=IqoalaSingleton(name="parity1"), value=0
                )
                block_assign_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_parity]
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [block_assign_parity]

                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][receive_from],
                )
                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1
                receive_msg = [
                    ReceiveCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        result=IqoalaSingleton(name="p1"),
                    )
                ]
                block_receive_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CC, receive_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_receive_msg,
                ]

                check_parity = BranchIfEqualOp(
                    value0=IqoalaSingleton(name="p1"),
                    value1=IqoalaSingleton(name="m1"),
                    block_name=f"b{self.block_i[node] + 3}",
                )
                block_check_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [check_parity]
                )
                self.block_i[node] += 1

                assign_parity = AssignCValueOp(
                    result=IqoalaSingleton(name="plus1"), value=1
                )
                block_assign_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_parity]
                )
                self.block_i[node] += 1
                add_parity = AddCValueOp(
                    result=IqoalaSingleton(name="parity1"),
                    value0=IqoalaSingleton(name="p1"),
                    value1=IqoalaSingleton(name="plus1"),
                )
                block_add_parity = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [add_parity]
                )
                self.block_i[node] += 1

                self.nodes_with_blocks[node][0] += [
                    block_check_parity,
                    block_assign_parity,
                    block_add_parity,
                ]

                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][send_to[1]],
                )
                send_msg = [
                    SendCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        value=IqoalaSingleton(name="parity1"),
                    )
                ]

                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1
                block_send_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, send_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_send_msg,
                ]
            else:
                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][send_to[0]],
                )
                send_msg = [
                    SendCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        value=IqoalaSingleton(name="m0"),
                    )
                ]

                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1
                block_send_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, send_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_send_msg,
                ]

                assign_csocket = AssignCValueOp(
                    result=IqoalaSingleton(name="csocket"),
                    value=self.sockets[node][send_to[1]],
                )

                block_assign_csocket = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
                )
                self.block_i[node] += 1

                send_msg = [
                    SendCMsgOp(
                        csocket=IqoalaSingleton(name="csocket"),
                        value=IqoalaSingleton(name="m1"),
                    )
                ]

                block_send_msg = BasicBlock(
                    "b" + str(self.block_i[node]), BasicBlockType.CL, send_msg
                )
                self.block_i[node] += 1
                self.nodes_with_blocks[node][0] += [
                    block_assign_csocket,
                    block_send_msg,
                ]

        # Perform corrections
        # First node Z corrections
        node = nodes[0]
        receive_from = nodes[-2]
        assign_csocket = AssignCValueOp(
            result=IqoalaSingleton(name="csocket"),
            value=self.sockets[node][receive_from],
        )
        block_assign_csocket = BasicBlock(
            "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
        )
        self.block_i[node] += 1
        receive_msg = [
            ReceiveCMsgOp(
                csocket=IqoalaSingleton(name="csocket"),
                result=IqoalaSingleton(name="p0"),
            )
        ]
        block_receive_msg = BasicBlock(
            "b" + str(self.block_i[node]), BasicBlockType.CC, receive_msg
        )
        self.block_i[node] += 1
        self.nodes_with_blocks[node][0] += [block_assign_csocket, block_receive_msg]

        corr_instructions = [
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(0)),
            LoadInstruction(
                reg=Register(RegisterName.C, 0),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 0)),
            ),
            # SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebits[0])),
            BezInstruction(reg=Register(RegisterName.C, 0), imm=Immediate(3)),
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebits[0])),
            GateZInstruction(reg=Register(RegisterName.Q, 0)),
            JmpInstruction(imm=Immediate(1)),
        ]
        correction_routine = LocalRoutine(
            f"ent_swap_correction_{self.block_i[node]}",
            Subroutine(instructions=corr_instructions, arguments=["p0"]),
            [],
            RoutineMetadata([ebits[0]], [ebits[0]]),
        )
        ops = [
            RunSubroutineOp(
                None, IqoalaTuple(["p0"]), f"ent_swap_correction_{self.block_i[node]}"
            )
        ]
        block = BasicBlock("b" + str(self.block_i[node]), BasicBlockType.QL, ops)
        self.nodes_with_blocks[node][0].append(block)
        self.nodes_with_blocks[node][1].append(correction_routine)
        self.block_i[node] += 1

        # Second node X corrections
        node = nodes[-1]
        receive_from = nodes[-2]
        assign_csocket = AssignCValueOp(
            result=IqoalaSingleton(name="csocket"),
            value=self.sockets[node][receive_from],
        )
        block_assign_csocket = BasicBlock(
            "b" + str(self.block_i[node]), BasicBlockType.CL, [assign_csocket]
        )
        self.block_i[node] += 1
        receive_msg = [
            ReceiveCMsgOp(
                csocket=IqoalaSingleton(name="csocket"),
                result=IqoalaSingleton(name="p1"),
            )
        ]
        block_receive_msg = BasicBlock(
            "b" + str(self.block_i[node]), BasicBlockType.CC, receive_msg
        )
        self.block_i[node] += 1
        self.nodes_with_blocks[node][0] += [block_assign_csocket, block_receive_msg]

        corr_instructions = [
            # SetInstruction(reg=Register(RegisterName.C, 15), imm=Immediate(0)),
            LoadInstruction(
                reg=Register(RegisterName.C, 0),
                entry=ArrayEntry(Address("input"), Register(RegisterName.R, 0)),
            ),
            # SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebits[-1])),
            BezInstruction(reg=Register(RegisterName.C, 0), imm=Immediate(3)),
            SetInstruction(reg=Register(RegisterName.Q, 0), imm=Immediate(ebits[-1])),
            GateXInstruction(reg=Register(RegisterName.Q, 0)),
            JmpInstruction(imm=Immediate(1)),
        ]
        correction_routine = LocalRoutine(
            f"ent_swap_correction_{self.block_i[node]}",
            Subroutine(instructions=corr_instructions, arguments=["p1"]),
            [],
            RoutineMetadata([ebits[-1]], [ebits[-1]]),
        )
        ops = [
            RunSubroutineOp(
                None, IqoalaTuple(["p1"]), f"ent_swap_correction_{self.block_i[node]}"
            )
        ]
        block = BasicBlock("b" + str(self.block_i[node]), BasicBlockType.QL, ops)
        self.nodes_with_blocks[node][0].append(block)
        self.nodes_with_blocks[node][1].append(correction_routine)
        self.block_i[node] += 1
