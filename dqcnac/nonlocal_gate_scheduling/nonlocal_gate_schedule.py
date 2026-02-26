"""The Nonlocal gates scheduler module."""

import logging
import time as tm
from copy import copy, deepcopy
from typing import Dict, List, Set, Tuple, Union

import networkx as nx
from qiskit import QuantumRegister
from qiskit.circuit import Barrier, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode

from ..network import Network
from .compilation_state import Cover, State
from .nonlocal_operations import CatDisEnt, CatEnt, Teleport

logger = logging.getLogger(__name__)

MIG_EXP = 3


class NonlocalGateSchedule:
    """
    A nonlocal gates' scheduler class.
    """

    def __init__(
        self,
        network: Network,
        nodes_mapping: Dict[Qubit, str],
        use_teleport: bool = True,
    ):
        """
        Initialize the remote gates' scheduler.

        Args:
            network: the network to use for remote operations.
            nodes_mapping:
            use_teleport: whether to use teledata operations or not.
        """

        self.nodes_mapping = nodes_mapping
        self.partitions = {node: [] for node in network.nodes}
        for q, p in self.nodes_mapping.items():
            self.partitions[p].append(q)
        self.network = network
        self.use_tel = use_teleport
        self.non_local_gate_counter: dict[int, int] = {}

    @staticmethod
    def local_op(qargs: List[Qubit], nodes_mapping: Dict[Qubit, str]) -> bool:
        """
        Check if an operation is local or not.

        Args:
            qargs: the qubits on which the operation acts
            nodes_mapping: the current mapping of qubits to nodes in a network

        Returns:
            True if the operation is local, False otherwise
        """

        # qargs already mapped with mapping
        if len(qargs) == 1:
            return True
        qpu0 = nodes_mapping[qargs[0]]
        return all([nodes_mapping[qarg] == qpu0 for qarg in qargs])

    def add_gate(
        self,
        gate,
        executed: List[Tuple[int, DAGOpNode]],
        times: Dict[Qubit, int],
        qargs: List[Qubit] = None,
    ):
        """
        Add a scheduled gate to the compilation state.

        Args:
            gate: the gate to add
            qargs: the qubits on which the gate acts
            executed: the current list of executed gates
            times: the times of last usage for each qubit
        """

        if isinstance(gate, Teleport):
            if self.nodes_mapping[gate.qargs[-1]] != self.nodes_mapping[gate.qargs[-2]]:
                logger.error(
                    f"TELEPORT: {gate.qargs} {self.nodes_mapping[gate.qargs[-2]]}-{self.nodes_mapping[gate.qargs[-1]]}"
                )
                raise Exception(
                    f"TELEPORT: {gate.qargs} {self.nodes_mapping[gate.qargs[-2]]}-{self.nodes_mapping[gate.qargs[-1]]}"
                )
        new_gate = copy(gate)
        if qargs is None:
            qargs = list(gate.qargs)
        else:
            new_gate = copy(gate)
            new_gate.qargs = qargs
        try:
            time = max([times[q] for q in qargs]) + 1
        except Exception as e:
            logger.error(f"QRAGS: {qargs}")
            logger.error(f"TIMES: {times}")
            raise e
        for q in qargs:
            times[q] = time
        executed.append((time, new_gate))

    @staticmethod
    def circuit_cost(executed: List[Tuple[int, DAGOpNode]]) -> int:
        """
        Compute the cost of the circuit regarding consumed ebits.

        Args:
            executed: the list of executed gates

        Returns:
            the  estimated cost of the circuit
        """

        cost = 0
        for item in executed:
            if isinstance(item[1], (Teleport, CatEnt)):
                cost += len(item[1].qargs[1:]) // 2
        return cost

    def is_covered(
        self,
        gate: DAGOpNode,
        migration: Cover,
        time1: int,
        times: Dict[Qubit, int],
        covered: Set[DAGOpNode],
        mapping: Dict[Qubit, Qubit],
        time2: int = None,
    ) -> bool:
        """
        Check if migration covers the gate, check also for coverage based on timing

        Args:
            gate: the gate to check coverage for
            migration: the migration to check coverage for
            time1: the time of the first migration
            times: the times of last usage for each qubit
            covered: the set of covered gates
            mapping: the mapping of qubits to qubits
            time2: the time of the second migration

        Returns:
            True if migration covers the gate, False otherwise
        """
        qargs = gate.qargs
        if migration.migrated[0] not in qargs and (
            time2 is not None and migration.migrated[1] not in qargs
        ):
            return False

        for p in self.dag_circuit.quantum_predecessors(gate):
            if not isinstance(p, DAGOpNode):
                continue
            if p not in covered:
                return False

        if (
            self.nodes_mapping[
                mapping[list(set(gate.qargs).difference([migration.migrated[0]]))[0]]
            ]
            == migration.migrations[migration.migrated[0]][-1].name
            and migration.migrated[0] in qargs
        ) or (
            time2 is not None
            and self.nodes_mapping[
                mapping[list(set(gate.qargs).difference([migration.migrated[1]]))[0]]
            ]
            == migration.migrations[migration.migrated[1]][-1].name
            and migration.migrated[1] in qargs
        ):
            if time2 is None:
                time2 = 0
            exec_time = max([times[mapping[q]] for q in qargs]) + 1
            if exec_time <= max([time1, time2]) + MIG_EXP:
                return True
        return False

    def run(
        self,
        dag_circuit: DAGCircuit,
        gate_grouping: bool,
        count_non_local_gates: bool = False,
    ) -> Tuple[State, Dict[Qubit, Qubit]]:
        """
        Run the remote operation scheduler over a circuit.

        Args:
            dag_circuit: the circuit to schedule
            gate_grouping: whether to group gates
            count_non_local_gates: whether to count non-local gates

        Returns:
        The state of compilation after scheduling and the final mapping of the qubits
        """
        logger.debug(f"\n\nDISTRIBUTED COMPILATION\n\n")

        q = QuantumRegister(
            sum([self.network.nodes[node].mem_cap for node in self.network.nodes]), "q"
        )

        old_qregs = dict()

        self.idle_qubits = {p: set() for p in self.partitions}

        self.nodes_mapping = dict()

        num_qubits = 0
        k = 0

        for p in self.partitions:
            for i in range(self.network.nodes[p].mem_cap):
                if i < len(self.partitions[p]):
                    old_qregs[self.partitions[p][i]] = q[i + k]
                else:
                    self.idle_qubits[p].add(q[i + k])
                self.nodes_mapping[q[i + k]] = p
                num_qubits += 1
            k = num_qubits

        logger.info(f"\nIDLE QUBITS: {self.idle_qubits}")
        logger.info(f"\nOLD QREGS: {old_qregs}")
        logger.info(f"NODES MAPPING: {self.nodes_mapping}\n")

        # create new circuit with just one qreg
        self.dag_circuit = DAGCircuit()
        self.dag_circuit.add_qreg(q)
        self.dag_circuit.metadata = dag_circuit.metadata
        self.dag_circuit.add_clbits(dag_circuit.clbits)
        for creg in dag_circuit.cregs.values():
            self.dag_circuit.add_creg(creg)

        for node in dag_circuit.topological_op_nodes():
            qargs = [old_qregs[qarg] for qarg in node.qargs]
            self.dag_circuit.apply_operation_back(node.op, qargs, node.cargs)

        self.qubits = {q: [0] for q in self.dag_circuit.qubits}
        self.mapping = {q: q for q in self.qubits}

        self.connections = {}
        for n1 in self.network.nodes:
            self.connections[n1] = {}
            for n2 in self.network.graph.nodes[self.network.nodes[n1]]["conn"]:
                self.connections[n1][n2] = self.network.graph.nodes[
                    self.network.nodes[n1]
                ]["conn"][n2]["ebits"]

        self.ebits = {node: set() for node in self.network.nodes}
        for n1 in self.connections:
            for n2, ebits in self.connections[n1].items():
                self.ebits[n1].update(ebits)
        for n1 in self.ebits:
            self.ebits[n1] = list(self.ebits[n1])

        timed_ent = dict()
        timed_ent_list = list()
        for n1 in self.network.nodes:
            for n2 in self.network.graph.nodes[self.network.nodes[n1]]["conn"]:
                if n2 != n1:
                    if (n2, n1) not in timed_ent:
                        timed_ent[(n1, n2)] = len(timed_ent_list)
                        timed_ent_list.append({-1: 0})
                    else:
                        timed_ent[(n1, n2)] = timed_ent[(n2, n1)]

        timed_tele = dict()
        timed_tele_list = list()
        for n1 in self.network.nodes:
            for n2 in self.network.graph.nodes[self.network.nodes[n1]]["conn"]:
                if n2 != n1:
                    if (n2, n1) not in timed_tele:
                        timed_tele[(n1, n2)] = len(timed_tele_list)
                        timed_tele_list.append({-1: 0})
                    else:
                        timed_tele[(n1, n2)] = timed_tele[(n2, n1)]

        ebits_cap = {
            (n1, n2): len(v)
            for n1 in self.connections
            for n2, v in self.connections[n1].items()
        }

        ebits = {n: {} for n in self.network.nodes}

        self.times = {q: -1 for q in self.qubits}

        self.executed = [(-1, DAGOpNode(Barrier(len(q)), qargs=q))]

        for n in self.ebits:
            er = QuantumRegister(len(self.ebits[n]), f"qr|{n}")

            for i, e in enumerate(self.ebits[n]):
                ebits[n][e] = er[i]
                self.times.update({er[i]: -1})
                self.mapping[er[i]] = er[i]
                self.nodes_mapping.update({er[i]: n})
            self.executed.append((-1, DAGOpNode(Barrier(1), qargs=er)))

        self.initial_mapping = copy(self.nodes_mapping)

        gates = [node for node in self.dag_circuit.topological_op_nodes()]

        self.uncovered = set(gates)
        for gate in gates:
            if self.local_op(
                [self.mapping[qarg] for qarg in gate.qargs], self.nodes_mapping
            ):
                self.uncovered.remove(gate)

        covered = set()
        all_covered = set()
        to_execute = copy(gates)

        state = State(
            **{
                "timed_ent": timed_ent,
                "timed_tele": timed_tele,
                "ebits_cap": ebits_cap,
                "times": self.times,
                "executed": self.executed,
                "ebits": ebits,
                "covered": covered,
                "all_covered": all_covered,
                "mapping": self.mapping,
                "nodes_mapping": self.nodes_mapping,
                "initial_mapping": self.initial_mapping,
                "idle_qubits": self.idle_qubits,
                "timed_ent_list": timed_ent_list,
                "timed_tele_list": timed_tele_list,
            }
        )

        if gate_grouping:
            pre_compiled = []

            _start = tm.time()

            tel_cost = None
            for instruction in to_execute:
                if "circuit" not in instruction.name:
                    gate = instruction
                    if gate in state.all_covered:
                        continue

                    else:
                        self.add_gate(
                            gate,
                            state.executed,
                            state.times,
                            qargs=[state.mapping[q] for q in gate.qargs],
                        )
                        state.all_covered.add(gate)
                        continue
                else:
                    self.compile_multi_gate(instruction, state, old_qregs)

        else:
            pre_compiled = []

            _start = tm.time()

            tel_cost = None
            for gate in to_execute:
                if gate in state.all_covered:
                    continue

                if len(gate.qargs) == 1 or self.local_op(
                    [state.mapping[qarg] for qarg in gate.qargs], state.nodes_mapping
                ):
                    self.add_gate(
                        gate,
                        state.executed,
                        state.times,
                        qargs=[state.mapping[q] for q in gate.qargs],
                    )
                    state.all_covered.add(gate)
                    continue
                if self.use_tel is True:
                    logger.info(f"\n\nEVALUATE TELEPORT FOR {gate.op} -> {gate.qargs}")
                    tel_cost, tel, tel_pre_compiled = self.get_teleport(
                        gate, deepcopy(state)
                    )
                    logger.debug(f"TEL STATE: {tel}\n")

                logger.info(f"\n\nEVALUATE MIGRATION")

                mig_state, compiled, mig_pre_compiled = self.compile(
                    self.dag_circuit,
                    deepcopy(state),
                    stop_early=True,
                    previous=pre_compiled,
                )

                logger.debug(f"MIG STATE: {compiled}\n")
                mig_cost = self.circuit_cost(mig_state.executed)
                logger.info(f"TEL COST: {tel_cost}\t MOG COST: {mig_cost}\n")

                if tel_cost is not None and tel_cost < mig_cost:
                    state = deepcopy(tel)
                    pre_compiled = tel_pre_compiled
                else:
                    state = deepcopy(compiled)
                    pre_compiled = mig_pre_compiled

                logger.debug(f"NODES MAPPING: {state.nodes_mapping}\n")
                logger.debug(f"REG MAPPING: {state.mapping}")
                logger.debug(f"IDLE QUBITS: {state.idle_qubits}")

        logger.info(f"NODES MAPPING: {self.nodes_mapping}\n")
        logger.info(f"REG MAPPING: {self.mapping}")

        logger.info("EXECUTED")
        for item in sorted(state.executed, key=lambda x: x[0]):
            logger.info(f"GATE {item[1].op}: {item[1].qargs}")

        if count_non_local_gates:
            non_local_gates = 0
            for k, v in self.non_local_gate_counter.items():
                print(f"Number of epr pairs used for {k} non-local gates: {v}")
                non_local_gates += v * k
            print(f"Total number of non-local gates: {non_local_gates}")

        return state, {v: k for k, v in old_qregs.items()}

    def find_teleports(self, gate: DAGOpNode, state: State) -> List[Cover]:
        """
        Find possible teledata operations to cover a gate

        Args:
            gate: the gate to cover
            state: the current compilation state

        Returns:
            a list of possible teledata operations
        """

        qargs = [state.mapping[qarg] for qarg in gate.qargs]
        p_start = state.nodes_mapping[qargs[0]]
        p_end = state.nodes_mapping[qargs[1]]
        teleports = []
        time = max([state.times[q] for q in qargs])
        path = nx.shortest_path(
            self.network.graph, self.network.nodes[p_start], self.network.nodes[p_end]
        )

        if len(state.idle_qubits[path[-1].name]) != 0:
            teleports.append(Cover(gate.qargs[0], path, time=time))
        if len(state.idle_qubits[path[0].name]) != 0:
            teleports.append(Cover(gate.qargs[1], path[::-1], time=time))

        if len(path) > 2:
            for i, mid_point in enumerate(path[1:-1]):
                if len(state.idle_qubits[mid_point.name]) >= 2:
                    teleport = Cover(gate.qargs[0], path[: i + 2], time=time)
                    teleport.add(gate.qargs[1], path[i + 1 :][::-1])
                    teleports.append(teleport)
        return teleports

    def get_min_ebit(
        self, state: State, node1: str, node2: str, exclude: Set = None
    ) -> Qubit:
        """
        Find the least used ebit from node1 to node 2.

        Args:
            state: the current compilation state
            node1: the first node
            node2: the second node
            exclude: ebits to exclude from the search

        Returns:
            the ebit
        """

        ebits = [state.ebits[node1][e] for e in self.connections[node1][node2]]
        if exclude is not None:
            ebits = set(ebits).difference(exclude)

        return min(list(ebits), key=lambda x: state.times[state.mapping[x]])

    def tel_tel(self, state: State, teleport: Cover, gate: DAGOpNode) -> List[Qubit]:
        """
        Add teleport to executed gates.

        Args:
            state: the current compilation state
            teleport: the teleport to add
            gate: the gate covered by the teleport operation

        Returns:
            the list of ebits used for the teleport
        """

        # find idle data qubits to use for teleportation at destination node
        state.idle_qubits[teleport.migrations[teleport.migrated[0]][0].name].add(
            state.mapping[teleport.migrated[0]]
        )

        # select ebits to use
        e1 = []
        e2 = []
        _e1 = []
        _e2 = []
        used = set()
        for a, b in zip(
            teleport.migrations[teleport.migrated[0]][:-1],
            teleport.migrations[teleport.migrated[0]][1:],
        ):
            e1.append(self.get_min_ebit(state, a.name, b.name, exclude=used))
            e2.append(self.get_min_ebit(state, b.name, a.name, exclude=used))
            used.add(e1[-1])
            used.add(e2[-1])

        ebit_path = []
        for e1_, e2_ in zip(e1, e2):
            ebit_path.extend([e1_, e2_])

        logger.info(f"EBITH PATH: {ebit_path}")

        # select the less recently used idle data qubits
        idle_qubit = min(
            [
                q
                for q in state.idle_qubits[
                    teleport.migrations[teleport.migrated[0]][-1].name
                ]
            ],
            key=lambda x: state.times[x],
        )

        self.add_gate(
            Teleport(
                qargs=[state.mapping[teleport.migrated[0]]] + ebit_path + [idle_qubit]
            ),
            state.executed,
            state.times,
        )

        # update idle qubits and change mapping of teleported data qubit to its destination
        state.idle_qubits[teleport.migrations[teleport.migrated[0]][-1].name].remove(
            idle_qubit
        )
        state.mapping[teleport.migrated[0]] = idle_qubit

        # update timing information
        for a, b in zip(
            teleport.migrations[teleport.migrated[0]][:-1],
            teleport.migrations[teleport.migrated[0]][1:],
        ):
            tele_time = max([state.times[state.mapping[q]] for q in gate.qargs])
            if (
                tele_time
                not in state.timed_tele_list[state.timed_tele[(a.name, b.name)]]
            ):
                state.timed_tele_list[state.timed_tele[(a.name, b.name)]].update(
                    {tele_time: 0}
                )
                state.timed_tele_list[state.timed_tele[(b.name, a.name)]].update(
                    {tele_time: 0}
                )
            state.timed_tele_list[state.timed_tele[(a.name, b.name)]][tele_time] -= 1
            state.timed_tele_list[state.timed_tele[(b.name, a.name)]][tele_time] -= 1

        # if both data qubits are teleported
        if len(teleport.migrated) > 1:
            for a, b in zip(
                teleport.migrations[teleport.migrated[1]][:-1],
                teleport.migrations[teleport.migrated[1]][1:],
            ):
                _e1.append(self.get_min_ebit(state, a.name, b.name, exclude=used))
                _e2.append(self.get_min_ebit(state, b.name, a.name, exclude=used))
                used.add(_e1[-1])
                used.add(_e2[-1])

            ebit_path = []
            for e1_, e2_ in zip(_e1, _e2):
                ebit_path.extend([e1_, e2_])

            idle_qubit = min(
                [
                    q
                    for q in state.idle_qubits[
                        teleport.migrations[teleport.migrated[1]][-1].name
                    ]
                ],
                key=lambda x: state.times[x],
            )

            state.idle_qubits[teleport.migrations[teleport.migrated[1]][0].name].add(
                state.mapping[teleport.migrated[1]]
            )

            self.add_gate(
                Teleport(
                    qargs=[state.mapping[teleport.migrated[1]]]
                    + ebit_path
                    + [idle_qubit]
                ),
                state.executed,
                state.times,
            )

            state.idle_qubits[
                teleport.migrations[teleport.migrated[1]][-1].name
            ].remove(idle_qubit)

            state.mapping[teleport.migrated[1]] = idle_qubit

            for a, b in zip(
                teleport.migrations[teleport.migrated[1]][:-1],
                teleport.migrations[teleport.migrated[1]][1:],
            ):
                tele_time = max([state.times[state.mapping[q]] for q in gate.qargs])
                if (
                    tele_time
                    not in state.timed_tele_list[state.timed_tele[(a.name, b.name)]]
                ):
                    state.timed_tele_list[state.timed_tele[(a.name, b.name)]].update(
                        {tele_time: 0}
                    )
                    state.timed_tele_list[state.timed_tele[(b.name, a.name)]].update(
                        {tele_time: 0}
                    )
                state.timed_tele_list[state.timed_tele[(a.name, b.name)]][
                    tele_time
                ] -= 1
                state.timed_tele_list[state.timed_tele[(b.name, a.name)]][
                    tele_time
                ] -= 1

        for i, (a, b) in enumerate(
            zip(
                teleport.migrations[teleport.migrated[0]][:-1],
                teleport.migrations[teleport.migrated[0]][1:],
            )
        ):

            exec_time = max([state.times[state.mapping[q]] for q in gate.qargs]) + 1
            if (
                exec_time
                not in state.timed_tele_list[state.timed_tele[(a.name, b.name)]]
            ):
                state.timed_tele_list[state.timed_tele[(a.name, b.name)]].update(
                    {exec_time: 0}
                )
                state.timed_tele_list[state.timed_tele[(b.name, a.name)]].update(
                    {exec_time: 0}
                )
            state.timed_tele_list[state.timed_tele[(a.name, b.name)]][exec_time] += 1
            state.timed_tele_list[state.timed_tele[(b.name, a.name)]][exec_time] += 1

        if len(teleport.migrated) > 1:
            for i, (a, b) in enumerate(
                zip(
                    teleport.migrations[teleport.migrated[1]][:-1],
                    teleport.migrations[teleport.migrated[1]][1:],
                )
            ):

                exec_time = max([state.times[state.mapping[q]] for q in gate.qargs]) + 1
                if (
                    exec_time
                    not in state.timed_tele_list[state.timed_tele[(a.name, b.name)]]
                ):
                    state.timed_tele_list[state.timed_tele[(a.name, b.name)]].update(
                        {exec_time: 0}
                    )
                    state.timed_tele_list[state.timed_tele[(b.name, a.name)]].update(
                        {exec_time: 0}
                    )
                state.timed_tele_list[state.timed_tele[(a.name, b.name)]][
                    exec_time
                ] += 1
                state.timed_tele_list[state.timed_tele[(b.name, a.name)]][
                    exec_time
                ] += 1
        return e1

    def tel_cov_succ(
        self, state: State, teleport: Cover, gate: DAGOpNode, e1: List[Qubit]
    ):
        """
        Check if teleport can cover more than one gate

        Args:
            state: the current compilation state
            teleport: the teleport operation
            gate: the gate covered by the teleport
            e1: the list of ebits used for the teleport
        """

        successors = list(self.dag_circuit.quantum_successors(gate))
        breaks = [False, False]

        if len(teleport.migrated) == 1:
            breaks[1] = True

        while len(successors) != 0:
            remove = set()
            extra = None
            for g in successors:
                if not isinstance(g, DAGOpNode):
                    remove.add(g)
                    continue
                if self.local_op(
                    [state.mapping[qarg] for qarg in g.qargs], state.nodes_mapping
                ):
                    if teleport.migrated[0] in g.qargs:
                        breaks[0] = breaks[0] or True
                        if breaks[1] == True:
                            break
                        else:
                            continue
                    elif len(teleport.migrated) > 1 and teleport.migrated[1] in g.qargs:
                        breaks[1] = breaks[1] or True
                        if breaks[0] == True:
                            break
                        else:
                            continue
                    else:
                        remove.add(g)
                        continue
                if g.qargs[0] not in gate.qargs and g.qargs[1] not in gate.qargs:
                    remove.add(g)
                    continue
                if g not in state.covered:
                    if len(teleport.migrated) > 1:
                        time2 = state.times[e1[0]]
                    else:
                        time2 = None
                    if self.is_covered(
                        g,
                        teleport,
                        state.times[e1[0]],
                        state.times,
                        state.all_covered,
                        state.mapping,
                        time2=time2,
                    ):
                        extra = g
                        remove.add(g)
                        state.covered.add(g)
                        state.all_covered.add(g)
                        self.add_gate(
                            g,
                            state.executed,
                            state.times,
                            qargs=[state.mapping[q] for q in g.qargs],
                        )
            if extra is None:
                break
            for r in remove:
                successors.remove(r)
            successors.extend(list(self.dag_circuit.quantum_successors(extra)))

    def tel_post_proces(
        self,
        _state: State,
        temp_nodes_mapping: Dict[Qubit, str],
        temp_mapping: Dict[Qubit, Qubit],
        temp_timed_tele_list: List[Dict[int, int]],
        temp_executed: List[Tuple[int, DAGOpNode]],
        temp_times: Dict[Qubit, int],
    ) -> Tuple[State, State]:
        """
        Update compilation state after teleport

        Args:
            _state: the current compilation state
            temp_nodes_mapping: the new nodes mapping
            temp_mapping: the new mapping
            temp_timed_tele_list:  the new rimed_tele_list
            temp_executed: the new executed list
            temp_times: the new times

        Returns:
            the new state and a copy of it for further processing
        """

        temp_state = _state.get_simple_copy()
        temp_state.nodes_mapping = temp_nodes_mapping
        temp_state.mapping = temp_mapping
        temp_state.timed_tele_list = temp_timed_tele_list
        temp_state.executed = temp_executed
        temp_state.times = temp_times

        tel_state = deepcopy(temp_state)

        return temp_state, tel_state

    def tel_for(
        self, teleports: List[Cover], state: State, gate: DAGOpNode, cost: int
    ) -> Tuple[int, State]:
        """
        Search for best teleport

        Args:
            teleports: the possible teleport operations
            state: the current compilation state
            gate: the gate to cover
            cost: the cost

        Returns:
            the cost of the best teleport and the consequent compilation state
        """

        best_state = None
        best_tel = None

        for teleport in teleports:

            _state = deepcopy(state)

            e1 = self.tel_tel(_state, teleport, gate)

            self.add_gate(
                gate,
                _state.executed,
                _state.times,
                qargs=[_state.mapping[q] for q in gate.qargs],
            )

            logger.info(f"TO COVER: {gate.op}: {gate.qargs}")

            _state.all_covered.add(gate)
            _state.covered.add(gate)

            temp_state, tel_state = self.tel_post_proces(
                _state,
                _state.nodes_mapping,
                _state.mapping,
                _state.timed_tele_list,
                _state.executed,
                _state.times,
            )

            temp_state, previous = self.tel_compile(
                self.dag_circuit, temp_state, stop_early=True
            )
            new_cost = self.circuit_cost(temp_state.executed)

            if cost is None or new_cost < cost:
                cost = new_cost
                best_state = tel_state
                best_tel = teleport

        if best_tel is not None:
            logger.info(
                f"BEST TELEPORT {best_tel.migrated[0]}: {[node.name for node in best_tel.migrations[list(best_tel.migrations.keys())[0]]]}\n"
            )
            logger.info(f"IDLE QUBITS: {best_state.idle_qubits}")

        return cost, best_state

    def get_teleport(self, gate: DAGOpNode, state: State) -> Tuple[int, State, List]:
        """
        Compile with teleports

        Args:
            gate: the gate to cover
            state: the current compiled state

        Returns:
            the cost of the teleport, the consequent compilation state, and an empty list
        """

        previous = list()
        teleports = self.find_teleports(gate, state)

        cost = None

        cost, tel_state = self.tel_for(teleports, state, gate, cost)

        return cost, tel_state, previous

    def comp_preprocess(
        self, dag_circuit: DAGCircuit, state: State, previous: List[State]
    ) -> Tuple[Dict[DAGOpNode, None], Set[DAGOpNode], int]:
        """
        Preprocess before compilation with cat-ent. Update list of uncovered and covered gates

        Args:
            dag_circuit: the dag circuit to compile
            state: the current compilation state
            previous: a list of previously compiled states, possibly derived from previous evaluations during coverage of other gates

        Returns:
            the list of gates, the update lists of uncovered and covered gates
        """

        gates = {
            node: None
            for node in dag_circuit.topological_op_nodes()
            if node not in state.all_covered
            or (len(previous) != 0 and node not in previous[-1].all_covered)
        }

        uncovered = set(gates)

        previous_covered = 0

        for gate in gates:
            if (
                self.local_op(
                    [state.mapping[qarg] for qarg in gate.qargs], state.nodes_mapping
                )
                or gate in state.all_covered
            ):
                uncovered.remove(gate)
            elif len(previous) != 0 and (
                self.local_op(
                    [previous[-1].mapping[qarg] for qarg in gate.qargs],
                    previous[-1].nodes_mapping,
                )
                or gate in previous[-1].all_covered
            ):
                previous_covered += 1
                uncovered.remove(gate)
            else:
                break

        return gates, uncovered, previous_covered

    # @profile
    def comp_for(
        self,
        gates: Dict[DAGOpNode, None],
        state: State,
        uncovered: Set[DAGOpNode],
        to_remove: Set[DAGOpNode],
        remote_ops_count: int,
        dag_circuit: DAGCircuit,
        to_cover: Union[DAGOpNode, None],
        best_cost: int,
    ):
        """
        Search for remote gate to cover

        Args:
            gates: the list of gates in the circuit
            state: the current compiled state
            uncovered: the list of uncovered gates
            to_remove: a list of gates to remove form the gates' list, already covered by previous compilation steps
            remote_ops_count: the number of remote operations already scheduled
            dag_circuit: the circuit to compile
            to_cover: the next gate to cover
            best_cost: the cost of the coverage

        Returns:
            the best coverage, either teleport or telegate, and the cost of the coverage
        """

        for gate in gates:
            if gate in state.all_covered:
                continue
            if remote_ops_count == len(uncovered):
                break
            if gate in to_remove:
                continue
            if len(gate.qargs) == 1 or self.local_op(
                [state.mapping[qarg] for qarg in gate.qargs], state.nodes_mapping
            ):
                self.add_gate(
                    gate,
                    state.executed,
                    state.times,
                    qargs=[state.mapping[q] for q in gate.qargs],
                )
                to_remove.add(gate)
                state.all_covered.add(gate)
                state.covered.add(gate)

                continue

            time = max([state.times[state.mapping[q]] for q in gate.qargs]) + 1

            cov_cost = self.cost(dag_circuit, gate, time, uncovered, state)

            if len(cov_cost.migrated) > 1:
                logger.debug(
                    f"MIG: {(cov_cost.migrated[0], cov_cost.migrations[cov_cost.migrated[0]][-1].name, cov_cost.migrated[1], cov_cost.migrations[cov_cost.migrated[1]][-1].name, cov_cost.time, cov_cost.migrations)}"
                )
            else:
                logger.debug(
                    f"MIG: {(cov_cost.migrated[0], cov_cost.migrations[cov_cost.migrated[0]][-1].name, cov_cost.time, cov_cost.cost)}"
                )
            best_cost = cov_cost
            to_cover = gate
            break

        return best_cost, to_cover

    def comp_ent(
        self,
        state: State,
        to_cover: DAGOpNode,
        to_remove: Set[DAGOpNode],
        best_cost: Cover,
    ) -> Tuple[
        Union[List[Qubit], None],
        Union[List[Qubit], None],
        Union[List[Qubit], None],
        Union[List[Qubit], None],
        Union[List[Qubit], None],
    ]:
        """
        Add cat-ent to cover gate (to_cover)

        Args:
            state: the current compiled state
            to_cover: the gate to cover
            to_remove: a list of gates to remove from the gates' list, already covered by previous compilation steps
            best_cost: the cost of the cat-ent

        Returns:
            the qubits which the covered gate acts, the ebits used to perform the cat-ent
        """

        logger.debug(f"TO COVER: {to_cover} {to_cover.qargs}")

        state.covered.add(to_cover)
        state.all_covered.add(to_cover)
        to_remove.add(to_cover)

        if (
            state.nodes_mapping[state.mapping[to_cover.qargs[0]]]
            == state.nodes_mapping[state.mapping[to_cover.qargs[1]]]
        ):
            self.add_gate(
                to_cover,
                state.executed,
                state.times,
                qargs=[
                    state.mapping[to_cover.qargs[0]],
                    state.mapping[to_cover.qargs[1]],
                ],
            )
            return None, None, None, None, None

        new_qargs = [state.mapping[to_cover.qargs[0]], state.mapping[to_cover.qargs[1]]]

        e1 = []
        e2 = []
        _e1 = []
        _e2 = []

        used = set()

        for a, b in zip(
            best_cost.migrations[best_cost.migrated[0]][:-1],
            best_cost.migrations[best_cost.migrated[0]][1:],
        ):
            e1.append(self.get_min_ebit(state, a.name, b.name, exclude=used))
            e2.append(self.get_min_ebit(state, b.name, a.name, exclude=used))
            used.add(e1[-1])
            used.add(e2[-1])

        ebit_path = []

        for e1_, e2_ in zip(e1, e2):
            ebit_path.extend([e1_, e2_])

        self.add_gate(
            CatEnt(qargs=[state.mapping[best_cost.migrated[0]]] + ebit_path),
            state.executed,
            state.times,
        )

        logger.debug(f"E1:{e1} - E2:{e2}")
        logger.debug(f"EBIT PATH: {ebit_path}")
        logger.debug(f"ADDED CAT ENT: {[best_cost.migrated[0]] + ebit_path}")
        new_qargs[new_qargs.index(state.mapping[best_cost.migrated[0]])] = ebit_path[-1]

        for a, b in zip(
            best_cost.migrations[best_cost.migrated[0]][:-1],
            best_cost.migrations[best_cost.migrated[0]][1:],
        ):
            ent_time = max([state.times[state.mapping[q]] for q in to_cover.qargs])
            if ent_time not in state.timed_ent_list[state.timed_ent[(a.name, b.name)]]:
                state.timed_ent_list[state.timed_ent[(a.name, b.name)]].update(
                    {ent_time: 0}
                )
                state.timed_ent_list[state.timed_ent[(b.name, a.name)]].update(
                    {ent_time: 0}
                )
            state.timed_ent_list[state.timed_ent[(a.name, b.name)]][ent_time] -= 1
            state.timed_ent_list[state.timed_ent[(b.name, a.name)]][ent_time] -= 1

        if len(best_cost.migrated) > 1:
            for a, b in zip(
                best_cost.migrations[best_cost.migrated[1]][:-1],
                best_cost.migrations[best_cost.migrated[1]][1:],
            ):
                _e1.append(self.get_min_ebit(state, a.name, b.name, exclude=used))
                _e2.append(self.get_min_ebit(state, b.name, a.name, exclude=used))
                used.add(_e1[-1])
                used.add(_e2[-1])

                ebit_path = []
                for e1_, e2_ in zip(_e1, _e2):
                    ebit_path.extend([e1_, e2_])

                self.add_gate(
                    CatEnt(qargs=[state.mapping[best_cost.migrated[1]]] + ebit_path),
                    state.executed,
                    state.times,
                )

            new_qargs[
                new_qargs.index(state.mapping[best_cost.migrated[1]])
            ] = ebit_path[-1]

            for a, b in zip(
                best_cost.migrations[best_cost.migrated[1]][:-1],
                best_cost.migrations[best_cost.migrated[1]][1:],
            ):
                disent_time = max(
                    [state.times[state.mapping[q]] for q in to_cover.qargs]
                )
                if (
                    disent_time
                    not in state.timed_ent_list[state.timed_ent[(a.name, b.name)]]
                ):
                    state.timed_ent_list[state.timed_ent[(a.name, b.name)]].update(
                        {disent_time: 0}
                    )
                    state.timed_ent_list[state.timed_ent[(b.name, a.name)]].update(
                        {disent_time: 0}
                    )
                state.timed_ent_list[state.timed_ent[(a.name, b.name)]][
                    disent_time
                ] += 1
                state.timed_ent_list[state.timed_ent[(b.name, a.name)]][
                    disent_time
                ] += 1

        self.add_gate(to_cover, state.executed, state.times, qargs=new_qargs)

        return new_qargs, e1, _e1, e2, _e2

    def comp_dis_ent(
        self,
        best_cost: Cover,
        state: State,
        to_cover: DAGOpNode,
        e2: List[Qubit],
        _e2: List[Qubit],
    ):
        """
        Add cat-disent after gate (to_cover).

        Args:
            best_cost: the cist of the cat-disent
            state: the current compilation state
            to_cover: the gate covered
            e2: the ebits used for the cat-disent, node2 to node1
            _e2: the ebits used for the cat-disent, only for migration on a middle node
        """

        qargs = e2[::-1] + [state.mapping[best_cost.migrated[0]]]
        self.add_gate(
            CatDisEnt(qargs=qargs),
            state.executed,
            state.times,
        )

        for i, (a, b) in enumerate(
            zip(
                best_cost.migrations[best_cost.migrated[0]][:-1],
                best_cost.migrations[best_cost.migrated[0]][1:],
            )
        ):

            exec_time = max([state.times[state.mapping[q]] for q in to_cover.qargs])

            if exec_time not in state.timed_ent_list[state.timed_ent[(a.name, b.name)]]:
                state.timed_ent_list[state.timed_ent[(a.name, b.name)]].update(
                    {exec_time: 0}
                )
                state.timed_ent_list[state.timed_ent[(b.name, a.name)]].update(
                    {exec_time: 0}
                )
            state.timed_ent_list[state.timed_ent[(a.name, b.name)]][exec_time] += 1
            state.timed_ent_list[state.timed_ent[(b.name, a.name)]][exec_time] += 1

        if len(best_cost.migrated) > 1:

            self.add_gate(
                CatDisEnt(qargs=_e2 + [state.mapping[best_cost.migrated[1]]]),
                state.executed,
                state.times,
            )

            for i, (a, b) in enumerate(
                zip(
                    best_cost.migrations[best_cost.migrated[1]][:-1],
                    best_cost.migrations[best_cost.migrated[1]][1:],
                )
            ):

                exec_time = max([state.times[state.mapping[q]] for q in to_cover.qargs])

                if (
                    exec_time
                    not in state.timed_ent_list[state.timed_ent[(a.name, b.name)]]
                ):
                    state.timed_ent_list[state.timed_ent[(a.name, b.name)]].update(
                        {exec_time: 0}
                    )
                    state.timed_ent_list[state.timed_ent[(b.name, a.name)]].update(
                        {exec_time: 0}
                    )
                state.timed_ent_list[state.timed_ent[(a.name, b.name)]][exec_time] += 1
                state.timed_ent_list[state.timed_ent[(b.name, a.name)]][exec_time] += 1

    def comp_cover_succ(
        self,
        to_cover: DAGOpNode,
        best_cost: Cover,
        state: State,
        to_remove: Set[DAGOpNode],
        remote_ops_count: int,
        new_qargs: List[Qubit],
        e1: List[Qubit],
        _e1: List[Qubit],
        e2: List[Qubit],
        _e2: List[Qubit],
    ):
        """
        Check if cat-ent can cover more than one gate.

        Args:
            to_cover: the gate to cover
            best_cost: the cost of the cat-ent
            state: the current compilation state
            to_remove: a set of gates to remove form the gates' list, already covered by previous compilation steps
            remote_ops_count: the number of remote operations scheduled
            new_qargs: the qubits on which the covered gate will act
            e1: the ebits used for the cat-ent, node1 to node2
            _e1: the ebits used for the cat-ent, node1 to middle, only for migration on a middle node
            e2:the ebits used for the cat-ent, node2 to node1
            _e2: the ebits used for the cat-ent, node2 to middle, only for migration on a middle node
        """

        successors = list(self.dag_circuit.quantum_successors(to_cover))
        breaks = [False, False]

        if len(best_cost.migrated) == 1:
            breaks[1] = True

        while len(successors) != 0:
            remove = set()
            extra = None

            for g in successors:
                if g in to_remove:
                    continue

                if not isinstance(g, DAGOpNode):
                    remove.add(g)
                    continue

                if self.local_op(
                    [state.mapping[qarg] for qarg in g.qargs], state.nodes_mapping
                ):
                    if best_cost.migrated[0] in g.qargs:
                        breaks[0] = breaks[0] or True
                        if breaks[1] == True:
                            break
                        else:
                            continue
                    elif (
                        len(best_cost.migrated) > 1 and best_cost.migrated[1] in g.qargs
                    ):
                        breaks[1] = breaks[1] or True
                        if breaks[0] == True:
                            break
                        else:
                            continue
                    else:
                        remove.add(g)
                        continue

                if (
                    g.qargs[0] not in to_cover.qargs
                    and g.qargs[1] not in to_cover.qargs
                ):
                    remove.add(g)
                    continue

                if g not in state.all_covered:
                    if len(best_cost.migrated) > 1:
                        time2 = max([state.times[e] for e in _e1 + _e2])
                    else:
                        time2 = None
                    logger.debug(
                        f"GATE {g.op} {g.qargs} IS COVERED {self.is_covered(g, best_cost, state.times[e1[0]], state.times, state.all_covered, state.mapping, time2=time2)}\n"
                    )
                    mapping = state.mapping.copy()
                    # comp ent does not modify the mapping, so need to be careful to find extra gates...
                    # find migrations already changes the q_args using the mapping, using index may not be the best way, maybe change find_migrations to not do that...

                    if best_cost.migrated[0] in g.qargs:
                        mapping[
                            g.qargs[g.qargs.index(best_cost.migrated[0])]
                        ] = mapping[e2[-1]]
                    if (
                        len(best_cost.migrated) != 1
                        and best_cost.migrated[1] in g.qargs
                    ):
                        mapping[
                            g.qargs[g.qargs.index(best_cost.migrated[1])]
                        ] = mapping[_e2[-1]]
                    if self.is_covered(
                        g,
                        best_cost,
                        max([state.times[e] for e in e1 + e2]),
                        state.times,
                        state.all_covered,
                        mapping,
                        time2=time2,
                    ):
                        extra = g
                        remove.add(g)
                        state.covered.add(g)
                        state.all_covered.add(g)
                        to_remove.add(g)
                        # check extra q_args with mapping...
                        extra_qargs = []
                        for i in range(2):
                            if g.qargs[i] in to_cover.qargs:
                                extra_qargs.append(
                                    new_qargs[to_cover.qargs.index(g.qargs[i])]
                                )
                            else:
                                extra_qargs.append(mapping[g.qargs[i]])

                        self.add_gate(g, state.executed, state.times, qargs=extra_qargs)
                        remote_ops_count += 1
            if extra is None:
                break
            for r in remove:
                successors.remove(r)
            successors.extend(list(self.dag_circuit.quantum_successors(extra)))

    def comp_post_process(
        self,
        state: State,
        gates: Dict[DAGOpNode, None],
        to_remove: Set[DAGOpNode],
        uncovered: Set[DAGOpNode],
    ):
        """
        Update list of uncovered gates.

        Args:
            state: the current compilation state
            gates: the list of gates in the circuit
            to_remove: a list of gates to remove form the gates' list, already covered by previous compilation steps
            uncovered: the list of uncovered gates
        """

        for cov in state.covered:
            if cov in uncovered:
                uncovered.remove(cov)
        for e in to_remove:
            logger.debug(f"REMOVING {e}: {e.op} {e.qargs}")
            del gates[e]

    def comp_while(
        self,
        uncovered: Set[DAGOpNode],
        stop: int,
        gates: Dict[DAGOpNode, None],
        state: State,
        dag_circuit: DAGCircuit,
        max_cost: int,
        previous: List[State],
        compiled: State,
        mig_found: bool,
        post_tel: bool = False,
    ) -> State:
        """
        Compile portion of circuit with telegates.

        Args:
            uncovered: the list of uncovered gates
            stop: the number of gates we will try to cover
            gates: the list of gates in the circuit
            state: the current compilation state
            dag_circuit: the circuit to compile
            max_cost: the maximum cost allowed for covering a gate
            previous: a list of previously compiled states, possibly derived from previous evaluations during coverage of other gates
            compiled: the compilation state to update
            mig_found: signal if the coverage of at least one gate was possible
            post_tel: signal if we are trying to compile after a teleport operation

        Returns:
            the updated compilation state
        """

        while len(uncovered) > stop and len(gates) != 0:

            to_remove = set()
            best_cost = max_cost
            to_cover = None
            remote_ops_count = 0

            best_cost, to_cover = self.comp_for(
                gates,
                state,
                uncovered,
                to_remove,
                remote_ops_count,
                dag_circuit,
                to_cover,
                best_cost,
            )

            if to_cover is not None:
                new_qargs, e1, _e1, e2, _e2 = self.comp_ent(
                    state, to_cover, to_remove, best_cost
                )

                if new_qargs == None:
                    continue

                self.comp_dis_ent(best_cost, state, to_cover, e2, _e2)

                to_log = {
                    mig: [node.name]
                    for mig in best_cost.migrations
                    for node in best_cost.migrations[mig]
                }
                logger.debug(f"SELECTED MIG: {to_log}")
                if mig_found is False and len(previous) == 0:
                    compiled = deepcopy(state)
                    mig_found = True
                    if post_tel is True:
                        previous.append(deepcopy(state))
                else:
                    previous.append(deepcopy(state))

            logger.debug(f"TO REMOVE: \n{to_remove}")

            self.comp_post_process(state, gates, to_remove, uncovered)

        return state

    def compile(
        self,
        dag_circuit: DAGCircuit,
        state: State,
        stop_early: bool = False,
        previous: List[State] = None,
    ) -> Tuple[State, State, List[State]]:
        """
        Compile circuit with telegates.

        Args:
            dag_circuit: the circuit to compile
            state: the current compilation state
            stop_early: signal if we want to put a limit on the gates covered by a single cat-ent
            previous: a list of previously compiled states, possibly derived from previous evaluations during coverage of other gates

        Returns:
            the updated compilation state
        """

        if previous is None:
            previous = []
        gates, uncovered, previous_covered = self.comp_preprocess(
            dag_circuit, state, previous
        )

        max_cost = nx.diameter(self.network.graph) * len(uncovered)
        copy_barrier = []
        for item in state.executed:
            if item[0] == -1:
                copy_barrier.append(item)
            else:
                break
        compiled = State(executed=copy_barrier)
        mig_found = False
        if len(previous) != 0:
            state = previous[-1]
            mig_found = True
            compiled = deepcopy(previous[0])
            previous = previous[1:]
        else:
            previous = list()

        if stop_early:
            # TODO use a customizable parameter
            # stop = int(len(uncovered)*0.7)
            stop = (
                len(uncovered)
                + previous_covered
                - min(
                    [
                        (int((dag_circuit.num_qubits() / 2) * 5) - previous_covered),
                        len(uncovered),
                    ]
                )
            )
        else:
            stop = 0

        compiled = self.comp_while(
            uncovered,
            stop,
            gates,
            state,
            dag_circuit,
            max_cost,
            previous,
            compiled,
            mig_found,
        )

        return state, compiled, previous

    def tel_compile(
        self, dag_circuit: DAGCircuit, state: State, stop_early: bool = False
    ) -> Tuple[State, List[State]]:
        """
        Compile circuit with telegates, after a teleport operation.

        Args:
            dag_circuit: the circuit to compile
            state: the current compilation state
            stop_early: signal if we want to put a limit on the gates covered by a single cat-ent

        Returns:
            the updated compilation state
        """

        previous = list()
        gates, uncovered, previous_covered = self.comp_preprocess(
            dag_circuit, state, previous
        )

        max_cost = nx.diameter(self.network.graph) * len(uncovered)
        copy_barrier = []
        for item in state.executed:
            if item[0] == -1:
                copy_barrier.append(item)
            else:
                break

        compiled = State(executed=copy_barrier)
        mig_found = False

        if stop_early:
            # TODO use a customizable parameter
            # stop = int(len(uncovered)*0.7)
            stop = (
                len(uncovered)
                + previous_covered
                - min(
                    [
                        (int((dag_circuit.num_qubits() / 2) * 5) - previous_covered),
                        len(uncovered),
                    ]
                )
            )
        else:
            stop = 0

        compiled = self.comp_while(
            uncovered,
            stop,
            gates,
            state,
            dag_circuit,
            max_cost,
            previous,
            compiled,
            mig_found,
            post_tel=True,
        )

        return state, previous

    def cost(
        self,
        dag_circuit: DAGCircuit,
        gate: DAGOpNode,
        time: int,
        uncovered: Set[DAGOpNode],
        state: State,
    ) -> Cover:
        """
        Select best migration.

        Args:
            dag_circuit: the circuit to compile
            gate: the gate covered
            time: the time at which we would like to execute the gate
            uncovered: a list of uncovered gates
            state: the current compilation state

        Returns:
            the best coverage found
        """

        found_migrations = self.find_migrations(
            gate, state.nodes_mapping, state.mapping
        )
        logger.debug(f"EVALUATING {len(found_migrations)} MIGRATIONS")
        best_mig = None
        best_cost = None

        for mig in found_migrations:
            extra_covered = set()
            path = mig.migrations[mig.migrated[0]]
            cost = len(path) - 1
            time_to_wait = 1

            for a, b in zip(path[:-1], path[1:]):
                cap_used = sum(
                    [
                        state.timed_ent_list[state.timed_ent[(a.name, b.name)]][k]
                        for k in sorted(
                            state.timed_ent_list[
                                state.timed_ent[(a.name, b.name)]
                            ].keys()
                        )
                        if k <= time
                    ]
                    + [
                        state.timed_tele_list[state.timed_tele[(a.name, b.name)]][k]
                        for k in sorted(
                            state.timed_tele_list[
                                state.timed_tele[(a.name, b.name)]
                            ].keys()
                        )
                        if k <= time
                    ]
                )

                if cap_used + state.ebits_cap[(a.name, b.name)] <= 0:
                    for i in sorted(
                        state.timed_ent_list[state.timed_ent[(a.name, b.name)]].keys(),
                        reverse=True,
                    ):
                        if (
                            i >= time
                            and state.timed_ent_list[state.timed_ent[(a.name, b.name)]][
                                i
                            ]
                            > 0
                        ):
                            time_to_wait += i - time
                            break

                    for i in sorted(
                        state.timed_tele_list[
                            state.timed_tele[(a.name, b.name)]
                        ].keys(),
                        reverse=True,
                    ):
                        if (
                            i >= time
                            and state.timed_tele_list[
                                state.timed_tele[(a.name, b.name)]
                            ][i]
                            > 0
                            and (i - time + 1) < time_to_wait
                        ):
                            time_to_wait = i - time + 1
                            break

            temp_time_to_wait = time_to_wait

            if len(mig.migrated) > 1:
                path2 = mig.migrations[mig.migrated[1]]
                cost += len(path2) - 1

                for a, b in zip(path2[:-1], path2[1:]):
                    cap_used = sum(
                        [
                            state.timed_ent_list[state.timed_ent[(a.name, b.name)]][k]
                            for k in sorted(
                                state.timed_ent_list[
                                    state.timed_ent[(a.name, b.name)]
                                ].keys()
                            )
                            if k <= time
                        ]
                        + [
                            state.timed_tele_list[state.timed_tele[(a.name, b.name)]][k]
                            for k in sorted(
                                state.timed_tele_list[
                                    state.timed_tele[(a.name, b.name)]
                                ].keys()
                            )
                            if k <= time
                        ]
                    )

                    if cap_used + state.ebits_cap[(a.name, b.name)] <= 0:
                        for i in sorted(
                            state.timed_ent_list[
                                state.timed_ent[(a.name, b.name)]
                            ].keys(),
                            reverse=True,
                        ):
                            if (
                                i >= time
                                and state.timed_ent_list[
                                    state.timed_ent[(a.name, b.name)]
                                ][i]
                                > 0
                            ):
                                time_to_wait += i - time
                                break

                    if cap_used == state.ebits_cap[(a.name, b.name)]:
                        for i in sorted(
                            state.timed_tele_list[
                                state.timed_tele[(a.name, b.name)]
                            ].keys(),
                            reverse=True,
                        ):
                            if (
                                i >= time
                                and state.timed_tele_list[
                                    state.timed_tele[(a.name, b.name)]
                                ][i]
                                > 0
                                and (temp_time_to_wait + i - time) < time_to_wait
                            ):
                                time_to_wait = temp_time_to_wait + i - time
                                break

            rel_time = 0

            for u in dag_circuit.quantum_successors(gate):
                if isinstance(u, DAGOutNode):
                    continue

                if self.local_op(
                    [state.mapping[qarg] for qarg in u.qargs], state.nodes_mapping
                ):
                    if len(u.qargs) == 2 and mig.migrated[0] == u.qargs[-1]:
                        break
                    if len(u.qargs) == 1 and mig.migrated[0] == u.qargs[0]:
                        break
                    continue

                # only valid for cz, should break the link with cnots...
                if (
                    u in uncovered
                    and mig.migrated[0] in u.qargs
                    and state.nodes_mapping[
                        state.mapping[
                            list(set(u.qargs).difference([mig.migrated[0]]))[0]
                        ]
                    ]
                    == path[-1].name
                ):
                    valid = True
                    for p in dag_circuit.quantum_predecessors(u):
                        if p in extra_covered or not isinstance(p, DAGOpNode):
                            continue
                        if p not in state.covered and p != gate:
                            valid = False
                    if valid is True and time + rel_time + 1 <= time + MIG_EXP:
                        rel_time += 1
                        extra_covered.add(u)

            rel_time_2 = 0

            if len(mig.migrated) > 1:
                for u in dag_circuit.quantum_successors(gate):
                    if isinstance(u, DAGOutNode):
                        continue

                    if self.local_op(
                        [state.mapping[qarg] for qarg in u.qargs], state.nodes_mapping
                    ):
                        if len(u.qargs) == 2 and mig.migrated[1] == u.qargs[-1]:
                            break
                        if len(u.qargs) == 1 and mig.migrated[1] == u.qargs[0]:
                            break
                        continue

                    # only valid for cz, should break the link with cnots...
                    if (
                        u in uncovered
                        and mig.migrated[1] in u.qargs
                        and state.nodes_mapping[
                            state.mapping[
                                list(set(u.qargs).difference([mig.migrated[1]]))[0]
                            ]
                        ]
                        == path2[-1].name
                    ):
                        valid = True
                        for p in dag_circuit.quantum_predecessors(u):
                            if p in extra_covered or not isinstance(p, DAGOpNode):
                                continue
                            if p not in state.covered and p != gate:
                                valid = False
                        if valid is True and time + rel_time_2 + 1 <= time + MIG_EXP:
                            rel_time_2 += 1
                            extra_covered.add(u)

            rel_cost = cost

            if len(extra_covered) != 0:
                rel_cost /= len(extra_covered) + 1
            rel_cost *= time_to_wait

            if best_cost is None or rel_cost < best_cost:
                if len(mig.migrated) == 1:
                    best_mig = Cover(
                        mig.migrated[0], path, cost=cost, time=time + time_to_wait - 1
                    )
                else:
                    evaluated_mig = Cover(
                        mig.migrated[0], path, cost=cost, time=time + time_to_wait - 1
                    )
                    evaluated_mig.add(mig.migrated[1], path2)
                    best_mig = evaluated_mig

        return best_mig

    def find_migrations(
        self,
        gate: DAGOpNode,
        nodes_mapping: Dict[Qubit, str],
        mappings: Dict[Qubit, Qubit],
    ) -> List[Cover]:
        """
        Find possible migrations.

        Args:
            gate: the gate to cover
            nodes_mapping: the current nodes mapping
            mappings: the current mapping

        Returns:
            a list of possible covers
        """

        # check q_args mapping...
        logger.debug(f"{gate.op}: {gate.qargs} -> {[nodes_mapping]}")
        qargs = [mappings[qarg] for qarg in gate.qargs]
        p_start = nodes_mapping[qargs[0]]
        p_end = nodes_mapping[qargs[1]]
        migrations = []
        path = nx.shortest_path(
            self.network.graph, self.network.nodes[p_start], self.network.nodes[p_end]
        )
        migrations.append(Cover(gate.qargs[0], path))
        migrations.append(Cover(gate.qargs[1], path[::-1]))

        if len(path) == 2:
            return migrations

        for i, mid_point in enumerate(path[1:-1]):
            mid_migration = Cover(gate.qargs[0], path[: i + 2])
            mid_migration.add(gate.qargs[1], path[i + 1 :][::-1])
            migrations.append(mid_migration)

        return migrations

    def compile_multi_gate(
        self, multi_gate: DAGOpNode, state: State, old_qregs: Dict[Qubit, Qubit]
    ):
        """
        Compile everything in a multi-gate
        Multi-gate could have many non-local gates, but they shouldn't break each other

        Args:
            multi_gate: the multi-gate to compile
            state: the current state
            old_qregs: the old qregs mapping
        """
        op = multi_gate.op
        definition = op.definition
        decomposition = circuit_to_dag(definition)
        gates_list = decomposition.op_nodes()
        for gate in gates_list:
            gate.qargs = tuple(old_qregs[qarg] for qarg in gate.qargs)

        communication_used: dict[Qubit, list[Union[list, int]]] = {}
        uncovered: set[DAGOpNode] = set(gates_list)

        for gate in gates_list:
            if not self.local_op(
                [state.mapping[qarg] for qarg in gate.qargs], state.nodes_mapping
            ):
                for q in gate.qargs:
                    if q in communication_used.keys():
                        self.add_gate(
                            gate,
                            state.executed,
                            state.times,
                            qargs=[state.mapping[q] for q in gate.qargs],
                        )
                        uncovered.remove(gate)
                        communication_used[gate.qargs[0]][1] += 1
                        state.all_covered.add(gate)
                        break

                else:  # if doesn't break
                    time = max([state.times[state.mapping[q]] for q in gate.qargs]) + 1

                    best_cost = self.cost(
                        self.dag_circuit, gate, time, uncovered, state
                    )

                    new_qargs, e1, _e1, e2, _e2 = self.comp_ent(
                        state, gate, set(), best_cost
                    )

                    uncovered.remove(gate)
                    state.all_covered.add(gate)

                    communication_used[gate.qargs[0]] = [[e2, _e2, best_cost], 1]
            else:
                self.add_gate(
                    gate,
                    state.executed,
                    state.times,
                    qargs=[state.mapping[q] for q in gate.qargs],
                )
                uncovered.remove(gate)
                state.all_covered.add(gate)

        last_gate = gates_list[-1]
        for q in communication_used.keys():
            gates_dict = communication_used[q]
            e2, _e2, best_cost = gates_dict[0]
            self.comp_dis_ent(best_cost, state, last_gate, e2, _e2)
            self.non_local_gate_counter[gates_dict[1]] = (
                self.non_local_gate_counter.setdefault(gates_dict[1], 0) + 1
            )
