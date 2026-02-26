"""Local compilation manager module."""

from qiskit import QuantumRegister
from qiskit.circuit import Barrier, Qubit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Layout
from rustworkx import NoEdgeBetweenNodes

from ..local_routing import BasicRouter
from ..mapping.local_mapping import External
from ..network import Network
from ..nonlocal_gate_scheduling import CatDisEntInst, CatEntInst, TeleportInst


class LocalManager:
    """
    A class to manage local mapping and routing.
    """

    def __init__(
        self,
        distributed_circuit: DAGCircuit,
        network: Network,
        partitions: dict[Qubit, int],
        mapper: str = "dense",
        router: str = "basic",
    ):
        """
        Initialize the local manager

        Args:
            distributed_circuit: an already distributed circuit
            network: the network over which the circuit has been distributed
            partitions: a mapping from qubits to network nodes ids
            mapper: the local mapper to be used
            router: the local router to be used
        """
        self.mapper = mapper
        self.router = router
        self.distributed_circ = distributed_circuit
        self.network = network
        self.partitions = partitions

        self.initial_layout = Layout()

        # dist circuit has multiple quantum registers, router needs only one
        self.dist_to_canonical = dict()

        # dictionary mapping nodes to their coupling map
        self.couplings = dict()
        # dictionary mapping qubits in the network to qubits on the nodes
        self.network_to_local = {node: dict() for node in self.network.nodes}

        self._couplings = dict()

        self.nodes_ebits = {node: [] for node in self.network.nodes}
        # depending on configuration and inputs, compiler may not need or have access to all ebits of a device
        # some will remain unutilized / used by other programs
        self.extra_ebits = {node: [] for node in self.network.nodes}

        for node in self.network.nodes:
            for conn in self.network.graph.nodes[self.network.nodes[node]]["conn"]:
                self.nodes_ebits[node].extend(
                    self.network.graph.nodes[self.network.nodes[node]]["conn"][conn][
                        "ebits"
                    ]
                )

        for node in self.network.nodes:
            for ebit in self.network.nodes[node].ebits:
                if ebit not in self.nodes_ebits[node]:
                    self.extra_ebits[node].append(ebit)

        for node in self.network.nodes:
            local_coupling = CouplingMap()
            ebits_indices = set()

            extra_ebits_idx = []
            for q in self.network.nodes[node].coupling_map.physical_qubits:
                index = local_coupling.graph.add_node(q)

                if q in self.extra_ebits[node]:
                    extra_ebits_idx.append(index)

            # remove unutilized ebits from graph, otherwise router will see them and try tu use them
            for idx in extra_ebits_idx:
                local_coupling.graph.remove_node(idx)

            for edge in self.network.nodes[node].coupling_map.get_edges():
                if (
                    edge[0] in self.extra_ebits[node]
                    or edge[1] in self.extra_ebits[node]
                ):
                    continue

                if (
                    edge[0] in self.network.nodes[node].ebits
                    or edge[1] in self.network.nodes[node].ebits
                ):
                    if edge[0] in self.network.nodes[node].ebits:
                        ebits_indices.add(edge[0])
                    if edge[1] in self.network.nodes[node].ebits:
                        ebits_indices.add(edge[1])
                    self.add_edge(
                        local_coupling,
                        edge[0],
                        edge[1],
                        float(self.network.nodes[node].coupling_map.size() * 2),
                    )
                    self.add_edge(
                        local_coupling,
                        edge[1],
                        edge[0],
                        float(self.network.nodes[node].coupling_map.size() * 2),
                    )
                else:
                    self.add_edge(local_coupling, edge[0], edge[1], 1.0)
                    self.add_edge(local_coupling, edge[1], edge[0], 1.0)

            self.couplings.update({node: local_coupling})
            _local_coupling = CouplingMap()
            for q in local_coupling.physical_qubits:
                index = _local_coupling.graph.add_node(q)

            for edge in local_coupling.get_edges():
                _local_coupling.add_edge(*edge)

            for ebit in ebits_indices:
                neighbors = _local_coupling.graph.neighbors(ebit)

                for neighbor in neighbors:
                    while True:
                        try:
                            _local_coupling.graph.remove_edge(ebit, neighbor)
                            _local_coupling.graph.remove_edge(neighbor, ebit)
                        except NoEdgeBetweenNodes:
                            break

            self._couplings.update({node: _local_coupling})

        self.new_dag = DAGCircuit()
        self.canonical_reg = QuantumRegister(
            sum([len(self.couplings[node].graph.nodes()) for node in self.couplings]),
            name="q",
        )
        self.new_dag.add_qreg(self.canonical_reg)

        self.dist_to_canonical = {}
        _qubits = {node: [] for node in self.network.nodes}
        q = 0
        for dist_reg in self.distributed_circ.qregs.values():
            if "|" not in dist_reg.name:
                for qarg in dist_reg:
                    self.dist_to_canonical[qarg] = self.canonical_reg[q]
                    _qubits[self.partitions[qarg]].append(qarg)
                    q += 1

        coupling_map = CouplingMap()
        self.local_to_network = {node: dict() for node in self.couplings}
        self.ebits = {node: [] for node in self.couplings}
        self.qubits = {node: [] for node in self.couplings}
        # dictionary mapping circuit register to nodes
        self.reg_to_node = dict()
        ebits = list()

        # create network coupling map, keep map between local qubit indices and network indices
        _q = 0
        width = len(self.canonical_reg)
        for node in self.couplings:
            coupling = self.couplings[node]
            for edge in coupling.get_edges():
                _q0 = f"{edge[0]}_{node}"
                _q1 = f"{edge[1]}_{node}"

                if (
                    edge[0] in self.extra_ebits[node]
                    or edge[1] in self.extra_ebits[node]
                ):
                    continue

                if _q0 not in self.local_to_network[node]:
                    self.local_to_network[node].update({_q0: _q})
                    if edge[0] in self.nodes_ebits[node]:
                        self.ebits[node].append((_q0, _q))
                        ebits.append(_q)
                    else:
                        self.qubits[node].append((_q0, _q))

                    self.network_to_local[node].update(
                        {self.local_to_network[node][_q0]: edge[0]}
                    )

                    _q += 1

                if _q1 not in self.local_to_network[node]:
                    self.local_to_network[node].update({_q1: _q})
                    if edge[1] in self.nodes_ebits[node]:
                        self.ebits[node].append((_q1, _q))
                        ebits.append(_q)
                    else:
                        self.qubits[node].append((_q1, _q))
                    self.network_to_local[node].update(
                        {self.local_to_network[node][_q1]: edge[1]}
                    )

                    _q += 1

                self.add_edge(
                    coupling_map,
                    self.local_to_network[node][_q0],
                    self.local_to_network[node][_q1],
                    1.0,
                )

        # create global unique register, keep map between single registers and global register
        _ebits = {node: {} for node in self.network.nodes}
        for dist_reg in self.distributed_circ.qregs.values():
            if "|" in dist_reg.name:
                node = dist_reg.name.split("|")[1]

                for i, _e in enumerate(dist_reg):
                    self.initial_layout.add(
                        self.canonical_reg[q],
                        self.local_to_network[node][f"{self.ebits[node][i][0]}"],
                    )
                    self.dist_to_canonical[_e] = self.canonical_reg[q]
                    self.reg_to_node.update({self.canonical_reg[q]: node})
                    _ebits[node][_e] = self.canonical_reg[q]
                    q += 1

        self.coupling_map = CouplingMap()

        for qubit in coupling_map.physical_qubits:
            self.coupling_map.add_physical_qubit(qubit)

        # add ebits to coupling map
        # use big weight between ebits and other qubits
        # to avoid router to choose those edges for swaps
        for edge in coupling_map.get_edges():
            if edge[0] in ebits or edge[1] in ebits:
                self.add_edge(self.coupling_map, edge[0], edge[1], float(width + 1))
            else:
                self.add_edge(self.coupling_map, edge[0], edge[1], 1.0)

        _coup = CouplingMap()
        _coup.graph = self.coupling_map.graph.to_undirected()

        self.local_circuits = {node: DAGCircuit() for node in network.nodes}
        self.dist_to_local = {node: {} for node in self.network.nodes}
        self.local_to_dist = {node: {} for node in self.network.nodes}

        # for local mapping, create local circuits for each device
        # comprised of only local operations
        for node in self.network.nodes:
            qr = QuantumRegister(self.network.nodes[node].mem_cap, name="qr")
            self.local_circuits[node].add_qreg(qr)

            for qarg in qr:
                dist = _qubits[node].pop()
                self.dist_to_local[node][dist] = qarg
                self.local_to_dist[node][qarg] = dist

        for gate in self.distributed_circ.topological_op_nodes():
            if len(gate.qargs) > 1 and not isinstance(
                gate.op, (CatEntInst, CatDisEntInst, TeleportInst, Barrier)
            ):
                if (
                    "|" not in gate.qargs[0]._register.name
                    and "|" not in gate.qargs[1]._register.name
                ):
                    node0 = self.partitions[gate.qargs[0]]
                    node1 = self.partitions[gate.qargs[1]]
                    if node0 == node1:
                        self.local_circuits[node0].apply_operation_back(
                            gate.op,
                            qargs=(
                                self.dist_to_local[node0][gate.qargs[0]],
                                self.dist_to_local[node0][gate.qargs[1]],
                            ),
                        )

        self.local_layouts = {node: None for node in self.network.nodes}

        # do local mapping for each node
        for node in self.network.nodes:
            if self.mapper == "dense":
                # The CouplingMap class from Qiskit only accepts contiguously indexed nodes starting at 0.
                # The local coupling map nodes are not contiguously indexed, missing the ebits.
                # Here we create a temporary mapping to avoid issues
                mapper_to_local = dict()
                local_to_mapper = dict()
                locals = set()
                _coupling_list = list()
                i = 0
                for edge in self._couplings[node]:
                    if edge[0] not in locals:
                        locals.add(edge[0])
                        local_to_mapper[edge[0]] = i
                        mapper_to_local[i] = edge[0]
                        i += 1
                    if edge[1] not in locals:
                        locals.add(edge[1])
                        local_to_mapper[edge[1]] = i
                        mapper_to_local[i] = edge[1]
                        i += 1
                    _coupling_list.append(
                        [local_to_mapper[edge[0]], local_to_mapper[edge[1]]]
                    )

                local_mapper = External("Dense", _coupling_list)
            else:
                raise Exception(f"Mapper {self.mapper} not defined.")

            mapper_layout = local_mapper.run(self.local_circuits[node])
            local_layout = Layout(
                {
                    mapper_to_local[q]: mapper_layout[q]
                    for q in mapper_layout.get_physical_bits()
                }
            )
            self.local_layouts[node] = local_layout

            for p in local_layout.get_physical_bits():
                self.initial_layout.add(
                    self.dist_to_canonical[self.local_to_dist[node][local_layout[p]]],
                    self.local_to_network[node][f"{p}_{node}"],
                )
                self.reg_to_node.update(
                    {
                        self.dist_to_canonical[
                            self.local_to_dist[node][local_layout[p]]
                        ]: node
                    }
                )

        self.partitions_dim = {
            node1: len(self.network.nodes[node1].coupling_map.physical_qubits)
            - sum(
                [
                    self.network.nodes[node1].connections[node2]["cap"]
                    for node2 in self.network.nodes[node1].connections
                ]
            )
            for node1 in self.network.nodes
        }

    @staticmethod
    def add_edge(coupling_map: CouplingMap, src: int, dst: int, weight: float) -> None:
        """
        Add a directed weighted edge to the *coupling_map* between *src* and *dst*

        Args:
            coupling_map: a coupling map
            src: the source node
            dst: the destination node
            weight: the weight of the edge
        """
        if src not in coupling_map.physical_qubits:
            coupling_map.add_physical_qubit(src)

        if dst not in coupling_map.physical_qubits:
            coupling_map.add_physical_qubit(dst)
        coupling_map.graph.add_edge(src, dst, weight)

        coupling_map._dist_matrix = None
        coupling_map._is_symmetric = None
        coupling_map._size = None

    def run(
        self,
    ) -> tuple[DAGCircuit, dict[int, Qubit], dict[int, tuple[str, int]], dict]:
        """
        Run the local manager

        Returns: the compiled circuit, a mapping from the network physical qubits and circuit virtual qubits,
        a mapping from network physical qubits to physical qubits of each node coupling map
        """
        # recreate the circuit using the global register
        for layer in self.distributed_circ.serial_layers():
            for gate in layer["graph"].topological_op_nodes():

                new_qargs = [self.dist_to_canonical[qarg] for qarg in gate.qargs]

                if isinstance(gate.op, TeleportInst):
                    if (
                        self.reg_to_node[new_qargs[-1]]
                        != self.reg_to_node[new_qargs[-2]]
                    ):
                        raise Exception(
                            f"{new_qargs[-1]} not on same node as {new_qargs[-2]} in Teleport instruction: {self.reg_to_node[new_qargs[-1]]}- {self.reg_to_node[new_qargs[-2]]}"
                        )
                self.new_dag.apply_operation_back(gate.op, tuple(new_qargs))

        # do local routing
        final_layout = Layout()
        if self.router == "basic":
            compiled_circuit, final_layout = BasicRouter(
                self.coupling_map,
                [e[1] for ebits in self.ebits.values() for e in ebits],
                self.initial_layout,
                self.network_to_local,
                self.reg_to_node,
                self.couplings,
            ).run(self.new_dag)
        elif self.router is None:
            compiled_circuit = self.new_dag
        else:
            raise Exception(f"Router {self.router} not implemented.")

        i = self.network.graph.number_of_edges()
        for gate in list(compiled_circuit.op_nodes(Barrier)):
            if i == 0:
                break
            if isinstance(gate.op, Barrier) and len(gate.qargs) == 1:
                i -= 1
                compiled_circuit.remove_op_node(gate)

        # Initial barriers may be an issued here
        for wire in list(compiled_circuit.idle_wires()):
            compiled_circuit._remove_idle_wire(wire)

        compiled_dag = DAGCircuit()
        canonical_to_dist = {
            value: key for key, value in self.dist_to_canonical.items()
        }

        if len(compiled_circuit.cregs) != 0:
            for creg in compiled_circuit.cregs:
                compiled_dag.add_creg(creg)

        for gate in compiled_circuit.topological_op_nodes():
            op = gate.op
            qargs = gate.qargs
            new_qargs = [canonical_to_dist[qarg] for qarg in qargs]
            for qarg in new_qargs:
                if qarg._register.name not in compiled_dag.qregs:
                    compiled_dag.add_qreg(qarg._register)

            compiled_dag.apply_operation_back(op, qargs=new_qargs, cargs=gate.cargs)

        return (
            compiled_dag,
            {
                key: canonical_to_dist[value]
                for key, value in self.initial_layout.get_physical_bits().items()
            },
            {
                item[1]: (item[0].split("_")[-1], int(item[0].split("_")[0]))
                for node in self.local_to_network
                for item in self.local_to_network[node].items()
            },
            {
                key: canonical_to_dist[value]
                for key, value in final_layout.get_physical_bits().items()
            },
        )
