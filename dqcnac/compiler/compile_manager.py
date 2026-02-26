"""Compilation manager module."""

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from ..mapping import circuit_partitioning
from ..network import Network
from ..nonlocal_gate_scheduling import NonlocalGateSchedule, state_to_dag
from .gate_grouping import GateGrouping
from .local_manager import LocalManager


class CompileManager:
    """
    Class to manage compilation workflow
    """

    def __init__(
        self,
        partitioner: str = "metis_based",
        scheduler: str = "basic",
        mapper: str = "dense",
        router: str = None,
    ) -> None:
        """
        Initialize the CompilerManager

        Args:
            partitioner: the name of the partitioner to be used
            scheduler: the name of the remote operation scheduler to be used
            mapper: the name of the local mapper to be used
            router: the name of the local router to be used
        """
        self.partitioner = partitioner
        self.scheduler = scheduler
        self.mapper = mapper
        self.router = router

    def run(
        self,
        circuit: QuantumCircuit,
        network: Network,
        use_tel: bool = True,
        use_gate_grouping=True,
        max_gates_in_a_group: int = 0,
        print_non_local_gates=False,
        check_commutations=True,
    ) -> tuple[QuantumCircuit, dict[Qubit, tuple[str, int]], dict]:

        """
        The run method of the CompilerManager

        Args:
            circuit: a monolithic circuit to distributed
            network: the network over which the *circuit* must be distributed
            use_tel: whether to use also TeleData or only TeleGates
            use_gate_grouping: whether to use pre_pass to optimize entanglement usage
            max_gates_in_a_group: if not 0, the max number of gates allowed in a group if pre_pass is True
            print_non_local_gates: whether to print information about non-local gates
            check_commutations: whether to check for swaps between gates

        Returns:
            the distributed circuit, a mapping from the network physical qubits and circuit virtual qubits,
            a mapping from network physical qubits to physical qubits of each node coupling map
        """
        if self.partitioner == "metis_based":
            partitions = circuit_partitioning(
                network, circuit, None, use_gate_grouping, max_gates_in_a_group
            )
        else:
            raise Exception(f"Partitioner {self.partitioner} not defined.")

        if self.scheduler == "basic":
            scheduler = NonlocalGateSchedule(network, partitions, use_teleport=use_tel)
        else:
            raise Exception(f"Scheduler {self.scheduler} not defined.")

        if use_gate_grouping:
            pre_pass = GateGrouping(
                circuit,
                partitions,
                network,
                max_gates_in_a_group=max_gates_in_a_group,
                check_swaps=check_commutations,
            )
            pre_passed_circuit = pre_pass.run()
            state, regs_mapping = scheduler.run(
                circuit_to_dag(pre_passed_circuit),
                use_gate_grouping,
                print_non_local_gates,
            )
        else:
            state, regs_mapping = scheduler.run(
                circuit_to_dag(circuit), use_gate_grouping
            )

        dag = state_to_dag(state)

        local_manager = LocalManager(
            dag, network, state.initial_mapping, mapper=self.mapper, router=self.router
        )

        compiled_circuit, layout, network_layout, final_layout = local_manager.run()

        compiled_circuit = dag_to_circuit(compiled_circuit)

        network_to_local = {v: network_layout[k] for k, v in layout.items()}

        return compiled_circuit, network_to_local, regs_mapping
