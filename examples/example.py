import sys
from pathlib import Path

from dqcnac.parser import InstrToBlock

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from os import path
from timeit import default_timer as timer

from qiskit import transpile
from qiskit.circuit import QuantumCircuit

from dqcnac.compiler import CompileManager
from dqcnac.network_configuration import simple_network
from dqcnac.stats import get_num_epr

device_type: str = sys.argv[1]

n_nodes: int = int(sys.argv[2])
network_topology: str = sys.argv[3]
network = simple_network(n_nodes, device_type, network_topology)

circuit_type: str = sys.argv[4]
n_qubits: int = int(sys.argv[5])

print("###### START ######")

try:
    circuit = QuantumCircuit.from_qasm_file(
        path.join(
            path.dirname(__file__),
            f"benchmark_circuits/{circuit_type}_n{n_qubits}.qasm",
        )
    )

except OSError:
    print("BAD INPUT CIRCUIT")
    print("###### STOP WITH ERROR ######")
    exit()

transpiled = transpile(
    circuit,
    basis_gates=["rx", "ry", "rz", "x", "y", "z", "h", "cz"],
    optimization_level=3,
)

compiler = CompileManager()

if sys.argv[6] == "True":
    use_tel = True
else:
    use_tel = False
if sys.argv[7] == "True":
    use_gate_grouping = True
else:
    use_gate_grouping = False
if len(sys.argv) >= 9:
    max_gates_in_a_group = int(sys.argv[8])
else:
    max_gates_in_a_group = 0
if len(sys.argv) >= 10 and sys.argv[9] == "True":
    check_commutations = True
else:
    check_commutations = False

if __name__ == "__main__":
    print("INPUT DEPTH: ", circuit.depth())
    start = timer()
    print_non_local_gates = True
    (
        compiled_circuit,
        network_to_local,
        network_layout,
        regs_mapping,
        programs,
        measured,
    ) = compiler.run(
        transpiled,
        network,
        use_tel,
        use_gate_grouping,
        max_gates_in_a_group,
        print_non_local_gates,
        check_commutations,
        parse=True,
    )
    stop = timer()
    print("OUTPUT DEPTH = ", compiled_circuit.depth())
    print("NUM EPR = ", get_num_epr(compiled_circuit))
    delta = stop - start
    print("Elapsed time [s]: ", delta)
    print("###### STOP ######")

    for p in programs:
        print(p, programs[p].serialize())
