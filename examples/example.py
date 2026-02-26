import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from os import path
from timeit import default_timer as timer

from dqcnac.compiler import CompileManager
from dqcnac.network_configuration import simple_network
from dqcnac.stats import get_num_epr
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

device_type = sys.argv[1]

n_nodes = int(sys.argv[2])
network_topology = sys.argv[3]
network = simple_network(n_nodes, device_type, network_topology)

circuit_type = sys.argv[4]
n_qubits = int(sys.argv[5])

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

compiler = CompileManager(router=None)

if sys.argv[6] == "True":
    use_tel = True
else:
    use_tel = False
if sys.argv[7] == "True":
    use_pre_pass = True
else:
    use_pre_pass = False
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
    compiled_circuit, layout, network_layout, = compiler.run(
        transpiled,
        network,
        use_tel,
        use_pre_pass,
        max_gates_in_a_group,
        print_non_local_gates,
        check_commutations,
    )
    stop = timer()
    print("OUTPUT DEPTH = ", compiled_circuit.depth())
    print("NUM EPR = ", get_num_epr(compiled_circuit))
    delta = stop - start
    print("Elapsed time [s]: ", delta)
    print("###### STOP ######")
