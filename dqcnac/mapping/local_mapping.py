"""Local mapper with qiskit module."""
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import DenseLayout

EXTERNAL = {"Dense"}


class External:
    """
    Class to use external libraries for local mapping
    """

    def __init__(self, mapper: str, coupling_map: list):
        """
        Initialize the External mapper class

        Args:
            mapper: the mapper to be used
            coupling_map: the coupling map as a list of tuple, each tuple represents an edge between two integer vertices
        """
        if mapper not in EXTERNAL:
            raise Exception(f"External {mapper} not supported.")
        match mapper:
            case "Dense":
                self.mapper = DenseLayout(CouplingMap(coupling_map))

    def run(self, dag: DAGCircuit) -> Layout:
        """
        Run the External mapper

        Args:
            dag: the dag circuit to map

        Returns:
            a layout for the *dag* over a coupling map
        """
        self.mapper.run(dag)
        return self.mapper.property_set["layout"]
