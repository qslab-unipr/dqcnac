"""Network class module."""

from typing import Optional

import networkx as nx

from .node import Node


class Network:
    """
    The network class representing a network of nodes and edges.
    """

    def __init__(self) -> None:
        """
        Initialize the network object
        """
        self.graph = nx.Graph()
        self.qubits_to_nodes = dict()
        self.n_qubits = 0
        self.nodes = {}
        self.distances = None

    def add_node(self, node: Node):
        """
        Add a node to the network.

        Args:
            node: the node to add.
        """

        self.graph.add_node(node, conn={})
        self.nodes.update({node.name: node})
        self.n_qubits += node.mem_cap

    def add_connection(
        self,
        node1: Node,
        node2: Node,
        properties: Optional[dict[str, list[int]]] = None,
    ):
        """
        Add a connection between two nodes.

        Args:
            node1: the first node.
            node2: the second node.
            properties: a dictionary of properties to add to the connection between node1 and node2.
        """

        self.graph.add_edge(node1, node2, properties=properties)
        distances = dict(nx.all_pairs_shortest_path_length(self.graph))
        self.distances = {n.name: {} for n in distances}
        for n1 in distances:
            for n2 in distances[n1]:
                self.distances[n1.name][n2.name] = distances[n1][n2]
        if properties is not None:
            self.nodes[node1.name].connections.update(
                {
                    node2.name: {
                        "ebits": properties[node1.name],
                        "cap": len(properties[node1.name]),
                    }
                }
            )
            self.graph.nodes[node1]["conn"].update(
                {
                    node2.name: {
                        "ebits": properties[node1.name],
                        "cap": len(properties[node1.name]),
                    }
                }
            )
            self.nodes[node2.name].connections.update(
                {
                    node1.name: {
                        "ebits": properties[node2.name],
                        "cap": len(properties[node2.name]),
                    }
                }
            )
            self.graph.nodes[node2]["conn"].update(
                {
                    node1.name: {
                        "ebits": properties[node2.name],
                        "cap": len(properties[node2.name]),
                    }
                }
            )
