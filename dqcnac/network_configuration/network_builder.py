"""Simple network_configuration classes module."""

from copy import copy
from os import path
from typing import Dict, List

import yaml

from ..network import Network, Node


def load_yaml(filename: str) -> dict:
    """
    Load a yaml file and parser it into a dictionary

    Args:
        filename: the name of the file to parser

    Returns:
        a dictionary
    """

    with open(f"{filename}.yaml", "r") as stream:
        parsed_yaml = yaml.load(stream, Loader=yaml.FullLoader)
    return parsed_yaml


def simple_network(
    n_nodes: int, device_type: str, network_topology: str = "lnn"
) -> Network:
    """
    Create a simple network.

    Args:
        n_nodes: number of nodes in the network
        device: device type of devices in the network, either the name of one of the supported devices or a dictionary containing the device properties
        network_topology: the topology of the network

    Returns:
        a network
    """

    device_properties = load_yaml(
        path.join(path.dirname(__file__), "devices/", device_type)
    )
    network = Network()
    nodes = {}
    for n in range(n_nodes):
        bdir = []
        node = Node(
            str(n),
            device_properties["n_qubits"],
            device_properties["coupling"] + bdir,
            device_properties["ebits"],
            device_properties["cap"],
        )
        nodes.update({str(n): node})
        network.add_node(node)
    match network_topology:
        case "lnn":
            lnn(network, nodes, device_properties["ebits"], n_nodes)
        case "ring":
            ring(network, nodes, device_properties["ebits"], n_nodes)
        case "star":
            star(network, nodes, device_properties["ebits"], n_nodes)
        case "complete":
            all_to_all(network, nodes, device_properties["ebits"], n_nodes)

    return network


def lnn(network: Network, nodes: Dict[str, Node], ebits: List[int], n_nodes: int):
    """
    Create a network with a linear nearest neighbor.

    Args:
        network: the network
        nodes: the nodes in the network
        ebits: the ebits
        n_nodes: the number of nodes in the network
    """

    for n0 in range(n_nodes - 1):
        node0 = nodes[str(n0)]
        node1 = nodes[str(n0 + 1)]
        network.add_connection(
            node0, node1, properties={node0.name: copy(ebits), node1.name: copy(ebits)}
        )


def ring(network: Network, nodes: Dict[str, Node], ebits: List[int], n_nodes: int):
    """
    Create a network with a ring topology.

    Args:
        network: the network
        nodes: the nodes in the network
        ebits: the ebits
        n_nodes: the number of nodes in the network
    """

    for n0 in range(n_nodes - 1):
        node0 = nodes[str(n0)]
        node1 = nodes[str(n0 + 1)]
        network.add_connection(
            node0, node1, properties={node0.name: copy(ebits), node1.name: copy(ebits)}
        )

    node0 = nodes[str(n_nodes - 1)]
    node1 = nodes["0"]
    network.add_connection(
        node0, node1, properties={node0.name: copy(ebits), node1.name: copy(ebits)}
    )


def star(network: Network, nodes: Dict[str, Node], ebits: List[int], n_nodes: int):
    """
    Create a network with a star topology.

    Args:
        network: the network
        nodes: the nodes in the network
        ebits: the ebits
        n_nodes: the number of nodes in the network
    """

    node0 = nodes["0"]
    for n in range(1, n_nodes, 1):
        node = nodes[str(n)]
        network.add_connection(
            node0, node, properties={node0.name: copy(ebits), node.name: copy(ebits)}
        )


def all_to_all(
    network: Network, nodes: Dict[str, Node], ebits: List[int], n_nodes: int
):
    """
    Create a network with an all to all topology.

    Args:
        network: the network
        nodes: the nodes in the network
        ebits: the ebits
        n_nodes: the number of nodes in the network
    """

    for n0 in range(n_nodes):
        node0 = nodes[str(n0)]
        for n1 in range(n0 + 1, n_nodes, 1):
            node1 = nodes[str(n1)]
            network.add_connection(
                node0,
                node1,
                properties={node0.name: copy(ebits), node1.name: copy(ebits)},
            )
