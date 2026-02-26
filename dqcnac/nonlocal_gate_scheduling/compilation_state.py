"""Compilation state class module"""

from copy import copy, deepcopy
from typing import List

from qiskit.circuit import Qubit

from ..network import Node


class State:
    """
    The class representing the state of the compilation.
    """

    def __init__(self, **kwargs):
        """
        Initialize the state of the compilation.

        Args:
            **kwargs: keyword arguments
        """

        for k in kwargs:
            if k == "timed_ent":
                self.timed_ent = kwargs["timed_ent"]
            elif k == "timed_tele":
                self.timed_tele = kwargs["timed_tele"]
            elif k == "ebits_cap":
                self.ebits_cap = kwargs["ebits_cap"]
            elif k == "executed":
                self.executed = kwargs["executed"]
            elif k == "times":
                self.times = kwargs["times"]
            elif k == "timed_ebits":
                self.timed_ebits = kwargs["timed_ebits"]
            elif k == "ebits":
                self.ebits = kwargs["ebits"]
            elif k == "covered":
                self.covered = kwargs["covered"]
            elif k == "all_covered":
                self.all_covered = kwargs["all_covered"]
            elif k == "mapping":
                self.mapping = kwargs["mapping"]
            elif k == "nodes_mapping":
                self.nodes_mapping = kwargs["nodes_mapping"]
            elif k == "initial_mapping":
                self.initial_mapping = kwargs["initial_mapping"]
            elif k == "idle_qubits":
                self.idle_qubits = kwargs["idle_qubits"]
            elif k == "timed_ent_list":
                self.timed_ent_list = kwargs["timed_ent_list"]
            elif k == "timed_tele_list":
                self.timed_tele_list = kwargs["timed_tele_list"]
            else:
                self.k = kwargs[k]

    # @profile
    def __deepcopy__(self, memo):
        kwargs = dict()

        kwargs["executed"] = self.executed.copy()
        kwargs["times"] = self.times.copy()
        kwargs["timed_tele"] = self.timed_tele
        kwargs["timed_tele_list"] = [d.copy() for d in self.timed_tele_list]
        kwargs["timed_ent"] = self.timed_ent
        kwargs["timed_ent_list"] = [d.copy() for d in self.timed_ent_list]

        # kwargs['timed_ebits'] = self.timed_ebits.copy()
        kwargs["ebits_cap"] = self.ebits_cap.copy()
        kwargs["covered"] = self.covered.copy()
        kwargs["all_covered"] = self.all_covered.copy()
        kwargs["ebits"] = self.ebits
        kwargs["mapping"] = self.mapping.copy()
        kwargs["nodes_mapping"] = self.nodes_mapping.copy()
        kwargs["initial_mapping"] = self.initial_mapping
        keys = list(self.idle_qubits.keys())
        kwargs["idle_qubits"] = {k: self.idle_qubits[k].copy() for k in keys}

        return State(**kwargs)

    def get_simple_copy(self) -> "State":
        """
        Get a shallow copy of the State object, i.e. different instance but same attributes

        Returns:
            a shallow copy of the State object
        """

        attributes = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attributes.update({a: copy(getattr(self, a))})
        return State(**attributes)

    def get_deep_copy(self) -> "State":
        """
        Get a deep copy of the State object, i.e. different instance and attributes

        Returns:
            a deep copy of the State object
        """

        attributes = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attributes.update({a: deepcopy(getattr(self, a))})
        return State(**attributes)


class Cover:
    """
    Class used to represent the coverage of a gate by "moving" one qubit of a gate over a path of nodes in the network.
    """

    def __init__(
        self, migrated: Qubit, path: List[Node], destination=None, cost=None, time=None
    ):
        """
        Initialize the Cover.

        Args:
            migrated: the qubit to "move"
            path: the path of nodes
            cost: the cost of the "movement"
            time: the time of the "movement"
        """

        self.migrated = [migrated]
        self.migrations = {migrated: path}
        self.cost = cost
        self.time = time
        self.destinations = {}
        if destination is not None:
            self.destinations[migrated] = destination

    def add(self, migrated: Qubit, path: List[Node], destination=None):
        """
        Add another qubit to the coverage, this may happen when "moving" both qubits of a gate to an intermediate node

        Args:
            migrated: the qubit to "move"
            path: the path of nodes
        """

        self.migrated.append(migrated)
        self.migrations[migrated] = path
        if destination is not None:
            self.destinations[migrated] = destination

    def set_path(self, migrated: Qubit, path: List[Node]):
        self.migrations[migrated] = path
