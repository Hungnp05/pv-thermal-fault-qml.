import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


# Apply CNOT gates between neighbouring qubits, closing the ring when topology is circular.
def entangle(n_qubits: int, topology: str) -> None:
    for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
    if topology == "circular" and n_qubits > 2:
        qml.CNOT(wires=[n_qubits - 1, 0])


# Build a variational circuit: angle encoding followed by trainable RY-RZ layers with entanglement.
def build_vqc_qnode(n_qubits: int, depth: int, topology: str, encoding: str):
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=encoding)
        for layer in range(depth):
            for wire in range(n_qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            entangle(n_qubits, topology)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    return circuit


class VQCClassifier(nn.Module):
    # Quantum layer that outputs one Pauli-Z expectation per qubit, followed by a linear map to class logits.
    def __init__(self, n_qubits: int, num_classes: int, depth: int, topology: str, encoding: str = "Y"):
        super().__init__()
        qnode = build_vqc_qnode(n_qubits, depth, topology, encoding)
        self.quantum = qml.qnn.TorchLayer(qnode, {"weights": (depth, n_qubits, 2)})
        self.output = nn.Linear(n_qubits, num_classes)

    # Map encoded features to class logits.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.quantum(x).float())


# Build a circuit that returns the statevector of the quantum feature map for one input.
def build_feature_map(n_qubits: int, depth: int, topology: str):
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def state(x):
        for _ in range(depth):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            entangle(n_qubits, topology)
        return qml.state()

    return state


# Compute the fidelity kernel |<phi(a)|phi(b)>|^2 between two sets of inputs from their simulated statevectors.
def fidelity_kernel(a: np.ndarray, b: np.ndarray, n_qubits: int, depth: int, topology: str) -> np.ndarray:
    state = build_feature_map(n_qubits, depth, topology)
    states_a = np.asarray(state(a))
    states_b = states_a if b is a else np.asarray(state(b))
    return np.abs(states_a @ states_b.conj().T) ** 2
