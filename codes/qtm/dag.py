
import qiskit
import scipy
import qtm.constant
import numpy as np
import types
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

look_up_operator = {
    "Identity": 'I',
    "Hadamard": 'H',
    "PauliX": 'X',
    'PauliY': 'Y',
    'PauliZ': 'Z',
    'S': 'S',
    'T': 'T',
    'SX': 'SX',
    'CNOT': 'CX',
    'CZ': 'CZ',
    'CY': 'CY',
    'SWAP': 'SWAP',
    'ISWAP': 'ISWAP',
    'CSWAP': 'CSWAP',
    'Toffoli': 'CCX',
    'RX': 'RX',
    'RY': 'RY',
    'RZ': 'RZ',
    'CRX': 'CRX',
    'CRY': 'CRY',
    'CRZ': 'CRZ',
    'U1': 'U1',
    'U2': 'U2',
    'U3': 'U3',
    'IsingXX': 'RXX',
    'IsingYY': 'RYY',
    'IsingZZ': 'RZZ',
}

def convert_string_to_int(string):
    return sum([ord(char) - 65 for char in string])


def circuit_to_dag(qc):
    """Convert a circuit to graph.
    Read more: 
    - https://qiskit.org/documentation/retworkx/dev/tutorial/dags.html
    - https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commutation_dag.html

    Args:
        qc (qiskit.QuantumCircuit): A qiskit quantum circuit

    Returns:
        DAG: direct acyclic graph
    """
    return qml.transforms.commutation_dag(qml.from_qiskit(qc))()


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.layers.append(GraphConvolution(hidden_dim, output_dim))

    def forward(self, x, adj):
        for layer in self.layers:
            x = F.relu(layer(x, adj))
        return x
    
def dag_to_node_features(dag):
    node_features = []
    for i in range(dag.size):
        node = dag.get_node(i)
        operation = qtm.dag.look_up_operator[node.op.base_name]
        params = node.op.parameters
        if len(params) == 0:
            params = [0]
        node_features.append([qtm.dag.convert_string_to_int(operation), *params])
    return np.array(node_features)

def dag_to_adjacency_matrix(dag):
    num_nodes = dag.size
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(dag.size):
        node = dag.get_node(i)
        for successor in node.successors:
            adjacency_matrix[i][successor] = 1
    return np.array(adjacency_matrix)


def graph_to_scalar(node_features, adjacency_matrix):
    num_nodes = node_features.shape[0]
    input_dim = node_features.shape[1]
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    # Convert node features and adjacency matrix to tensors
    node_features = torch.tensor(node_features, dtype=torch.float32)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    # Create GCN model
    gcn = qtm.dag.GCN(input_dim, hidden_dim, output_dim, num_layers)
    # Forward pass through the model
    graph_embedding = gcn(node_features, adjacency_matrix)
    # Apply global sum pooling to obtain a scalar representation
    # Apply global sum pooling to obtain a scalar representation
    graph_scalar = torch.sum(graph_embedding)
    # Sigmod activation to ranged value from 0 to 1
    return 1 / (1 + np.exp(-graph_scalar.item()))

def circuit_to_scalar(qc: qiskit.QuantumCircuit)->float:
    """Evaluate circuit

    Args:
        qc (qiskit.QuantumCircuit): encoded circuit

    Returns:
        float: Value from 0 to 1
    """
    dag = qtm.dag.circuit_to_dag(qc)
    node_features = qtm.dag.dag_to_node_features(dag)
    adjacency_matrix = qtm.dag.dag_to_adjacency_matrix(dag)
    return qtm.dag.graph_to_scalar(node_features, adjacency_matrix)