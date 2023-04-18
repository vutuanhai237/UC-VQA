import qtm.qcompilation
import numpy as np
import types
num_qubits = 3
num_layers = 1
thetas = np.ones(3*num_qubits*num_layers)


compiler = qtm.qcompilation.QuantumCompilation(
    u = qtm.ansatz.create_hypergraph_ansatz,
    vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
    optimizer = 'adam',
    loss_func = 'loss_basic',
    thetas = thetas,
    is_evolutional=True,
    num_layers = num_layers
)
compiler.fit(num_steps = 60, verbose = 1)