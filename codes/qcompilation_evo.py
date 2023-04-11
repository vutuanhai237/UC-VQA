import qtm.qcompilation
import numpy as np
import types
num_qubits = 3
num_layers = 1
thetas = np.ones(num_qubits*num_layers*5)

compiler = qtm.qcompilation.QuantumCompilation(
    u = qtm.ansatz.create_linear_ansatz,
    vdagger = qtm.state.create_ghz_state(num_qubits).inverse(),
    optimizer = 'adam',
    loss_func = 'loss_basic',
    thetas = thetas,
    num_layers = num_layers,
    is_evolutional=True
)
compiler.fit(num_steps = 5, verbose = 1)