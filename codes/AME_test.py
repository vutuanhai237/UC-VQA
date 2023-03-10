import qiskit
import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.progress_bar
importlib.reload(qtm.base)
importlib.reload(qtm.constant)
import cmath
import qtm.qcompilation
import numpy as np
import types

num_qubits = 3
num_layers = 2
thetas = np.zeros(2*num_qubits*num_layers)

compiler = qtm.qcompilation.QuantumCompilation(
    u = qtm.ansatz.create_AMEchecker_polygongraph,
    vdagger = qtm.state.create_AME_state(num_qubits).inverse(),
    optimizer = 'adam',
    loss_func = 'loss_fubini_study',
    thetas = thetas,
    num_layers = num_layers
)
compiler.fit(num_steps = 10, verbose = 1)

# amplitude_state_1 = np.array([
#     0.27,
#     0.363,
#     0.326,
#     0,
#     0.377,
#     0,
#     0,
#     0.740*(np.cos(-0.79*np.pi)+1j*np.sin(-0.79*np.pi))])
# #0.740*np.exp(-0.79*np.pi*1j)
# amplitude_state_1 = amplitude_state_1/np.sqrt(sum(np.absolute(amplitude_state_1) ** 2))

# q = qiskit.QuantumRegister(3)
# qc = qiskit.QuantumCircuit(q)
# qc.prepare_state(amplitude_state_1, [q[0],q[1],q[2]])
# qc.draw('mpl')
