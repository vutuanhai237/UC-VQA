import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding
import importlib
importlib.reload(qtm.base)
importlib.reload(qtm.constant)
importlib.reload(qtm.onequbit)
importlib.reload(qtm.nqubit)
importlib.reload(qtm.fubini_study)

num_qubits = 3
num_layers = 1
thetas = np.random.random(num_layers*num_qubits*4)
psi = 2*np.random.rand(2**num_qubits)-1
psi = psi / np.linalg.norm(psi)
encoder = qtm.encoding.Encoding(psi, 'amplitude_encoding')
qc = qiskit.QuantumCircuit(num_qubits, num_qubits)

loss_values = []
thetass = []
for i in range(0, 20):

    print('W_chain: ', i)
    grad_loss = qtm.base.grad_loss(
        qc, 
        qtm.nqubit.create_Wchainchecker_haar,
        thetas, r = 1/2, s = np.pi/2, num_layers = num_layers, encoder = encoder)
    thetas -= qtm.constant.learning_rate*(grad_loss) 
    qc_copy = qtm.nqubit.create_Wchainchecker_haar(qc.copy(), thetas, num_layers, encoder)  
    loss = qtm.base.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
    loss_values.append(loss)
    thetass.append(thetas)

np.savetxt("../../tomography_wchain/" + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
np.savetxt("../../tomography_wchain/" + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")