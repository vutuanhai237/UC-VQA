import qiskit
import numpy as np
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.nqubit, qtm.fubini_study, qtm.encoding
num_qubits = 4
num_layers = 3
thetas = np.ones(num_layers*num_qubits*4)
psi = 2*np.random.rand(2**num_qubits)-1
psi = psi / np.linalg.norm(psi)
qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
qc.initialize(psi, range(0, num_qubits))

loss_values = []
thetass = []

for i in range(0, 400):
    if i % 20 == 0:
        print('W_chain: (' + str(num_layers) + ',' + str(num_qubits) + '): ' + str(i))

    grad_loss = qtm.base.grad_loss(
        qc, 
        qtm.nqubit.create_Wchain_layerd_state,
        thetas, r = 1/2, s = np.pi/2, num_layers = num_layers)
    thetas -= qtm.constant.learning_rate*(grad_loss) 
    thetass.append(thetas.copy())
    qc_copy = qtm.nqubit.create_Wchain_layerd_state(qc.copy(), thetas, num_layers)  
    loss = qtm.base.loss_basis(qtm.base.measure(qc_copy, list(range(qc_copy.num_qubits))))
    loss_values.append(loss)

traces = []
fidelities = []

for thetas in thetass:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = qtm.nqubit.create_Wchain_layerd_state(qc, thetas, num_layers = num_layers).inverse()
    psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc)
    # Calculate the metrics
    trace, fidelity = qtm.base.get_metrics(psi, psi_hat)
    traces.append(trace)
    fidelities.append(fidelity)
np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + '/' + str(num_qubits) + "/loss_values.csv", loss_values, delimiter=",")
np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + '/' + str(num_qubits) + "/thetass.csv", thetass, delimiter=",")
np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + '/' + str(num_qubits) + "/traces.csv", traces, delimiter=",")
np.savetxt("../../experiments/tomography/tomography_wchain_" + str(num_layers) + '/' + str(num_qubits) + "/fidelities.csv", fidelities, delimiter=",")