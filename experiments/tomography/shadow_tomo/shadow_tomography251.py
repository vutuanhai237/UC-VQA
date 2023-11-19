
import sys
from itertools import combinations
import qiskit
import numpy as np
import tqix


sys.path.insert(1, '../../')
import qsee.measure
import qsee.ansatz
import qsee.gradient

def self_tensor(matrix, n):
    product = matrix
    for i in range(1, n):
        product = np.kron(product, matrix)
    return product
    
num_qubits = 5
psi = [0.26424641, 0.23103536, 0.11177099, 0.17962657, 0.18777508, 0.07123707,
       0.20165063, 0.27101361, 0.21302122, 0.11930997, 0.09439792, 0.1763813,
       0.28546319, 0.0394065, 0.19575109, 0.09014811, 0.12315693, 0.03726953,
       0.10579994, 0.26516434, 0.21545716, 0.11265348, 0.20488736, 0.10268576,
       0.27819402, 0.0785904, 0.09997989, 0.17438181, 0.16625928, 0.23213874,
       0.01231226, 0.18198155]

num_layers = 2
thetas = np.ones(num_layers*num_qubits*4)

qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
qc.initialize(psi, range(0, num_qubits))



for i in range(0, 400):
    grad_loss = qsee.measure.grad_loss(
        qc,
        qsee.ansatz.create_Wchain_layerd_state,
        thetas, r=1/2, s=np.pi/2, num_layers=num_layers)
    if i == 0:
        m, v = list(np.zeros(thetas.shape[0])), list(
            np.zeros(thetas.shape[0]))
    thetas = qsee.optimizer.adam(thetas, m, v, i, grad_loss)
    thetass.append(thetas.copy())
    qc_copy = qsee.ansatz.create_Wchain_layerd_state(
        qc.copy(), thetas, num_layers)
    loss = qsee.loss.loss_basis(qsee.measure.measure(
        qc_copy, list(range(qc_copy.num_qubits))))
    loss_values.append(loss)
variances = []
for thetas in thetass:
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    qc = qsee.ansatz.create_Wchain_layerd_state(
        qc, thetas, num_layers=num_layers).inverse()
    psi_hat = qiskit.quantum_info.Statevector.from_instruction(qc).data
    variances.append((np.conjugate(np.transpose(psi_hat)) @ self_tensor(tqix.sigmaz(), num_qubits) @ psi_hat)
                     ** 2 - (np.conjugate(np.transpose(psi)) @ self_tensor(tqix.sigmaz(), num_qubits) @ psi)**2)


np.savetxt("./thetass" + str(num_qubits) + ".csv",
           thetass,
           delimiter=",")
np.savetxt("./variances" + str(num_qubits) + ".csv",
           variances,
           delimiter=",")

print(min((abs(x), x) for x in variances)[1])
print(variances[-1])
