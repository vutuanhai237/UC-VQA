from qsee.compilation.qsp import QuantumStatePreparation
from qsee.core import ansatz, state
from qsee.backend import constant, utilities
import matplotlib.pyplot as plt
import numpy as np, qiskit
from scipy.linalg import expm
from qiskit.quantum_info import Pauli, Statevector
from qsee.compilation.qsp import QuantumStatePreparation

# Define Pauli matrices
XX = Pauli("XX").to_matrix()
YY = Pauli("YY").to_matrix()
ZZ = Pauli("ZZ").to_matrix()
ZI = Pauli("ZI").to_matrix()
IZ = Pauli("IZ").to_matrix()
# Set coefficients
J = 1.0
D = 0.0
h = 0.0
# Define H
H = J * (XX + YY) + D * ZZ + h * (ZI - IZ)
# define V(t), calculate U and local magnetization
time = np.linspace(0, 10, 10)
magnetization = []
for t in time:
    V = expm(-1j * t * H)
    compiler = QuantumStatePreparation(u=ansatz.zxz_WchainCNOT(2), target_state=V).fit(num_steps = 30)
    statevector = Statevector.from_instruction(
        compiler.u.assign_parameters(compiler.thetas)
    )
    # print(V)
    # print(V)
    print(compiler.fidelity)
    # print(compiler.thetas)
    psi_t = np.array(statevector.data).reshape(-1, 1)
    psi_t_dag = np.conj(psi_t.T)
    mag = (psi_t_dag @ ZI @ psi_t)[0][0] - (psi_t_dag @ IZ @ psi_t)[0][0]
    magnetization.append(0.5 * mag)

plt.plot(time, magnetization)
