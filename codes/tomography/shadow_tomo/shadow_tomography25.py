from itertools import combinations
import qiskit
import numpy as np
import tqix


num_qubits = 5
psi = [0.26424641, 0.23103536, 0.11177099, 0.17962657, 0.18777508, 0.07123707,
       0.20165063, 0.27101361, 0.21302122, 0.11930997, 0.09439792, 0.1763813,
       0.28546319, 0.0394065, 0.19575109, 0.09014811, 0.12315693, 0.03726953,
       0.10579994, 0.26516434, 0.21545716, 0.11265348, 0.20488736, 0.10268576,
       0.27819402, 0.0785904, 0.09997989, 0.17438181, 0.16625928, 0.23213874,
       0.01231226, 0.18198155]
rho = qiskit.quantum_info.DensityMatrix(psi).data
def self_tensor(matrix, n):
    product = matrix
    for i in range(1, n):
        product = np.kron(product, matrix)
    return product

def generate_u_pauli(num_qubits):
    lis = [0, 1, 2]
    coms = []
    if num_qubits == 2:
        for i in lis:
            for j in lis:
                coms.append([i, j])
    if num_qubits == 3:
        for i in lis:
            for j in lis:
                for k in lis:
                    coms.append([i, j, k])
    if num_qubits == 4:
        for i in lis:
            for j in lis:
                for k in lis:
                    for l in lis:
                        coms.append([i, j, k, l])
    if num_qubits == 5:
        for i in lis:
            for j in lis:
                for k in lis:
                    for l in lis:
                        for m in lis:
                            coms.append([i, j, k, l, m])
    sigma = [tqix.sigmax(), tqix.sigmay(), tqix.sigmaz()]
    Us = []
    for com in coms:
        U = sigma[com[0]]
        for i in range(1, num_qubits):
            U = np.kron(U, sigma[com[i]])
        Us.append(U)

    return Us[: 3**num_qubits]


def create_basic_vector(num_qubits: int):
    """Generate list of basic vectors

    Args:
        num_qubits (int): number of qubits

    Returns:
        np.ndarray: |00...0>, |00...1>, ..., |11...1>
    """
    bs = []
    for i in range(0, 2**num_qubits):
        b = np.zeros((2**num_qubits, 1))
        b[i] = 1
        bs.append(b)
    return bs


def calculate_sigma(U: np.ndarray, b: np.ndarray):
    """Calculate measurement values

    Args:
        U (np.ndarray): operator
        b (np.ndarray): basic vector

    Returns:
        np.ndarray: sigma operator
    """
    return (np.conjugate(np.transpose(U)) @ b @ np.conjugate(np.transpose(b)) @ U)

# def calculate_mu(density_matrix):
#     M = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
#     for i in range(0, num_observers):
#         for j in range(0, 2**num_qubits):
#             k = sigmass[i][j]
#             M += np.trace(k @ density_matrix) * k
#     M /= num_observers
#     return M


def calculate_mu_inverse(density_matrix, num_qubits):
    k = 3*density_matrix - \
        np.trace(density_matrix) * np.identity(2 **
                                               num_qubits, dtype=np.complex128)
    # M = k.copy()
    # for i in range(1, num_qubits):
    #     M = np.kron(M, k)
    return k





def shadow(num_experiments):

    num_observers = 3**num_qubits
    Us, bs = [], []
    bs = create_basic_vector(num_qubits)
    Us = generate_u_pauli(num_qubits)
    count_i = [0] * (num_observers)
    sum_b_s = [np.zeros((2**num_qubits, 2**num_qubits),
                        dtype=np.complex128)] * (num_observers)
    for i in range(0, num_experiments):
        r = np.random.randint(0, num_observers)
        count_i[r] += 1
        U = Us[r]
        sum_b = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        for j in range(0, 2**num_qubits):
            k = calculate_sigma(U, bs[j])
            sum_b_s[r] += np.trace(k @ rho)*calculate_mu_inverse(k, num_qubits)
            temp = sum_b_s[r].copy()
            sum_b_s[r] = (np.conjugate(np.transpose(
                temp)) @ temp) / (np.trace(np.conjugate(np.transpose(temp)) @ temp))

    ps = np.zeros(num_observers)
    rho_hat = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
    rho_hat_variant = 0
    for i in range(0, num_observers):
        ps[i] = count_i[i] / num_experiments
        traceA = np.trace(self_tensor(tqix.sigmaz(), num_qubits) @ sum_b_s[i])
        traceB = np.trace(self_tensor(tqix.sigmaz(), num_qubits) @ rho)
        rho_hat_variant += ps[i] * (traceA - traceB)**2
        rho_hat += ps[i] * sum_b_s[i]
        return rho_hat_variant, rho_hat


rho_hat_variantss = []
noe_large = [10**2, 10**3, 10**4, 10**5]
for noe in noe_large:
    rho_hat_variants = []
    for i in range(0, 10):
        rho_hat_variant, rho_hat = shadow(noe)
        rho_hat_variants.append(rho_hat_variant)
    rho_hat_variantss.append(rho_hat_variants)
np.savetxt("./rho_hat_variantss" + str(num_qubits) + ".csv",
           rho_hat_variantss,
           delimiter=",")

averages_var = [0]*4
averages_std = [0]*4
for i in range(len(noe_large)):
    averages_var[i] = np.mean(rho_hat_variantss[i])
    averages_std[i] = np.std(rho_hat_variantss[i])
print(averages_var)
print(averages_std)
