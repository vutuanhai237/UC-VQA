import qiskit
import scipy
import qtm.constant
import numpy as np
import types
def extract_state(qc: qiskit.QuantumCircuit):
    """Get infomation about quantum circuit

    Args:
        - qc (qiskit.QuantumCircuit): Extracted circuit

    Returns:
       - tuple: state vector and density matrix
    """
    psi = qiskit.quantum_info.Statevector.from_instruction(qc)
    rho_psi = qiskit.quantum_info.DensityMatrix(psi)
    return psi, rho_psi


def trace_distance(rho, sigma):
    """Since density matrices are Hermitian, so trace distance is 1/2 (Sigma(|lambdas|)) with lambdas are the eigenvalues of (rho_psi - rho_psi_hat) matrix

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    w, _ = np.linalg.eig((rho - sigma).data)
    return 1 / 2 * sum(abs(w))


def trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    rho = rho.data
    sigma = sigma.data
    return np.trace(
        scipy.linalg.sqrtm(
            (scipy.linalg.sqrtm(rho)) @ (rho)) @ (scipy.linalg.sqrtm(sigma)))


def get_metrics(psi, psi_hat):
    """Get different metrics between the origin state and the reconstructed state

    Args:
        - psi (Statevector): first state vector
        - psi_hat (Statevector): second state vector

    Returns:
        - Tuple: trace and fidelity
    """
    rho = qiskit.quantum_info.DensityMatrix(psi)
    sigma = qiskit.quantum_info.DensityMatrix(psi_hat)
    return qtm.utilities.trace_distance(rho,
                                   sigma), qtm.utilities.trace_fidelity(rho, sigma)


def calculate_state_preparation_metrics(create_u_func: types.FunctionType, v: qiskit.QuantumCircuit, thetass, **kwargs):
    traces = []
    fidelities = []
    n = v.num_qubits
    for thetas in thetass:
        # Target state
        psi = qiskit.quantum_info.Statevector.from_instruction(v)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        # Preparation state
        u = qiskit.QuantumCircuit(n, n)
        u = create_u_func(u, thetas, **kwargs)
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(u)
        rho_psi_hat = qiskit.quantum_info.DensityMatrix(psi_hat)
        # Calculate the metrics
        trace, fidelity = qtm.utilities.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)
    return traces, fidelities


def calculate_tomography_metrics(u: qiskit.QuantumCircuit, create_vdagger_func: types.FunctionType, thetass, **kwargs):
    traces = []
    fidelities = []
    n = u.num_qubits   
    for thetas in thetass:
        psi = qiskit.quantum_info.Statevector.from_instruction(u)
        rho_psi = qiskit.quantum_info.DensityMatrix(psi)
        v = qiskit.QuantumCircuit(n, n)
        v = create_vdagger_func(v, thetas, **kwargs).inverse()
        psi_hat = qiskit.quantum_info.Statevector.from_instruction(v)
        # Calculate the metrics
        trace, fidelity = qtm.utilities.get_metrics(psi, psi_hat)
        traces.append(trace)
        fidelities.append(fidelity)

    return traces, fidelities