
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


def get_metrics(rho, sigma):
    """Get different metrics between the origin state and the reconstructed state

    Args:
        - psi (Statevector): first state vector
        - psi_hat (Statevector): second state vector

    Returns:
        - Tuple: trace and fidelity
    """
    return qtm.utilities.trace_distance(rho,
                                        sigma), qtm.utilities.trace_fidelity(rho, sigma)

def calculate_state_preparation_metrics_tiny2(create_u_func: types.FunctionType, additional_u, vdagger: qiskit.QuantumCircuit, thetas, **kwargs):
    n = vdagger.num_qubits
    # Target state
    rho = qiskit.quantum_info.DensityMatrix.from_instruction(vdagger.inverse())
    # Preparation state
    u = qiskit.QuantumCircuit(n, n)
    u = create_u_func(u, thetas, **kwargs).combine(additional_u)
    sigma = qiskit.quantum_info.DensityMatrix.from_instruction(u)
    # Calculate the metrics
    trace, fidelity = qtm.utilities.get_metrics(rho, sigma)
    return trace, fidelity

def calculate_state_preparation_metrics_tiny(create_u_func: types.FunctionType, vdagger: qiskit.QuantumCircuit, thetas, **kwargs):
    n = vdagger.num_qubits
    # Target state
    rho = qiskit.quantum_info.DensityMatrix.from_instruction(vdagger.inverse())
    # Preparation state
    u = qiskit.QuantumCircuit(n, n)
    u = create_u_func(u, thetas, **kwargs)
    sigma = qiskit.quantum_info.DensityMatrix.from_instruction(u)
    # Calculate the metrics
    trace, fidelity = qtm.utilities.get_metrics(rho, sigma)
    return trace, fidelity


def calculate_state_preparation_metrics(create_u_func: types.FunctionType, vdagger: qiskit.QuantumCircuit, thetass, **kwargs):
    traces = []
    fidelities = []
    n = vdagger.num_qubits
    for thetas in thetass:
        # Target state
        rho = qiskit.quantum_info.DensityMatrix.from_instruction(vdagger.inverse())
        # Preparation state
        u = qiskit.QuantumCircuit(n, n)
        u = create_u_func(u, thetas, **kwargs)
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(u)
        # Calculate the metrics
        trace, fidelity = qtm.utilities.get_metrics(rho, sigma)
        traces.append(trace)
        fidelities.append(fidelity)
    return traces, fidelities


def calculate_state_tomography_metrics(u: qiskit.QuantumCircuit, create_vdagger_func: types.FunctionType, thetass, **kwargs):
    traces = []
    fidelities = []
    n = u.num_qubits
    for thetas in thetass:
        rho = qiskit.quantum_info.DensityMatrix.from_instruction(u)
        v = qiskit.QuantumCircuit(n, n)
        v = create_vdagger_func(v, thetas, **kwargs).inverse()
        sigma = qiskit.quantum_info.DensityMatrix.from_instruction(v)
        # Calculate the metrics
        trace, fidelity = qtm.utilities.get_metrics(rho, sigma)
        traces.append(trace)
        fidelities.append(fidelity)

    return traces, fidelities


def haar_measure(n):
    """A Random matrix distributed with Haar measure

    Args:
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = (scipy.randn(n, n) + 1j*scipy.randn(n, n))/scipy.sqrt(2.0)
    q, r = scipy.linalg.qr(z)
    d = scipy.diagonal(r)
    ph = d/scipy.absolute(d)
    q = scipy.multiply(q, ph, q)
    return q


def normalize_matrix(matrix):
    """Follow the formula from Bin Ho

    Args:
        matrix (numpy.ndarray): input matrix

    Returns:
        numpy.ndarray: normalized matrix
    """
    return np.conjugate(np.transpose(matrix)) @ matrix / np.trace(np.conjugate(np.transpose(matrix)) @ matrix)


def is_pos_def(matrix, error=1e-8):
    return np.all(np.linalg.eigvalsh(matrix) > -error)


def is_normalized(matrix):
    print(np.trace(matrix))
    return np.isclose(np.trace(matrix), 1)
