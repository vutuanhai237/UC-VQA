import numpy as np


def create_QFIM(psi: np.ndarray, grad_psi: np.ndarray):
    """Create Quantum Fisher Information matrix base on 
    \n https://quantum-journal.org/views/qv-2021-10-06-61/

    Args:
        - psi (np.ndarray): current state vector, is a N x 1 matrix
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix

    Returns:
        np.ndarray: N x N matrix
    """
    num_params = grad_psi.shape[0]
    # Calculate elements \bra\psi|\partial_k \psi\ket
    F_elements = np.zeros(num_params, dtype=np.complex128)
    for i in range(num_params):
        F_elements[i] = np.transpose(np.conjugate(psi)) @ (grad_psi[i])
    # Calculate F[i, j] = 4*Re*[\bra\partial_i \psi | \partial_j \psi \ket -
    # \bra\partial_i\psi | \psi\ket * \bra\psi|\partial_j \psi\ket]
    F = np.zeros([num_params, num_params])
    for i in range(0, num_params):
        for j in range(0, num_params):
            F[i, j] = 4 * np.real(
                np.transpose(np.conjugate(grad_psi[i])) @ (grad_psi[j]) -
                np.transpose(np.conjugate(F_elements[i])) * (F_elements[j]))
            if F[i, j] < 10**(-15):
                F[i, j] = 0
    return F
