import numpy as np
import qtm.constant
def sgd(thetas: np.ndarray, grad_loss: np.ndarray):
    """Standard gradient descent

    Args:
        - thetas (np.ndarray): parameters
        - grad_loss (np.ndarray): gradient value

    Returns:
        - np.ndarray: new params
    """
    thetas -= qtm.constant.learning_rate * grad_loss
    return thetas


def adam(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, iteration: int,
         grad_loss: np.ndarray):
    """Adam Optimizer. Below codes are copied from somewhere :)

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    num_thetas = thetas.shape[0]
    beta1, beta2, epsilon = 0.8, 0.999, 10**(-8)
    for i in range(0, num_thetas):
        m[i] = beta1 * m[i] + (1 - beta1) * grad_loss[i]
        v[i] = beta2 * v[i] + (1 - beta2) * grad_loss[i]**2
        mhat = m[i] / (1 - beta1**(iteration + 1))
        vhat = v[i] / (1 - beta2**(iteration + 1))
        thetas[i] -= qtm.constant.learning_rate * mhat / (np.sqrt(vhat) +
                                                          epsilon)
    return thetas


def qng_fubini_study(thetas: np.ndarray, G: np.ndarray, grad_loss: np.ndarray):
    """_summary_

    Args:
        - thetas (np.ndarray): parameters
        - G (np.ndarray): Fubini-study matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    thetas = np.real(thetas - qtm.constant.learning_rate *
                     (np.linalg.inv(G) @ grad_loss))
    return thetas


def qng_qfim(thetas: np.ndarray, psi: np.ndarray, grad_psi: np.ndarray,
             grad_loss: np.ndarray):
    """Update parameters based on quantum natural gradient algorithm
    \n thetas^{i + 1} = thetas^{i} - alpha * F^{-1} * nabla L

    Args:
        - thetas (np.ndarray): parameters
        - psi (np.ndarray): current state 
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    F = qtm.qfim.create_QFIM(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.inv(F)
    thetas -= qtm.constant.learning_rate * (inverse_F @ grad_loss)
    return thetas


def qng_adam(thetas: np.ndarray, m: np.ndarray, v: np.ndarray, i: int,
             psi: np.ndarray, grad_psi: np.ndarray, grad_loss: np.ndarray):
    """After calculating the QFIM, use it in Adam optimizer

    Args:
        - thetas (np.ndarray): parameters
        - m (np.ndarray): params for Adam
        - v (np.ndarray): params for Adam
        - i (int): params for Adam
        - psi (np.ndarray): current state 
        - grad_psi (np.ndarray): all partial derivatives of $\psi$, is a N x N matrix
        - grad_loss (np.ndarray): gradient of loss function, is a N x 1 matrix

    Returns:
        - np.ndarray: parameters after update
    """
    F = qtm.qfim.create_QFIM(psi, grad_psi)
    # Because det(QFIM) can be nearly equal zero
    if np.isclose(np.linalg.det(F), 0):
        inverse_F = np.identity(F.shape[0])
    else:
        inverse_F = np.linalg.inv(F)

    grad = inverse_F @ grad_loss
    thetas = qtm.optimizer.adam(thetas, m, v, i, grad)
    return thetas