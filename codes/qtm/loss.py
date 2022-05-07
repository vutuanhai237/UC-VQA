import numpy as np

def loss_basis(measurement_value):
    """Return loss value for loss function L = 1 - P_0
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (float): P_0 value

    Returns:
        - float: loss value
    """
    return 1 - measurement_value


def loss_fubini_study(measurement_value):
    """Return loss value for loss function C = (1 - P_0)^(1/2)
    \n Here P_0 ~ 1 or L ~ 0 will be the best value

    Args:
        - measurement_value (float): P_0 value

    Returns:
        - float: loss value
    """
    return np.sqrt(1 - measurement_value)