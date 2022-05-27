import qtm.base
import qtm.optimizer
import qtm.loss
import qtm.utilities
import numpy as np
import typing, types
import qiskit
import matplotlib.pyplot as plt

class QuantumCompilation():
    def __init__(self) -> None:
        self.u = None
        self.vdagger = None
        self.is_trained = False
        self.optimizer = None
        self.loss_func = None
        self.thetas = None
        self.thetass = []
        self.loss_values = []
        self.fidelities = []
        self.traces = []
        self.kwargs = None
        return

    def __init__(self, u: typing.Union[types.FunctionType, qiskit.QuantumCircuit], vdagger: typing.Union[types.FunctionType, qiskit.QuantumCircuit], optimizer: typing.Union[types.FunctionType, str], loss_func: typing.Union[types.FunctionType, str], thetas: np.ndarray = np.array([]), **kwargs):
        """_summary_

        Args:
            - u (typing.Union[types.FunctionType, qiskit.QuantumCircuit]): In quantum state preparation problem, this is the ansatz. In tomography, this is the circuit that generate random Haar state.
            - vdagger (typing.Union[types.FunctionType, qiskit.QuantumCircuit]): In quantum tomography problem, this is the ansatz. In state preparation, this is the circuit that generate random Haar state.
            - optimizer (typing.Union[types.FunctionType, str]): You can put either string or function here. If type string, qcompilation produces some famous optimizers such as: 'sgd', 'adam', 'qng-fubini-study', 'qng-qfim', 'qng-adam'.
            - loss_func (typing.Union[types.FunctionType, str]): You can put either string or function here. If type string, qcompilation produces some famous optimizers such as: 'loss_basic'  (1 - p0) and 'loss_fubini_study' (\sqrt{(1 - p0)}).
            - thetas (np.ndarray, optional): initial parameters. Note that it must fit with your ansatz. Defaults to np.array([]).
        """
        self.set_u(u)
        self.set_vdagger(vdagger)
        self.set_optimizer(optimizer)
        self.set_loss_func(loss_func)
        self.set_kwargs(**kwargs)
        self.set_thetas(thetas)
        return

    def set_u(self, _u: typing.Union[types.FunctionType, qiskit.QuantumCircuit]):
        """In quantum state preparation problem, this is the ansatz. In tomography, this is the circuit that generate random Haar state.

        Args:
            - _u (typing.Union[types.FunctionType, qiskit.QuantumCircuit]): init circuit
        """
        if callable(_u) or isinstance(_u, qiskit.QuantumCircuit):
            self.u = _u
        else:
            raise ValueError('The U part must be a function f: thetas -> qiskit.QuantumCircuit or a determined quantum circuit')
        return

    def set_vdagger(self, _vdagger):
        """In quantum state tomography problem, this is the ansatz. In state preparation, this is the circuit that generate random Haar state.

        Args:
            - _vdagger (typing.Union[types.FunctionType, qiskit.QuantumCircuit]): init circuit
        """
        if callable(_vdagger) or isinstance(_vdagger, qiskit.QuantumCircuit):
            self.vdagger = _vdagger
        else:
            raise ValueError('The V dagger part must be a function f: thetas -> qiskit.QuantumCircuit or a determined quantum circuit')
        return

    def set_loss_func(self, _loss_func: typing.Union[types.FunctionType, str]):
        """Set the loss function for compiler

        Args:
            - _loss_func (typing.Union[types.FunctionType, str])

        Raises:
            ValueError: when you pass wrong type
        """
        if callable(_loss_func):
            self.loss_func = _loss_func
        elif isinstance(_loss_func, str):
            if _loss_func == 'loss-basic':
                self.loss_func = qtm.loss.loss_basis
            elif _loss_func == 'loss-fubini-study':
                self.loss_func = qtm.loss.loss_fubini_study
        else:
            raise ValueError('The loss function must be a function f: measurement value -> loss value or string in ["loss_basic", "loss_fubini_study"]')
        return

    def set_optimizer(self, _optimizer: typing.Union[types.FunctionType, str]):
        """Change the optimizer of the compiler

        Args:
            - _optimizer (typing.Union[types.FunctionType, str])

        Raises:
            ValueError: when you pass wrong type
        """
        if callable(_optimizer):
            self.optimizer = _optimizer
        elif isinstance(_optimizer,str):
            if _optimizer == 'sgd':
                self.optimizer = qtm.optimizer.sgd
            elif _optimizer == 'adam':
                self.optimizer = qtm.optimizer.adam
            elif _optimizer == 'qng-fubini-study':
                self.optimizer = qtm.optimizer.qng_fubini_study
            elif _optimizer == 'qng-qfim':
                self.optimizer = qtm.optimizer.qng_qfim
            elif _optimizer == 'qng-adam':
                self.optimizer = qtm.optimizer.qng_adam
        else:
            raise ValueError('The optimizer must be a function f: thetas -> thetas or string in ["sgd", "adam", "qng_qfim", "qng_fubini_study", "qng_adam"]')
        return
    
    def set_num_step(self, _num_step: int):
        """Set the number of iteration for compiler

        Args:
            - _num_step (int): number of iterations

        Raises:
            ValueError: when you pass a nasty value
        """
        if _num_step > 0 and isinstance(_num_step, int):
            self.num_step = _num_step
        else:
            raise ValueError('Number of iterations must be a integer, such that 10 or 100.')
        return

    def set_thetas(self, _thetas: np.ndarray):
        """Set parameter, it will be updated at each iteration

        Args:
            _thetas (np.ndarray): parameter for u or vdagger
        """
        if isinstance(_thetas, np.ndarray):
            self.thetas = _thetas
        else:
            raise ValueError('The parameter must be numpy array')
        return

    def set_kwargs(self, **kwargs):
        """Arguments supported for u or vdagger only. Ex: number of layer
        """
        self.__dict__.update(**kwargs)
        self.kwargs = kwargs
        return

    def fit(self, num_steps: int = 100, verbose: int = 0):
        """Optimize the thetas parameters

        Args:
            - num_steps: number of iterations
            - verbose (int, optional): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per 10 steps. Verbose 1 is good for timing training time, verbose 2 if you want to log loss values to a file. Please install package tdqm if you want to use verbose 1. 
        
        """
        self.thetass, self.loss_values = qtm.base.fit(
            self.u, self.vdagger, self.thetas, num_steps, self.loss_func, self.optimizer, verbose, is_return_all_thetas=True, **self.kwargs)
        self.is_trained = True
        if callable(self.u):
            self.traces, self.fidelities = qtm.utilities.calculate_state_preparation_metrics(self.u, self.vdagger, self.thetass, **self.kwargs)
        else:
            self.traces, self.fidelities = qtm.utilities.calculate_state_tomography_metrics(self.u, self.vdagger, self.thetass, **self.kwargs)
        return

    def save(self, metric: str = "", text = "", path = './', save_all: bool = False):
        """_summary_

        Args:
            - metric (str)
            - text (str): Defaults to './'. Additional file name string
            - path (str, optional): Defaults to './'.
            - save_all (bool, optional): Save thetass, fidelity, trace and loss_value if save_all = True

        Raises:
            ValueError: if save_all = False and metric is not right.
        """
        if save_all:
            np.savetxt(path + "/thetass" + text + ".csv", self.thetass, delimiter=",")
            np.savetxt(path + "/fidelities"+ text + ".csv", self.fidelities, delimiter=",")
            np.savetxt(path + "/traces" + text + ".csv", self.traces, delimiter=",")
            np.savetxt(path + "/loss_values" + text + ".csv", self.loss_values, delimiter=",")
        else:
            if metric == 'thetas':
                np.savetxt(path + "/thetass" + text + ".csv", self.thetass, delimiter=",")
            elif metric == 'fidelity':
                np.savetxt(path + "/fidelities" + text + ".csv", self.fidelities, delimiter=",")
            elif metric == 'trace':
                np.savetxt(path + "/traces" + text + ".csv", self.traces, delimiter=",")
            elif metric == 'loss_value':
                np.savetxt(path + "/loss_values" + text + ".csv", self.loss_values, delimiter=",")
            else:
                raise ValueError('The metric must be thetas, fidelity, trace or loss_value')
            print("Saved " + metric + " at " + path)
        return

    def reset(self):
        """Delete all current property of compiler
        """
        self.u = None
        self.vdagger = None
        self.is_trained = False
        self.optimizer = None
        self.loss_func = None
        self.num_step = 0
        self.thetas = None
        self.thetass = []
        self.loss_values = []
        return
