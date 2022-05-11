import qtm.base
import qtm.optimizer
import qtm.loss
import numpy as np
import typing, types
import qiskit

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

    def set_u(self, _u):
        self.u = _u
        return

    def set_vdagger(self, _vdagger):
        self.vdagger = _vdagger
        return

    def set_loss_func(self, _loss_func):
        if callable(_loss_func):
            self.loss_func = _loss_func
        elif isinstance(_loss_func, str):
            if _loss_func == 'loss_basic':
                self.loss_func = qtm.loss.loss_basis
            if _loss_func == 'loss_fubini_study':
                self.loss_func = qtm.loss.loss_fubini_study
        return

    def set_optimizer(self, _optimizer):
        if callable(_optimizer):
            self.optimizer = _optimizer
        elif isinstance(_optimizer,str):
            if _optimizer == 'sgd':
                self.optimizer = qtm.optimizer.sgd
            if _optimizer == 'adam':
                self.optimizer = qtm.optimizer.adam
            if _optimizer == 'qng_fubini_study':
                self.optimizer = qtm.optimizer.qng_fubini_study
            if _optimizer == 'qng_qfim':
                self.optimizer = qtm.optimizer.qng_qfim
            if _optimizer == 'qng_adam':
                self.optimizer = qtm.optimizer.qng_adam
        return
    
    def set_num_step(self, _num_step):
        self.num_step = _num_step
        return

    def set_thetas(self, _thetas):
        self.thetas = _thetas
        return 

    def set_kwargs(self, **kwargs):
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
        return

    def reset(self):
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
