import types
import qiskit
import qtm.evolution
import qtm.random_circuit
import qtm.state
import qtm.qcompilation
import qtm.ansatz
class ECircuit():
    def __init__(self, qc: qiskit.QuantumCircuit, fitness_func: types.FunctionType) -> None:
        self.qc = qc
        self.fitness_func = fitness_func
        self.fitness = 0
        self.compile()
        return
    def compile(self):
        self.fitness = self.fitness_func(self.qc)
        return
