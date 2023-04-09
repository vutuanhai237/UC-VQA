import qiskit
import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys
sys.path.insert(1, '../')
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.progress_bar
importlib.reload(qtm.base)
importlib.reload(qtm.constant)
import cmath
import qtm.qcompilation
import numpy as np
import types
import qtm.base, qtm.constant, qtm.ansatz, qtm.fubini_study, qtm.progress_bar
importlib.reload(qtm.base)
importlib.reload(qtm.constant)
import cmath
import qtm.qcompilation
import numpy as np
import types
import pickle 

from qiskit.quantum_info import Statevector
from qiskit import QuantumRegister, QuantumCircuit, Aer
import qiskit
v = qtm.state.create_AME_state(3)
probs = qtm.utilities.concentratable_entanglement(v,exact=True)
print(probs)
