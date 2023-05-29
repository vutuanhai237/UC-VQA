from typing import (
    List,
    Sequence,
    Any,
    Tuple,
    Callable,
    Iterator,
    Optional,
    Union,
    Iterable,
    Dict,
)
Tensor = Any 
from pennylane import numpy as np
from qdas.utils import get_op_pool,set_op_pool
import pennylane as qml 
import tensorflow as tf
from backends import get_backend

npdtype = np.complex64
backend = get_backend("tensorflow")

def array_to_tensor(*num: np.array) -> Any:
    l = [backend.convert_to_tensor(n.astype(npdtype)) for n in num]
    if len(l) == 1:
        return l[0]
    return l


def GHZ_vag(gdata: Tensor,nnp: Tensor, preset: Sequence[int]
            , verbose: bool = False, n: int = 3) -> Tuple[Tensor, Tensor]:
    reference_state = np.zeros([2**n])
    reference_state[0] = 1 / np.sqrt(2)
    reference_state[-1] = 1 / np.sqrt(2)
    reference_state = tf.constant(reference_state,dtype=tf.complex64)
    nnp = nnp.numpy() 
    pnnp = [nnp[i,j] for i,j in enumerate(preset)]
    pnnp = array_to_tensor(np.array(pnnp))
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(pnnp,preset,cset,n):
        for i,j in enumerate(preset):
            gate = cset[j]
            if gate[0].startswith('R'):
                getattr(qml,gate[0])(pnnp[i],gate[1])
            elif gate[0] == 'Hadamard':
                getattr(qml,gate[0])(gate[1])
            elif gate[0] == "CNOT":
                qml.CNOT(wires=(gate[1], gate[2]))
            elif gate[0] == "Identity":
                continue
        return qml.state()
    cset = get_op_pool()
    with tf.GradientTape() as t:
        t.watch(pnnp)
        s = circuit(pnnp,preset,cset,n)
        # print("predict state:",s)
        s = tf.cast(s,dtype=tf.complex64)
        loss = tf.math.reduce_sum(tf.math.abs(s - reference_state))
    gr = t.gradient(loss, pnnp)
    if gr is None:
        gr = tf.zeros_like(pnnp)
    gr = backend.real(gr)
    gr = tf.where(tf.math.is_nan(gr), 0.0, gr)
    gmatrix = np.zeros_like(nnp)
    for i, j in enumerate(preset):
        gmatrix[i, j] = gr[i]
    gmatrix = tf.constant(gmatrix)
    return loss, gmatrix,circuit
    
if __name__ == '__main__':
    p=4
    ghz_pool = [
    ("RY", 0),
    ("RY", 1),
    ("RY", 2),
    ("CNOT", 0, 1),
    ("CNOT", 1, 0),
    ("CNOT", 0, 2),
    ("CNOT", 2, 0),
    ("H", 0),
    ("H", 1),
    ("H", 2),]
    c = len(ghz_pool)
    set_op_pool(ghz_pool)
    nnp = tf.Variable(np.random.uniform(size=[p, c]))
    preset = tf.constant([0,5,3,1])
    a,b = GHZ_vag(nnp,preset)
    print(a,b)