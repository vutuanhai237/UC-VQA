
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
from DQASsearch import DQAS_search
from utils import set_op_pool
from vag import GHZ_vag 
from pennylane import numpy as np
import tensorflow as tf 
import qtm.constant


set_op_pool(qtm.constant.ghz_pool)
c = len(qtm.constant.ghz_pool)
p = 5

stp, nnp, history, circuit, cand_weight, cand_preset,cset, qcircuit = DQAS_search(
    GHZ_vag,
    nq=3,
    p=p,
    batch=10,
    epochs=3,
    verbose=False,
    nnp_initial_value=np.zeros([p, c]),
    structure_opt=tf.keras.optimizers.Adam(learning_rate=0.15),
)

