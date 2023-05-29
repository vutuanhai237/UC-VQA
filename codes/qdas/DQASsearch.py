import tensorflow as tf 
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
Tensor= Any 
Array= Any 
Opt = Any 
from qdas.utils import get_op_pool,set_op_pool 
from pennylane import numpy as np
import pennylane as qml 

def preset_byprob(prob: Tensor) -> Sequence[int]:
    preset = []
    p = prob.shape[0]
    c = prob.shape[1]
    for i in range(p):
        j = np.random.choice(np.arange(c), p=np.array(prob[i]))
        preset.append(j)
    return preset

def void_generator() -> Iterator[Any]:
    while True:
        yield None

def get_preset(stp: Tensor) -> Tensor:
    return tf.argmax(stp, axis=1)

def get_weights(
    nnp: Tensor, stp: Tensor = None, preset: Optional[Sequence[int]] = None
) -> Tensor:
    """
    works only when nnp has the same shape as stp, i.e. one parameter for each op

    :param nnp:
    :param stp:
    :param preset:
    :return:
    """
    if preset is None:
        preset = get_preset(stp)
    p = nnp.shape[0]
    ind_ = tf.stack([tf.cast(tf.range(p), tf.int32), tf.cast(preset, tf.int32)])
    return tf.gather_nd(nnp, tf.transpose(ind_))

def DQAS_search(kernel_func: Callable[[Any, Tensor, Sequence[int]], Tuple[Tensor, Tensor]],
                op_pool: Optional[Sequence[Any]] = None,
                nq: Optional[int] = None,
                p: Optional[int] = None,
                p_nnp: Optional[int] = None,
                p_stp: Optional[int] = None,
                g: Optional[Iterator[Any]] = None,
                batch: int = 10,
                epochs: int = 100,
                verbose: bool = False,
                prob_clip: Optional[float] = None,
                baseline_func: Optional[Callable[[Sequence[float]], float]] = None,
                nnp_initial_value: Optional[Array] = None,
                stp_initial_value: Optional[Array] = None,
                network_opt: Optional[Opt] = None,
                structure_opt: Optional[Opt] = None,
                stp_regularization: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                nnp_regularization: Optional[Callable[[Tensor, Tensor], Tensor]] = None,):
    dtype = tf.float32
    if op_pool is None:
        op_pool = get_op_pool()
    c = len(op_pool)
    set_op_pool(op_pool)
    
    if g is None:
        g = void_generator()

    if network_opt is None:
        network_opt = tf.keras.optimizers.Adam(learning_rate=0.1)  # network

    if structure_opt is None:
        structure_opt = tf.keras.optimizers.Adam(learning_rate=0.1,beta_1=0.8,beta_2=0.99)
    
    if nnp_initial_value is None:
        if p_nnp is None:
            if p is not None:
                p_nnp = p
            else:
                raise ValueError("Please give the shape information on nnp")
        nnp_initial_value = np.random.uniform(size=[p_nnp, c])
    
    if stp_initial_value is None:
        if p_stp is None:
            if p is not None:
                p_stp = p
            else:
                raise ValueError("Please give the shape information on stp")
        stp_initial_value = np.zeros([p_stp, c])
    if p is None:
        p = stp_initial_value.shape[0]
    if baseline_func is None:
        baseline_func = np.mean
    nnp = tf.Variable(initial_value=nnp_initial_value, dtype=dtype)
    stp = tf.Variable(initial_value=stp_initial_value, dtype=dtype)

    history = []

    # prob =  tf.math.exp(stp) / tf.tile(
    #     tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:,tf.newaxis],[1,c]
    # )   
    avcost1 = 0 
    
    for epoch in range(epochs):
        prob =  tf.math.exp(stp) / tf.tile(
            tf.math.reduce_sum(tf.math.exp(stp), axis=1)[:,tf.newaxis],[1,c]
        )
        if prob_clip:
            prob = tf.clip_by_value(prob, (1 - prob_clip) / c, prob_clip)
            prob = prob / tf.tile(
                    tf.reshape(tf.reduce_sum(prob, axis=1), [prob.shape[0], 1]),
                    tf.constant([1, prob.shape[1]]),
                ) 
        if verbose:
            print("probability: \n", prob.numpy())
        print("----------new epoch %s-----------" % epoch)

        deri_stp = []
        deri_nnp = []

        avcost2 = avcost1 
        costl = []

        if stp_regularization is not None:
            stp_penalty_gradient = stp_regularization(stp, nnp)
            if verbose:
                print("stp_penalty_gradient:", stp_penalty_gradient.numpy())
        else:
            stp_penalty_gradient = 0.0
        if nnp_regularization is not None:
            nnp_penalty_gradient = nnp_regularization(stp, nnp)
            if verbose:
                print("nnpp_penalty_gradient:", nnp_penalty_gradient.numpy())

        else:
            nnp_penalty_gradient = 0.0
        for _, gdata in zip(range(batch), g):
            preset = preset_byprob(prob)

            loss, gnnp,circuit = kernel_func(gdata,nnp,preset)

            gs = tf.tensor_scatter_nd_add(tf.cast(-prob,dtype=dtype),
                                        tf.constant(list(zip(range(p),preset))),
                                        tf.ones([p],dtype=dtype))
            deri_stp.append(
                (tf.cast(loss,dtype=dtype) - tf.cast(avcost2,dtype=dtype))*tf.cast(gs,dtype=dtype)
            )
            deri_nnp.append(gnnp)
            costl.append(loss.numpy())
        avcost1 = tf.convert_to_tensor(baseline_func(costl))

        print(
            "batched average loss: ",
            np.mean(costl),
            " batched loss std: ",
            np.std(costl),
            "\nnew baseline: ",
            avcost1.numpy(), 
        )
        history.append(np.mean(costl))

        batched_gs = tf.math.reduce_mean(
            tf.convert_to_tensor(deri_stp,dtype=dtype),axis=0
        )
        batched_gnnp = tf.math.reduce_mean(
            tf.convert_to_tensor(deri_nnp,dtype=dtype),axis=0
        )
        if verbose:
            print("batched gradient of stp: \n", batched_gs.numpy())
            print("batched gradient of nnp: \n", batched_gnnp.numpy())

        network_opt.apply_gradients(
            zip([batched_gnnp+nnp_penalty_gradient],[nnp])
        )
        structure_opt.apply_gradients(
            zip([batched_gs+stp_penalty_gradient],[stp])
        )

        if verbose:
            print(
                "strcuture parameter: \n",
                stp.numpy(),
                "\n network parameter: \n",
                nnp.numpy(),
            )
        
        cand_preset = get_preset(stp).numpy()
        for i in cand_preset:
            print(op_pool[i])
        cset = get_op_pool()
        cand_weight = get_weights(nnp, stp).numpy()
        drawer = qml.draw(circuit)
        print(drawer(cand_weight,cand_preset,cset,nq))
    return stp, nnp, history, circuit
