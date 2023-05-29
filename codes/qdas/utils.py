import sys 
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


thismodule = sys.modules[__name__]

_op_pool: Sequence[Any] = []

def set_op_pool(l: Sequence[Any]) -> None:
    # sometimes, to make parallel mode work, one should set_op_pool in global level of the script
    global _op_pool
    _op_pool = l

def get_op_pool() -> Sequence[Any]:
    global _op_pool
    return _op_pool