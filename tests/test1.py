import pytest
import person



def testa():
    assert 5 == person.sum(1, 4)

def test_less():
    num = 100
    assert num < 200
