import numpy as np
from numpy import dtype

from tensor import Tensor

t = Tensor(np.array(5))
a = Tensor(5)
b = Tensor(5.)
c = Tensor(True)
d = Tensor(np.array("", dtype=dtype("object")))
