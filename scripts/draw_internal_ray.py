from julia import CubicMap, CubicNewtonMap
from time import time
import numpy as np
import matplotlib.pyplot as plt
import cmath
from itertools import product

cubic = CubicMap(a=complex(2), b=complex(2))
newton = CubicNewtonMap(cubic)
newton.draw_ray(res_x=1024, res_y=1024,
                root=cubic.roots[0]).show()
