from julia import CubicMap, CubicNewtonMap
import numpy as np
from time import time

t_0 = time()
cnewt = CubicNewtonMap(
    # CubicMap(a=complex(2), b=complex(1))
    CubicMap(a=complex(0.473, -0.107), b=complex(0.167, 0.213))
)
print(time()-t_0)
print(cnewt.cubic.roots)

x_range = (-3, 3)
y_range = (-3, 3)

t_0 = time()
im = cnewt.draw_julia(res_x=600, res_y=600,
                      x_range=x_range, y_range=y_range, iterations=16)
print(time()-t_0)

t_0 = time()
im = cnewt.draw_ray(im=im, line_weight=1,
                    x_range=x_range, y_range=y_range,
                    angles=[2*np.pi/12*i for i in range(12)])
print(time()-t_0)

# t_0 = time()
# im = cnewt.draw_eqpot(im=im, line_weight=1,
#                      x_range = x_range, y_range = y_range,
#                      potentials = [0.1], res_eqpot = 4096)
# print(time()-t_0)

im.show()
