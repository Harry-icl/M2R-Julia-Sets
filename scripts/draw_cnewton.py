from julia import CubicMap, CubicNewtonMap
import numpy as np
from time import time

t_0 = time()
cnewt = CubicNewtonMap(CubicMap(a=complex(3), b=complex(2)))
print(cnewt.cubic.roots)
print(time()-t_0)

t_0 = time()
im = cnewt.draw_julia(res_x=600, res_y=600, x_range=(-2,
                      2), y_range=(-2, 2))
print(time()-t_0)

t_0 = time()
im = cnewt.draw_ray(im=im, line_weight=4,
                    x_range=(-2, 2), y_range=(-2, 2),
                    angles=[0, -2*np.pi/3, 2*np.pi/3])
print(time()-t_0)

t_0 = time()
im = cnewt.draw_eqpot(im=im, line_weight=2,
                      x_range=(-2, 2), y_range=(-2, 2),
                      potentials=[0.5, 1, 2])
print(time()-t_0)

im.show()
