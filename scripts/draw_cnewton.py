from julia import CubicMap, CubicNewtonMap
import numpy as np
from time import time

cnewt = CubicNewtonMap(
    CubicMap(a=complex(1), b=complex(2))
    # CubicMap(a=complex(0.76, -0.593), b=complex(1, 0))
)

x_range = (-3, 3)
y_range = (-3, 3)

im = cnewt.draw_julia(res_x=600, res_y=600,
                      x_range=x_range, y_range=y_range, iterations=16)
print(cnewt.cubic.roots)

im = cnewt.draw_ray(im=im, line_weight=1,
                    x_range=x_range, y_range=y_range,
                    angles=[2*np.pi/12*i for i in range(12)])


# t_0 = time()
# im = cnewt.draw_eqpot(im=im, line_weight=1,
#                       x_range=x_range, y_range=y_range,
#                       potentials=[0.5], res_eqpot=4096)
# print(time()-t_0)

im.show()
