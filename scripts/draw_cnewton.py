from julia import CubicMap, CubicNewtonMap
import numpy as np

qnewt = CubicNewtonMap(CubicMap(a=complex(2), b=complex(2)))
im = qnewt.draw_julia(res_x=1440, res_y=1440, x_range=(-2, 2), y_range=(-2, 2))
for angle in np.linspace(np.pi/2, 5*np.pi/2, 2, endpoint=False):
    im = qnewt.draw_ray(im=im, angle=angle, line_weight=4,
                        x_range=(-2, 2), y_range=(-2, 2))
qnewt.draw_eqpot(im=im, line_weight=2,
                 x_range=(-2, 2), y_range=(-2, 2)).show()
