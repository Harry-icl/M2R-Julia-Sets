from julia import CubicMap, CubicNewtonMap
import numpy as np

cnewt = CubicNewtonMap(CubicMap(a=complex(1), b=complex(0)))
im = cnewt.draw_julia(res_x=600, res_y=600, x_range=(-2, 2), y_range=(-2, 2))
im = cnewt.draw_ray(im=im, line_weight=4,
                    x_range=(-2, 2), y_range=(-2, 2),
                    angles=[0, -2*np.pi/3, 2*np.pi/3])
im = cnewt.draw_eqpot(im=im, line_weight=2,
                      x_range=(-2, 2), y_range=(-2, 2))
im.show()
