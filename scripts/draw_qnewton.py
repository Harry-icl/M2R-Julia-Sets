from julia import QuadraticMap, QuadraticNewtonMap
import numpy as np

qnewt = QuadraticNewtonMap(QuadraticMap(c=1j))
im = qnewt.draw_julia(res_x=600, res_y=600, x_range=(-2, 2), y_range=(-2, 2))
im = qnewt.draw_ray(im=im, line_weight=4,
                    x_range=(-2, 2), y_range=(-2, 2),
                    angles=[0, -2*np.pi/3, 2*np.pi/3])
im = qnewt.draw_eqpot(im=im, line_weight=2,
                      x_range=(-2, 2), y_range=(-2, 2))
im.show()
