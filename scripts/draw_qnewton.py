from julia import QuadraticMap, QuadraticNewtonMap
import numpy as np

qnewt = QuadraticNewtonMap(QuadraticMap(c=1j))
im = qnewt.draw_julia(res_x=1440, res_y=1440)
for angle in np.linspace(0, 2*np.pi, 3, endpoint=False):
    im = qnewt.draw_ray(im=im, angle=angle, line_weight=10)
qnewt.draw_eqpot(im=im, line_weight=5).show()
