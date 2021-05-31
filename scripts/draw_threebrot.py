import time
time0 = time.time()
from julia import CubicMap

threebrot = CubicMap(a=0)
threebrot.draw_mandelbrot(res_x=2048, res_y=2048, iterations=100, x_range=(-2, 2), y_range=(-2, 2), z_max=2, multiprocessing=False)
threebrot.b = complex(0)
threebrot.draw_julia(z_max=1)
print(time.time() - time0)