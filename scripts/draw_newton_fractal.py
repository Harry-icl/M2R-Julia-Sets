
from julia import CubicMap, CubicNewtonMap

cubic = CubicMap(a=1, b=0)
newton = CubicNewtonMap(cubic)
newton.draw_julia(multiprocessing=False, res_x=4096, res_y=4096).show()
