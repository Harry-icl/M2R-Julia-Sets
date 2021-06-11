
from julia import CubicMap, CubicNewtonMap

cubic = CubicMap(a=0, b=-1)
newton = CubicNewtonMap(cubic)
newton.draw_julia(multiprocessing=False).show()
