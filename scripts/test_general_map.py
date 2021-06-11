from julia import GeneralMap
from cmath import log, cosh
from numpy import inf

def func(z, c):
    return z**(abs(c) + 1)

abs_z_map = GeneralMap(func, c=1)

man = abs_z_map.draw_mandelbrot(iterations=50)
man.show()
"""
jul = abs_z_map.draw_julia(iterations=50)
jul.show()
"""