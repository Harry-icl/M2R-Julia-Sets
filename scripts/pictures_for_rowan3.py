from julia import CubicMap
from matplotlib import cm

cubic_map = CubicMap(a=0, b=complex(3, 3))

img = cubic_map.draw_julia(res_x=1440, res_y=1440, iterations=10, z_max=2, colormap=cm.bone_r)
img.show()