from julia import CubicMap
from matplotlib import cm

cubm = CubicMap(a=3*(-1)**(3/2), b=0)

cubm.draw_julia(res_x=1440, res_y=1440, iterations=10, z_max=2, colormap=cm.bone_r).show()