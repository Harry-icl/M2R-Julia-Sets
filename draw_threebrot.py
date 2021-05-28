from julia import CubicMap

threebrot = CubicMap(a=0)
threebrot.draw_multibrot(res_x=400, res_y=400, iterations=100, x_range=(-2, 2), y_range=(-2, 2))