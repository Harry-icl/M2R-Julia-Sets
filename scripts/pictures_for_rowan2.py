if __name__ == "__main__":
    from julia import QuadraticMap
    from matplotlib import cm
    quadratic = QuadraticMap(c=complex(-0.892))
    interior = quadratic.draw_julia(res_x=1440, res_y=1440, iterations=4, x_range=(-3 , 3), y_range=(-3, 3), z_max=2, multiprocessing=True, colormap=cm.bone_r)
    interior.show()
    quadratic.c = complex(1, 0.733)
    exterior = quadratic.draw_julia(res_x=1440, res_y=1440, iterations=10, x_range=(-3, 3), y_range=(-3, 3), z_max=2, multiprocessing=True, colormap=cm.bone_r)
    exterior.show()
