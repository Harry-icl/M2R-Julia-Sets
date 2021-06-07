if __name__ == "__main__":
    import time
    from julia import CubicMap, CubicNewtonMap

    cubic = CubicMap(a=complex(1.5, 0.5), b=complex(-1, -1))
    time0 = time.time()
    cubic.draw_julia(res_x=1000, res_y=1000, iterations=40)
    print(f"No multiprocessing: {time.time() - time0}")
    time0 = time.time()
    cubic.draw_julia(res_x=1000, res_y=1000, iterations=40, multiprocessing=True)
    print(f"Multiprocessing: {time.time() - time0}")