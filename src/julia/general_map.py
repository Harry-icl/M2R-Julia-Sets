import numpy as np
from functools import partial
import multiprocessing as mp
from numbers import Number

from .map import Map

class GeneralMap(Map):
    """A quadratic mapping f: C -> C."""

    def __init__(self, func, c: float = None):
        """
        Construct an instance of the GeneralMap class.

        A complex map f(z, c). It must have a single critical point at 0.

        Parameters
        ----------
        func: function
            The function of the general map.
        c: float
            The term c in the quadratic map.
        """
        self.func = func
        self.c = c

    def __call__(self, z: complex) -> complex:  # noqa D102
        return self.func(z, self.c)

    def derivative(self, z: complex) -> complex:  # noqa D102
        raise NotImplementedError

    def _escape_time_mandelbrot(self, c, iterations, z_max):
        z = 0
        i = 0
        while i < iterations and abs(z) < z_max:
            z = self.func(z, c)
            i += 1
        return i / iterations

    def _escape_time_julia(self, z, c, iterations, z_max):
        i = 0
        while i < iterations and abs(z) < z_max:
            z = self.func(z, c)
            i += 1
        return i / iterations

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-2, 2),
                              y_range: tuple = (-2, 2),
                              z_max: float = 2,
                              multiprocessing: bool = False) -> np.ndarray:
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_mandelbrot,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_mandelbrot,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-2, 2),
                         y_range: tuple = (-2, 2),
                         z_max: float = 2,
                         multiprocessing: bool = False) -> np.ndarray:
        if not isinstance(self.c, Number):
            raise ValueError(f'Expected Number for c, got {type(self.c)}.')
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_julia,
                                           c=self.c,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_julia,
                                      c=self.c,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results
