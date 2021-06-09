"""Module containing map classes for cubic functions."""
import cmath
from functools import partial
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit
from numbers import Number

from .map import Map


class CubicMap(Map):
    """A cubic mapping f: C -> C."""

    def __init__(self, a: float = None, b: float = None):
        """
        Construct an instance of the CubicMap class.

        A complex cubic map p: C -> C of the form:
            p(z) = z^3 - az + b

        Parameters
        ----------
        a: float
            The term a in the cubic map.
        b: float
            The term b in the cubic map.
        """
        self.a = a
        self.b = b

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**3 - self.a*z + self.b

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 3*z**2 - self.a

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(b, a, c1, c2, iterations, z_max):
        z1 = c1
        z2 = c2
        i = 0
        while i < iterations and abs(z1 - c1) < z_max and abs(z2 - c2) < z_max:
            z1 = z1**3 - a*z1 + b
            z2 = z2**3 - a*z2 + b
            i += 1
        return i / iterations

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, a, b, iterations, z_max):
        i = 0
        while i < iterations and abs(z) < z_max:
            z = z**3 - a*z + b
            i += 1
        return i / iterations

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False) -> np.ndarray:
        if not isinstance(self.a, Number):
            raise ValueError(f"Expected Number for a, got {type(self.a)}.")
        c1 = -cmath.sqrt(self.a/3)
        c2 = cmath.sqrt(self.a/3)
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_mandelbrot,
                                           a=self.a,
                                           c1=c1,
                                           c2=c2,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_mandelbrot,
                                      a=self.a,
                                      c1=c1,
                                      c2=c2,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3,
                         multiprocessing: bool = False) -> np.ndarray:
        if not isinstance(self.a, Number):
            raise ValueError(f"Expected Number for a, got {type(self.a)}.")
        if not isinstance(self.b, Number):
            raise ValueError(f"Expected Number for b, got {type(self.b)}.")
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._escape_time_julia,
                                           a=self.a,
                                           b=self.b,
                                           iterations=iterations,
                                           z_max=z_max), num_list)
            results = np.reshape(result_list, (res_y, res_x))
        else:
            result_list = map(partial(self._escape_time_julia,
                                      a=self.a,
                                      b=self.b,
                                      iterations=iterations,
                                      z_max=z_max), num_list)
            results = np.reshape(np.fromiter(result_list, dtype=float),
                                 (res_y, res_x))

        return results

    def external_ray(self, theta, D=20, S=10, R=200, error=0.001):
        """
        Construct an array of points on the external ray of angle theta.

        Parameters
        ----------
        theta: float
            angle of the external ray
        D: int
            depth of the ray
        S: int
            sharpness of the ray
        R: int
            radius
        error: float
            error used for convergence of newton method
        """
        points = [R * cmath.exp(2 * np.pi * theta * 1j)]

        for i in range(1, D+1):
            for q in range(1, S+1):

                r_m = R ** (1 / (3 ** (i - 1 + q / S)))
                t_m = r_m**(3**(i)) * cmath.exp(2
                                                * np.pi
                                                * 1j
                                                * theta
                                                * 3**(i))
                b_next = points[-1]
                b_previous = 0

                while abs(b_previous - b_next) >= error:
                    C_k = b_next
                    D_k = [0, -self.a + 1]
                    for x in range(i):
                        D_k.append(3 * D_k[-1] * C_k ** 2
                                   - self.a * D_k[-2] + 1)
                        C_k = C_k ** 3 - self.a * C_k + b_next
                    b_previous = b_next
                    b_next = b_previous - (C_k - t_m) / D_k[-1]

                points.append(b_next)

        # filter to be in range [-2,2]
        points = filter(lambda x: abs(x.real) < 2 and abs(x.imag) < 2, points)

        return points

    def draw_ray(self, theta, D=20, S=10, R=50, error=0.1):
        """
        Draw an external ray on matplotlib.

        Oskar - can you add some more description to this docstring, as well as
        types for the parameters.
        """
        results = self.external_ray(theta, D, S, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()


class CubicNewtonMap(Map):
    """A Newton map f(z) = z - g'(z)/g(z) where g is cubic."""

    def __init__(self, cubic: CubicMap):
        """
        Construct an instance of the CubicNewtonMap class.

        Parameters
        ----------
        cubic: CubicNewtonMap
            The cubic to find the Newton map for.
        """
        self.cubic = cubic

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z - self.cubic.derivative(z)/self.cubic(z)

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False):
        raise NotImplementedError

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3,
                         multiprocessing: bool = False):
        raise NotImplementedError
