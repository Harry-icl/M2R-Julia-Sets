"""Module containing map classes for cubic functions."""
import cmath
from functools import partial
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit
from PIL import Image

from .map import Map


class CubicMap(Map):
    """A cubic mapping f: C -> C."""

    def __init__(self, a: float = 0, b: float = 0):
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

        if a == b == 0:
            self.roots = np.array([complex(0)])
            return None
        elif a == 0:
            gamma = complex((-b)**(1/3))
        else:
            gamma = (-b/2+cmath.sqrt(b**2/4-a**3/27))**(1/3)
        omega = cmath.rect(1, 2*np.pi/3)
        omega_ = cmath.rect(1, -2*np.pi/3)
        self.roots = [gamma + a/(3*gamma)]
        if np.all(~np.isclose(gamma*omega + a/(3*gamma)*omega_, self.roots)):
            self.roots.append(gamma*omega + a/(3*gamma)*omega_)
        if np.all(~np.isclose(gamma*omega_ + a/(3*gamma)*omega, self.roots)):
            self.roots.append(gamma*omega_ + a/(3*gamma)*omega)
        self.roots = np.array(self.roots)

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**3 - self.a*z + self.b

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 3*z**2 - self.a

    @ staticmethod
    @ jit(nopython=True)
    def _escape_time_mandelbrot(b, a, c1, c2, iterations, z_max):
        z1 = c1
        z2 = c2
        z1_diverge = False
        z2_diverge = False
        for i in range(iterations):
            z1 = z1**3 - a*z1 + b if not z1_diverge else z1
            z2 = z2**3 - a*z2 + b if not z2_diverge else z2
            if abs(z1 - c1) > z_max:
                z1_diverge = True
            if abs(z2 - c2) > z_max:
                z2_diverge = True
            if z1_diverge and z2_diverge:
                return i / iterations
        else:
            return 1

    @ staticmethod
    @ jit(nopython=True)
    def _escape_time_julia(z, a, b, iterations, z_max):
        for i in range(iterations):
            z = z**3 - a*z + b
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False) -> np.ndarray:
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

    @ staticmethod
    @ jit(nopython=True)
    def _conv_time_julia(z, a, b, roots, iterations, tol):
        result = [0, 0, 0]
        for i in range(iterations):
            z -= (z**3 - a*z + b)/(3*z**2 - a)
            for j, r in enumerate(roots):
                if abs(z-r) < tol:
                    result[j] = 255*(1-i/iterations)
                    return result
        return result

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         tol: float = 1e-12,
                         multiprocessing: bool = False) -> np.ndarray:
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        for i, r in enumerate(self.cubic.roots):
            for r_ in self.cubic.roots[i+1:]:
                tol = min(tol, abs(r-r_)/3)
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            result_list = pool.map(partial(self._conv_time_julia,
                                           a=self.cubic.a,
                                           b=self.cubic.b,
                                           roots=self.cubic.roots,
                                           iterations=iterations,
                                           tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        else:
            result_list = map(partial(self._conv_time_julia,
                                      a=self.cubic.a,
                                      b=self.cubic.b,
                                      roots=self.cubic.roots,
                                      iterations=iterations,
                                      tol=tol), num_list)
            results = np.reshape(np.array(list(result_list), dtype=np.uint8),
                                 (res_y, res_x, 3))
        return results

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   iterations: int = 32,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   tol: float = 1e-12,
                   multiprocessing: bool = False) -> Image.Image:
        """
        Draw the Julia set for this map with the current parameter values.

        Parameters
        ----------
        res_x: int
            The horizontal resolution of the image.
        res_y: int
            The vertical resolution of the image.
        iterations: int
            The maximum number of times to apply the map iteratively.
        x_range: (float, float)
            The range of x values to consider.
        y_range: (float, float)
            The range of y values to consider.
        tol: float
            The minimum distance before considering the orbit to have
            converged.
        multiprocessing: bool
            Determines whether to use multiprocessing.


        Returns
        -------
        im: Image.Image
            The image of the Mandelbrot set as a Pillow image object.
        """
        results = self._calculate_julia(res_x,
                                        res_y,
                                        iterations,
                                        x_range,
                                        y_range,
                                        tol,
                                        multiprocessing)
        im = Image.fromarray(results[::-1], 'RGB')
        return im
