"""Module containing the QuadraticMap class."""
import numpy as np
import multiprocessing as mp
from functools import partial
from numba import jit
import cmath
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
from matplotlib import cm

from .map import Map


class QuadraticMap(Map):
    """A quadratic mapping f: C -> C."""

    def __init__(self, c: float = None):
        """
        Construct an instance of the QuadraticMap class.

        A complex quadratic map p: C -> C of the form:
            p(z) = z^2 + c

        Parameters
        ----------
        c: float
            The term c in the quadratic map.
        """
        self.c = c

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z**2 + self.c

    def derivative(self, z: complex) -> complex:  # noqa D102
        return 2*z

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(c, iterations, z_max):
        z = c
        for i in range(iterations):
            z = z**2 + c
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z, c, iterations, z_max):
        for i in range(iterations):
            z = z**2 + c
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

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
            for q in range(1, S + 1):

                r_m = R ** (1 / (2 ** (i - 1 + q / S)))
                t_m = r_m**(2**(i)) * cmath.exp(2 *
                                                np.pi * 1j * theta * 2**(i))
                c_next = points[-1]
                c_previous = 0

                while abs(c_previous - c_next) >= error:
                    C_k = c_next
                    D_k = 1
                    for x in range(i):
                        D_k = 2 * D_k * C_k + 1
                        C_k = C_k ** 2 + c_next
                    c_previous = c_next
                    c_next = c_previous - (C_k - t_m) / D_k

                points.append(c_next)

        # filter to be in range [-2,2]
        points = filter(lambda x: abs(x.real) < 2 and abs(x.imag) < 2, points)

        return points

    def draw_ray(self, theta, D=20, S=10, R=50, error=0.1):
        """Draw external ray."""
        results = self.external_ray(theta, D, S, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()

    def bottcher(self, c, n=5):
        """
        Find the Bottcher coordinate of point c.

        Parameters
        ----------
        c: complex
            point whose Bottcher coordinate we want
        n:
            precision of the Bottcher function
        """
        result = c
        for i in range(n - 1):
            result = self.__call__(result)
        result = 1 + c / (result ** 2)
        interim_result = result
        for j in range(n):
            result *= interim_result ** (1/(2**j))
        result = c * result
        return result

    def potential(self, c, n=5):
        """
        Find the Bottcher potential of point c.

        Parameters
        ----------
        c: complex
            point whose Bottcher potential we want
        n:
            precision of the Bottcher function
        """
        return math.log(abs(self.bottcher(c, n)))

    def calculate_equipotential(self,
                                equipotential,
                                res_x=600,
                                res_y=600,
                                x_range=(-3, 3),
                                y_range=(-3, 3),
                                n=5,
                                tol=10**(-6)):
        """
        Calculate equipotential curve.

        Parameters
        ----------
        equipotential: float
            target potential value
        n:
            precision of the Bottcher function
        tol:
            tolerance of isclose approximation
        """
        results = np.ones((res_x, res_y))

        for x_i, x in enumerate(np.linspace(x_range[0], x_range[1], res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0],
                                                y_range[1],
                                                res_y)):
                c = complex(x, y)
                pot = self.potential(c, n)
                # math.isclose(pot, equipotential, rel_tol=tol):
                if pot in [equipotential - x_range[0]/res_x,
                           equipotential + x_range[0]/res_x]:
                    results[x_i, y_i] = 0

        return results

    def draw_equipotential(self,
                           equipotential,
                           res_x=600,
                           res_y=600,
                           x_range=(-3, 3),
                           y_range=(-3, 3),
                           n=5,
                           tol=10**(-6)) -> Image.Image:
        """
        Draw equipotential lines.

        Add docstring please.
        """
        results = self.calculate_equipotential(
            equipotential, res_x, res_y, x_range, y_range, n, tol)
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
        return im


class QuadraticNewtonMap(Map):
    """A Newton mapping f: C -> C, i.e. f(z) = z - g'(z)/g(z)."""

    def __init__(self, quadratic: QuadraticMap):
        """
        Construct an instance of the QuadraticNewtonMap class.

        Parameters
        ----------
        quadratic: QuadraticNewtonMap
            The quadratic to find the Newton map for.
        """
        self.quadratic = quadratic

    def __call__(self, z: complex) -> complex:  # noqa D102
        return z - self.quadratic.derivative(z)/self.quadratic(z)

    @staticmethod
    @jit(nopython=True)
    def _escape_time_mandelbrot(c,
                                iterations,
                                z_max):
        z = c
        for i in range(iterations):
            z = z - (2*z)/(z**2 + c)
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _escape_time_julia(z,
                           c,
                           iterations,
                           z_max):
        for i in range(iterations):
            z = z - (2*z) / (z**2 + c)
            if abs(z) > z_max:
                return i / iterations
        else:
            return 1

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-2, 2),
                              y_range: tuple = (-2, 2),
                              z_max: float = 2,
                              multiprocessing: bool = False):
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
                         multiprocessing: bool = False):
        num_list = [complex(x, y)
                    for y in np.linspace(y_range[0], y_range[1], res_y)
                    for x in np.linspace(x_range[0], x_range[1], res_x)]
        pass

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   line_weight: int = 1) -> Image.Image:
        """
        Draw a Julia set for the current map.

        Docstring please
        """
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        if self.quadratic.c == 0:
            return im
        d = ImageDraw.Draw(im)
        real = cmath.sqrt(self.quadratic.c).real
        imag = cmath.sqrt(self.quadratic.c).imag
        if abs(real) >= abs(imag):
            d.line(
                [self._complex_to_pixel(x_range[0]+x_range[0]*imag/real*1j,
                                        res_x, res_y, x_range, y_range),
                 self._complex_to_pixel(x_range[1]+x_range[1]*imag/real*1j,
                                        res_x, res_y, x_range, y_range)],
                fill=0,
                width=line_weight
            )
        else:
            d.line(
                [self._complex_to_pixel(y_range[0]*real/imag+y_range[0]*1j,
                                        res_x, res_y, x_range, y_range),
                 self._complex_to_pixel(y_range[1]*real/imag+y_range[1]*1j,
                                        res_x, res_y, x_range, y_range)],
                fill=0,
                width=line_weight
            )
        im.show()
        return im

    def _inv_bottcher(self, z: complex):
        return 1j*cmath.sqrt(self.quadratic.c)*(z+1)/(z-1)

    def _complex_to_pixel(self,
                          z: complex,
                          res_x: int = 600,
                          res_y: int = 600,
                          x_range: tuple = (-3, 3),
                          y_range: tuple = (-3, 3)) -> tuple:
        return (round((z.real-x_range[0])/(x_range[1]-x_range[0])*(res_x-1)),
                round((z.imag-y_range[1])/(y_range[0]-y_range[1])*(res_y-1)))

    def _calculate_rays(self,
                        res_x: int = 600,
                        res_y: int = 600,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3),
                        multiples: int = 12,
                        res_ray: int = 1024):
        return [[self._complex_to_pixel(
            self._inv_bottcher(cmath.rect(1/np.cos(r), phi)),
            res_x,
            res_y,
            x_range,
            y_range
        )
            for r in np.linspace(0, np.pi/2, res_ray+2)[1:-1]]
            for phi in np.linspace(0, 2*np.pi, multiples+1)[:-1]]

    def draw_rays(self,
                  res_x: int = 600,
                  res_y: int = 600,
                  x_range: tuple = (-3, 3),
                  y_range: tuple = (-3, 3),
                  multiples: int = 12,
                  res_ray: int = 1024,
                  line_weight: int = 1) -> Image.Image:
        """
        Draw external rays.

        Oskar - docstring please.
        """
        im = self.draw_julia(res_x, res_y, x_range, y_range, line_weight)
        d = ImageDraw.Draw(im)
        for ray in self._calculate_rays(res_x,
                                        res_y,
                                        x_range,
                                        y_range,
                                        multiples,
                                        res_ray):
            d.line(ray, fill=0, width=line_weight, joint="curve")
        im.show()
        return im

    def _calculate_eqpots(self,
                          res_x: int = 600,
                          res_y: int = 600,
                          x_range: tuple = (-3, 3),
                          y_range: tuple = (-3, 3),
                          levels: int = 12,
                          res_eqpot: int = 1024):
        return [[self._complex_to_pixel(
            self._inv_bottcher(cmath.rect(1/np.cos(r), phi)),
            res_x,
            res_y,
            x_range,
            y_range
        )
            for phi in np.linspace(0, 2*np.pi, res_eqpot+1)[:-1]]
            for r in np.linspace(0, np.pi/2, levels+2)[1:-1]]

    def draw_eqpots(self,
                    res_x: int = 600,
                    res_y: int = 600,
                    x_range: tuple = (-3, 3),
                    y_range: tuple = (-3, 3),
                    levels: int = 12,
                    res_eqpot: int = 1024,
                    line_weight: int = 1) -> Image.Image:
        """
        Draw an equipotential line.

        Oskar - docstring please.
        """
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        d = ImageDraw.Draw(im)
        for eqpot in self._calculate_eqpots(res_x,
                                            res_y,
                                            x_range,
                                            y_range,
                                            levels,
                                            res_eqpot):
            d.line(eqpot, fill=0, width=line_weight, joint="curve")
        im.show()
        return im
