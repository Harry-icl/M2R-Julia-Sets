"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from matplotlib import cm
import cmath
import multiprocessing as mp
from functools import partial
from numba import jit


def draw_from_array(array: np.ndarray, colormap: cm = cm.cubehelix_r) -> Image:
    """
    Draw an image from an array of values between 0 and 1.

    Parameters
    ----------
    array: np.ndarray
        The array to draw the image from.
    colormap: cm
        The colormap to use for the image.
    """
    return Image.fromarray(np.uint8(colormap(array)*255))


class Map(ABC):
    """A mapping f: C -> C."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, z: complex) -> complex:
        """
        Return the result of the map acting on a value z.

        Parameters
        ----------
        z: complex
            The value to apply the map to.

        Returns
        -------
        f(z): complex
            The result of the mapping applied to z.
        """
        pass

    def derivative(self, z: complex) -> complex:
        """
        Return the derivative of the map at the given point.

        Parameters
        ----------
        z: complex
            The point to evaluate the derivative at.

        Returns
        -------
        df/dz | z: complex
            The value of the derivative at the given point.
        """
        pass

    @abstractmethod
    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3,
                              multiprocessing: bool = False) -> np.ndarray:
        """
        Calculate the escape time of given points as c values in the map.

        Parameters
        ----------
        res_x: int
            The number of points in the horizontal axis of the array.
        res_y: int
            The number of points in the vertical axis of the array.
        iterations: int
            The maximum number of times to apply the map iteratively.
        x_range: (float, float)
            The range of x values to consider.
        y_range: (float, float)
            The range of y values to consider.
        z_max: float
            The maximum z value before considering the point to have escaped.
        """
        pass

    def draw_mandelbrot(self,
                        res_x: int = 600,
                        res_y: int = 600,
                        iterations: int = 200,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3),
                        z_max: float = 3,
                        multiprocessing: bool = False) -> Image.Image:
        """
        Draw the Mandelbrot set for this map.

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
        z_max: float
            The maximum z value before considering the point to have escaped.

        Returns
        -------
        im: Image.Image
            The image of the Mandelbrot set as a Pillow image object.
        """
        results = self._calculate_mandelbrot(res_x,
                                             res_y,
                                             iterations,
                                             x_range,
                                             y_range,
                                             z_max,
                                             multiprocessing)
        im = draw_from_array(results)
        return im

    @abstractmethod
    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3,
                         multiprocessing: bool = False) -> np.ndarray:
        """
        Calculate the escape time of given points as z values in the map.

        Parameters
        ----------
        res_x: int
            The number of points in the horizontal axis of the array.
        res_y: int
            The number of points in the vertical axis of the array.
        iterations: int
            The maximum number of times to apply the map iteratively.
        x_range: (float, float)
            The range of x values to consider.
        y_range: (float, float)
            The range of y values to consider.
        z_max: float
            The maximum z value before considering the point to have escaped.
        """
        pass

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   iterations: int = 200,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   z_max: float = 3,
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
        z_max: float
            The maximum z value before considering the point to have escaped.

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
                                        z_max)
        im = draw_from_array(results)
        return im


class CubicMap(Map):
    """A polynomial mapping f: C -> C."""

    def __init__(self, a: float = None, b: float = None) -> None:
        """
        Construct an instance of the PolynomialMap class.

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

    @staticmethod
    @jit(nopython=True)
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


class CubicNewtonMap(Map):
    """A Newton mapping f: C -> C, i.e. f(z) = z - g'(z)/g(z)."""

    def __init__(self, cubic: CubicMap) -> None:
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
                              z_max: float = 3):
        raise NotImplementedError

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3):
        raise NotImplementedError
