"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import cm
from PIL import Image


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
    def __init__(self):
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
        im = draw_from_array(results[::-1])
        return im

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
                                        z_max,
                                        multiprocessing)
        im = draw_from_array(results[::-1])
        return im
