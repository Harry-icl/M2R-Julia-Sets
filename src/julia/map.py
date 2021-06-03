"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm
import cmath


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
                              z_max: float = 3) -> np.ndarray:
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
                        z_max: float = 3) -> Image.Image:
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
                                             z_max)
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
        return im

    @abstractmethod
    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3) -> np.ndarray:
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
                   z_max: float = 3) -> Image.Image:
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
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
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

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3) -> np.ndarray:
        results = np.ones((res_x, res_y))
        c1 = -cmath.sqrt(self.a/3)
        c2 = cmath.sqrt(self.a/3)
        for x_i, x in enumerate(np.linspace(x_range[0],
                                            x_range[1],
                                            res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0],
                                                y_range[1],
                                                res_y)):
                self.b = complex(x, y)
                z1 = c1
                z2 = c2
                i = 0
                z1_diverge = False
                z2_diverge = False
                while i < iterations:
                    z1 = self(z1) if not z1_diverge else z1
                    z2 = self(z2) if not z2_diverge else z2
                    if abs(z1 - c1) > z_max:
                        z1_diverge = True
                    if abs(z2 - c2) > z_max:
                        z2_diverge = True
                    if z1_diverge and z2_diverge:
                        results[x_i, y_i] = i/iterations
                        break
                    i += 1

        return results

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3) -> np.ndarray:
        results = np.ones((res_x, res_y))
        for x_i, x in enumerate(np.linspace(x_range[0],
                                            x_range[1],
                                            res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0],
                                                y_range[1],
                                                res_y)):
                z = complex(x, y)
                i = 0
                while i < iterations:
                    z = self(z)
                    if abs(z) > z_max:
                        results[x_i, y_i] = i/iterations
                        break
                    i += 1

        return results


class CubicNewtonMap(Map):
    """A Newton map f(z) = z - g'(z)/g(z) where g is cubic."""

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


# temporary so I can implement quadratic newton
class QuadraticMap(Map):
    def __init__(self, c: complex = None) -> None:
        """
        Construct an instance of the QuadraticMap class.

        A complex quadratic map p: C -> C of the form:
            p(z) = z^2 + c

        Parameters
        ----------
        c: complex
            The parameter c in the quadratic map.
        """
        self.c = c

    def __call__(self, z: complex) -> complex:
        return z**2 + self.c

    def derivative(self, z: complex) -> complex:
        return 2*z

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3) -> np.ndarray:
        pass

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3) -> np.ndarray:
        pass


class QuadraticNewtonMap(Map):

    def __init__(self, quad: QuadraticMap) -> None:
        self.quad = quad

    def __call__(self, z: complex) -> complex:
        try:
            return (z - self.quad(z) / self.quad.derivative(z)
                    if self.quad.c != 0 else z/2)
        except ZeroDivisionError:
            return cmath.inf

    def _calculate_mandelbrot(self,
                              res_x: int = 600,
                              res_y: int = 600,
                              iterations: int = 200,
                              x_range: tuple = (-3, 3),
                              y_range: tuple = (-3, 3),
                              z_max: float = 3) -> np.ndarray:
        pass

    def draw_mandelbrot(self,
                        res_x: int = 600,
                        res_y: int = 600,
                        iterations: int = 200,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3),
                        z_max: float = 3) -> Image.Image:
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        d = ImageDraw.Draw(im)
        d.point([round(x_range[0]*(res_x-1)/(x_range[0] - x_range[1])),
                round(y_range[1]*(res_y-1)/(y_range[1] - y_range[0]))],
                fill=0)
        im.show()
        return im

    def _calculate_julia(self,
                         res_x: int = 600,
                         res_y: int = 600,
                         iterations: int = 200,
                         x_range: tuple = (-3, 3),
                         y_range: tuple = (-3, 3),
                         z_max: float = 3) -> np.ndarray:
        ref_pt = cmath.sqrt(self.quad.c)  # the Julia "line" goes through this
        re = ref_pt.real
        im = ref_pt.imag
        results = np.ones((res_x, res_y))
        if ref_pt == 0:
            results[round(-y_range[0]), :] = 0

    def draw_julia(self,
                   res_x: int = 600,
                   res_y: int = 600,
                   x_range: tuple = (-3, 3),
                   y_range: tuple = (-3, 3),
                   line_weight: int = 1) -> Image.Image:
        im = Image.fromarray(255*np.ones((res_x, res_y)))
        if self.quad.c == 0:
            return im
        d = ImageDraw.Draw(im)
        real = cmath.sqrt(self.quad.c).real
        imag = cmath.sqrt(self.quad.c).imag
        if abs(real) >= abs(imag):
            d.line([0,
                    round((x_range[0]*imag/real-y_range[1])
                          / (y_range[0]-y_range[1])*(res_y-1)),
                    res_x-1,
                    round((x_range[1]*imag/real-y_range[1])
                          / (y_range[0]-y_range[1])*(res_y-1))],
                   fill=0,
                   width=line_weight)
        else:
            d.line([round((y_range[1]*real/imag-x_range[0])
                          / (x_range[1]-x_range[0])*(res_y-1)),
                    0,
                    round((y_range[0]*real/imag-x_range[0])
                          / (x_range[1]-x_range[0])*(res_y-1)),
                    res_y-1],
                   fill=0,
                   width=line_weight)
        im.show()
        return im

    def _inv_bottcher(self, z: complex):
        return 1j*cmath.sqrt(self.quad.c)*(z+1)/(z-1)

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
