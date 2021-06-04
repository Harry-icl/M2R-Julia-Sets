"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from matplotlib import cm
import cmath
import matplotlib.pyplot as plt
import math

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
                t_m = r_m**(3**(i)) * cmath.exp(2 * np.pi * 1j * theta * 3**(i))
                b_next = points[-1]
                b_previous = 0   

                while abs(b_previous - b_next) >= error:
                    C_k = b_next
                    D_k = [0, -self.a + 1]
                    for x in range(i):
                        D_k.append(3 * D_k[-1] * C_k ** 2 - self.a * D_k[-2] + 1)
                        C_k = C_k ** 3 - self.a * C_k + b_next
                    b_previous = b_next
                    b_next = b_previous - (C_k - t_m) / D_k[-1]
                
                points.append(b_next)
        
        # filter to be in range [-2,2]
        points = filter(lambda x: abs(x.real) < 2 and abs(x.imag) < 2, points)
        
        return points
    
    def draw_ray(self, theta, D=20, S=10, R=50, error=0.1):

        results = self.external_ray(theta, D, S, R, error)
        results = [[i.real, i.imag] for i in results]
        x = [x[0] for x in results]
        y = [x[1] for x in results]
        plt.plot(x, y)
        plt.show()


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


class QuadraticMap(Map):
    """A polynomial mapping f: C -> C."""

    def __init__(self, c):
        """
        Construct an instance of the PolynomialMap class.

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
                t_m = r_m**(2**(i)) * cmath.exp(2 * np.pi * 1j * theta * 2**(i))
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
    
    def calculate_equipotential(self, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3,3), n=5, tol=10**(-6)):
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
            for y_i, y in enumerate(np.linspace(y_range[0], y_range[1], res_y)):
                c = complex(x, y)
                pot = self.potential(c, n)
                if pot in [equipotential - x_range[0]/res_x, equipotential + x_range[0]/res_x]: #math.isclose(pot, equipotential, rel_tol=tol):
                    results[x_i, y_i] = 0
        
        return results
    
    def draw_equipotential(self, equipotential, res_x=600, res_y=600, x_range=(-3, 3), y_range=(-3, 3), n=5, tol=10**(-6)) -> Image.Image:

        results = self.calculate_equipotential(equipotential, res_x, res_y, x_range, y_range,n, tol)
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
        return im