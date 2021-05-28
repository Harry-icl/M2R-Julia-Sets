"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
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

    def derivative(self, z) -> "CubicMap":
        """
        Return the derivative of the polynomial.

        Returns
        -------
        derivative: PolynomialMap
            The derivative of the polynomial.
        """
        return 3*z**2 - self.a

    def _calculate_multibrot(self,
                             res_x: int = 600,
                             res_y: int = 600,
                             iterations: int = 200,
                             x_range: tuple = (-3, 3),
                             y_range: tuple = (-3, 3)) -> np.ndarray:
        results = np.ones((res_x, res_y))
        c1 = -cmath.sqrt(self.a/3)
        c2 = cmath.sqrt(self.a/3)
        for x_i, x in enumerate(np.linspace(x_range[0], x_range[1], res_x)):
            for y_i, y in enumerate(np.linspace(y_range[0], y_range[1], res_y)):
                self.b = complex(x, y)
                z = complex(0)
                i = 0
                while i < iterations:
                    z = self(z)
                    if abs(z) > 3:
                        results[x_i, y_i] = i/iterations
                        break
                    i += 1

        return results

    def draw_multibrot(self,
                        res_x: int = 600,
                        res_y: int = 600,
                        iterations: int = 200,
                        x_range: tuple = (-3, 3),
                        y_range: tuple = (-3, 3)):
        results = self._calculate_multibrot(res_x, res_y, iterations, x_range, y_range)
        im = Image.fromarray(np.uint8(cm.cubehelix_r(results)*255))
        im.show()
        return im


class CubicNewtonMap(Map):
    """A Newton mapping f: C -> C, i.e. f(z) = z - g'(z)/g(z)."""

    def __init__(self, cubic: CubicMap) -> None:
        """
        Construct an instance of the CubicNewtonMap class.

        The iterative formula for Newton's method to find roots of a polynomial. For a cubic p(z), the Newton map will be:
            f(z) = z - p'(z)/p(z).
        
        Parameters
        ----------
        cubic: CubicNewtonMap
            The cubic to find the Newton map for.
        """
        self.cubic = cubic
    
    def __call__(self, z: complex) -> complex:  #noqa D102
        return z - self.cubic.derivative(z)/self.cubic(z)
