"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


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


class PolynomialMap(Map):
    """A polynomial mapping f: C -> C with a parameter c."""

    def __init__(self, coefficients: tuple, power: int, c: complex=None) -> None:
        """
        Construct an instance of the PolynomialMap class.

        A complex polynomial p: C -> C of the form:
            p: z |-> a + bz + cz^2 + ...

        Parameters
        ----------
        coefficients: tuple(complex)
            A tuple of coefficients from smallest order to largest, i.e. (a, b, c, d) would correspond to the polynomial a + bz + cz^2 + dz^3.
        power: int
            The power of z in the term with c.
        """
        self.coefficients = coefficients
        self.power = power
        self.c = c

    def __call__(self, z: complex) -> complex:  # noqa D102
        if self.c:
            return sum([a*z**i for i, a in enumerate(self.coefficients)]) + self.c*z**self.power
        else:
            raise ValueError("No value set for c.")
    
    def derivative(self) -> "PolynomialMap":
        """
        Return the derivative of the polynomial.

        Returns
        -------
        derivative: PolynomialMap
            The derivative of the polynomial.
        """
        return PolynomialMap(tuple((i + 1)*a for i, a in enumerate(self.coefficients[1:])))
    
    def _calculate_mandelbrot(self, res_x: int=600, res_y: int=600, max_x: float=2, max_y: float=2, iterations: int=200):
        results = np.ones((res_x, res_y), dtype=bool)
        for x_i, x in enumerate(np.linspace(-max_x, max_x, res_x)):
            for y_i, y in enumerate(np.linspace(-max_y, max_y, res_y)):
                self.c = complex(x, y)
                z = complex(0)
                for _ in range(iterations):
                    z = self(z)
                    if abs(z) > 2:
                        results[x_i, y_i] = False
                        break
                
        return results
    
    def draw_mandelbrot(self, res_x: int=600, res_y: int=600, max_x: float=2, max_y: float=2, iterations: int=200):
        results = self._calculate_mandelbrot()
        points = [(x, y) for x_i, x in enumerate(np.linspace(-max_x, max_x, res_x)) for y_i, y in enumerate(np.linspace(-max_y, max_y, res_y)) if results[x_i, y_i]]
        plt.scatter([point[0] for point in points], [point[1] for point in points], s=0.1)
        plt.show()                    


class PolynomialNewtonMap(Map):
    """A Newton mapping f: C -> C, i.e. f(z) = z - g'(z)/g(z)."""

    def __init__(self, polynomial: PolynomialMap) -> None:
        """
        Construct an instance of the PolynomialNewtonMap class.

        The iterative formula for Newton's method to find roots of a polynomial. For a polynomial p(z), the Newton map will be:
            f(z) = z - p'(z)/p(z).
        
        Parameters
        ----------
        polynomial: PolynomialMap
            The polynomial to find the Newton map for.
        """
        self.polynomial = polynomial
    
    def __call__(self, z: complex) -> complex:  #noqa D102
        return z - self.polynomial.derivative()(z)/self.polynomial(z)
