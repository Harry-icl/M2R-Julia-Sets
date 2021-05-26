"""Module containing all of the classes for mappings."""
from abc import ABC, abstractmethod


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
    """A polynomial mapping f: C -> C."""

    def __init__(self, coefficients: tuple) -> None:
        """
        Construct an instance of the PolynomialMap class.

        A complex polynomial p: C -> C of the form:
            p: z |-> a + bz + cz^2 + ...

        Parameters
        ----------
        coefficients: tuple(complex)
            A tuple of coefficients from smallest order to largest, i.e. (a, b, c, d) would correspond to the polynomial a + bz + cz^2 + dz^3.
        """
        self.coefficients = coefficients

    def __call__(self, z: complex) -> complex:  # noqa D102
        return sum([a*z**i for i, a in enumerate(self.coefficients)])
    
    def derivative(self) -> "PolynomialMap":
        """
        Return the derivative of the polynomial.

        Returns
        -------
        derivative: PolynomialMap
            The derivative of the polynomial.
        """
        return PolynomialMap(tuple((i + 1)*a for i, a in enumerate(self.coefficients[1:])))


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
