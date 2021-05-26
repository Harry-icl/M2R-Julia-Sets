import pytest
import cmath
try:
    from julia import PolynomialMap, PolynomialNewtonMap
except ImportError:
    pass


def test_import():
    from julia import PolynomialMap, PolynomialNewtonMap


@pytest.mark.parametrize("z, f_z", [
    (complex(1), complex(2/5)),
    (complex(1, 1), complex(-13/5, 7/5)),
    (complex(5, 2), complex(62, 704/5))
])
def test_call_polynomial(z, f_z):
    legendre_cubic = PolynomialMap((complex(0), complex(-3/5), complex(0), complex(1)))
    assert cmath.isclose(legendre_cubic(z), f_z, rel_tol=1e-6), \
        f"expected result of {f_z} but got {legendre_cubic(z)}"


@pytest.mark.parametrize("z, f_z", [
    (complex(1), complex(-5)),
    (complex(1, 1), complex(-31/218, 587/218)),
    (complex(5, 2), complex(662665/147929, 327520/147929))
])
def test_call_polynomial_newton(z, f_z):
    legendre_cubic = PolynomialMap((complex(0), complex(-3/5), complex(0), complex(1)))
    newton_legendre_cubic = PolynomialNewtonMap(legendre_cubic)
    assert cmath.isclose(newton_legendre_cubic(z), f_z, rel_tol=1e-6), \
        f"expected result of {f_z} but got {newton_legendre_cubic(z)}"
