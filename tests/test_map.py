import pytest
import cmath
try:
    from julia import CubicMap, CubicNewtonMap
except ImportError:
    pass


def test_import():
    from julia import CubicMap, CubicNewtonMap


@pytest.mark.parametrize("z, f_z", [
    (complex(1), complex(1)),
    (complex(1, 1), complex(-2, 2)),
    (complex(5, 2), complex(65, 142))
])
def test_call_polynomial(z, f_z):
    basic_cubic = CubicMap(0, 0)
    assert cmath.isclose(basic_cubic(z), f_z, rel_tol=1e-6), \
        f"expected result of {f_z} but got {basic_cubic(z)}"


@pytest.mark.parametrize("z, f_z", [
    (complex(1), complex(-2)),
    (complex(1, 1), complex(-1/2, 5/2)),
    (complex(5, 2), complex(130/29, 64/29))
])
def test_call_polynomial_newton(z, f_z):
    legendre_cubic = CubicMap(0, 0)
    newton_legendre_cubic = CubicNewtonMap(legendre_cubic)
    assert cmath.isclose(newton_legendre_cubic(z), f_z, rel_tol=1e-6), \
        f"expected result of {f_z} but got {newton_legendre_cubic(z)}"
