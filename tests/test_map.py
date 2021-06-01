import pytest
import cmath
import pickle
from PIL import ImageChops
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
    basic_cubic = CubicMap(0, 0)
    newton_basic_cubic = CubicNewtonMap(basic_cubic)
    assert cmath.isclose(newton_basic_cubic(z), f_z, rel_tol=1e-6), \
        f"expected result of {f_z} but got {newton_basic_cubic(z)}"

def test_mandelbrot_drawing():
    file = open('./tests/data/threebrot_mandelbrot.pkl', 'rb')
    mndl = pickle.load(file)
    threebrot = CubicMap(a=0)
    im = threebrot.draw_mandelbrot()
    diff = ImageChops.difference(mndl, im)
    assert not diff.getbbox()

def test_julia_drawing():
    file = open('./tests/data/threebrot_julia.pkl', 'rb')
    jul = pickle.load(file)
    threebrot = CubicMap(a=0, b=0)
    im = threebrot.draw_julia()
    diff = ImageChops.difference(jul, im)
    assert not diff.getbbox()