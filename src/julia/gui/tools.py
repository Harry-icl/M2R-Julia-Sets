"""Tools for the interactive GUI."""


def to_complex(x, y, x_range, y_range, x_res, y_res):
    """
    Convert coordinates from window to a complex number.

    Parameters
    ----------
    x: int
        The x coordinate.
    y: int
        The y coordinate.

    Returns
    -------
    z: complex
        The corresponding complex number.
    """
    x_val = x_range[0] + ((x / x_res)
                          * (x_range[1] - x_range[0]))
    y_val = y_range[0] + (((y_res - y) / y_res)
                          * (y_range[1] - y_range[0]))
    return complex(x_val, y_val)


def title_generator(a, x_range, y_range):
    """
    Generate a window title for GUI.

    Parameters
    ----------
    a: complex
        The a parameter in the cubic.
    x_range: tuple
        The range of x values displayed.
    y_range: tuple
        The range of y values displayed.

    Returns
    -------
    title: str
        The window title.
    """
    func_name = (f"z^3 - ({round(a.real, 3)} + {round(a.imag, 3)}i)z + b"
                 if a.imag > 0
                 else
                 f"z^3 - ({round(a.real, 3)} {round(a.imag, 3)}i)z + b")
    bottom_left = (f"{round(x_range[0], 3)} + {round(y_range[0], 3)}i"
                   if y_range[0] > 0
                   else f"{round(x_range[0], 3)} {round(y_range[0], 3)}i")
    top_right = (f"{round(x_range[1], 3)} + {round(y_range[1], 3)}i"
                 if y_range[1] > 0
                 else f"{round(x_range[1], 3)} {round(y_range[1], 3)}i")
    return f"{func_name}, ({bottom_left}, {top_right})"
