# GUI for M2R-Julia-Sets
The interactive GUI plots the Mandelbrot set for a function alongside the Julia set. You can visualise either the quadratic or the cubic Mandelbrot and Julia sets.

## Usage
First, install the package via the instructions in the [README](./README.md).

Then, to run the GUI, use the command `python -m julia`. The default is to use the quadratic function z^2 + c, to use the cubic function z^3 - az + b use the command `python -m julia -c`. To enable multiprocessing, use the `-m` option.

## Controls
* **Zooming** - To zoom in on an area of a plot, click and drag the area you'd like to zoom in on.
* **Choosing a** - To choose a new value for a, left click any point on the Mandelbrot plot and the corresponding complex number will be the new value.
* **Choosing b/c** - To choose a new value for b or c, for the Julia set, right click any point on the plot and this complex number will be the new value.
* **Reset** - Press the m key to reset the Mandelbrot set to its original zoom and the j key to reset the Julia set.
* **External rays** - To draw external rays, press the e key and use the dialog box to draw the chosen rays.
* **Quit** - Press the q key to quit the application.