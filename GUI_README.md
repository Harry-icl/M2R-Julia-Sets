# GUI for M2R-Julia-Sets
The interactive gui plots the Mandelbrot set for the function z^3 - az + b, the plot shows the Mandelbrot set wrt b, and you can vary a by clicking new points on the plot itself.

## Usage
First, install the package via the instructions in the [README](./README.md).

Then, to run the GUI, use the command `python -m julia`. The default is to use the quadratic function z^2 + c, to use the cubic function z^3 - az + b use the command `python -m julia -c`.

## Controls
* **Zooming** - To zoom in on an area of a plot, click and drag the area you'd like to zoom in on.
* **Choosing a** - To choose a new value for a, click any point on the plot and this complex number will be the new value.
* **Choosing b** - To choose a new value for b for the Julia set (or c in the quadratic case), right click any point on the plot and this complex number will be the new value.
* **Reset** - Press the m key to reset the Mandelbrot set to its original zoom and the j key to reset the Julia set.
* **External Rays** - To draw an external ray, press the e key and then enter the angle in the text box. To reset the external rays, press e and then click on the reset button.
* **Quit** - Press the q key to quit the application.