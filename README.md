Local Resolution Map (ResMap)
==========================
Local Resolution Map (ResMap) is a Python (NumPy/SciPy) application with a Tkinter GUI. It is a software package for computing the local resolution of 3D density maps studied in structural biology, primarily electron cryo-microscopy (cryo-EM).

How Do I Use It?
----------------
Please read the documentation within the source files.

What Should I Download?
-----------------------
Sourceforge should automatically present you with the latest binary for your platform.

The binaries have been packaged using PyInstaller and were tested on:

* Windows: 7 and 8
* Mac: 10.6+
* Linux: Fedora 14+, CentOS 6+

You may also download the source files (Python) and run them using your own Python setup.

Requirements are:

* Python (2.7+) [not tested on Python 3.X]
* NumPy (1.6+)
* SciPy (0.11+)

We have also provided simulated volumes (under test-data) of a radial symmetric "chirp signal" with two levels of white and 1/f noise.

Version History
---------------
* **(1.0.1)** [30/05/2013] Mask volume bug corrected.
* **(1.0.0)** [24/05/2013] Initial commit.
