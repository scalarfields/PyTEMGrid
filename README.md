# PyTEMGrid

**A Validated SEM Approach for Measuring the Electron Transparency of Graphene on TEM Grids.**

PyTEMGrid is a Python package designed to automate the characterization of Transmission Electron Microscopy (TEM) grids using Scanning Electron Microscopy (SEM) images. It provides robust, mathematically validated tools to perform marker-based watershed segmentation, calculate the geometric transmission of the grid, and estimate monolayer graphene coverage.

## Features
* **Robust Segmentation:** Marker-based watershed algorithm to accurately separate grid holes from the metallic rim.
* **Holes Area Measuerement:** Estimate average hole area to compute the geometric transmission.
* **Coverage Analysis:** Built-in thresholding methods (Otsu, Triangle, Saddle-point) to differentiate bare holes from graphene-covered areas.
* **Preprocessing Toolkit:** Customizable filters (Median, Gaussian, CLAHE) to generalize the method across different SEM datasets and noise profiles.



