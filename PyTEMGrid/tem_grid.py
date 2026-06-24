import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import os
import subprocess
import pandas as pd
import tifffile
from skimage import measure, draw, morphology
from skimage.filters import threshold_triangle
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import shutil
from scipy.fft import fft2, ifft2, fftshift
from skimage.feature import peak_local_max

def get_pytemgrid_style():
    """Restituisce il dizionario di stile, controllando se LaTeX è disponibile."""
    style = {
        "font.family": "serif",
        "font.weight": "bold",
        "font.size": 17,
        "axes.edgecolor": "gray",
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "xtick.bottom": True,
        "xtick.labelbottom": True,
        "ytick.left": True,
        "ytick.labelleft": True,
        "xtick.direction": "inout",
        "ytick.direction": "inout",
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.pad": 10,
        "ytick.major.pad": 10,
        "legend.fancybox": False
    }
    
    # Controlla se 'latex' è installato nel sistema dell'utente
    if shutil.which("latex"):
        style["text.usetex"] = True
    else:
        style["text.usetex"] = False
        # Fallback: usa il font Computer Modern di Matplotlib per simulare LaTeX
        style["mathtext.fontset"] = "cm" 
        
    return style

def circularize_holes(mask, radius, move_center = False):
    """" 
    Improves the holes mask by substituting the holes with areas in [700, 1800] with circles whose area is randomly
    extracted from the mean and std you provide through the radius parameter. Additionally, it is possible to add a random shift 
    to the substituing circles activating the move_center flag.

    Parameters
    ----------
    mask : (M, N) ndarray of bool
        2D boolean array defining the hole mask.
    radius : sequence of float or ndarray of shape (2,)
        Mean and standard deviation of the circle radius. 
    move_center : bool, optional
        If True, randomly shifts the center of each replacement circle.
    
    Returns
    ----------
    circularized: (M,N) ndarray of bool
        new mask after circularization operations
    """
    if not isinstance(radius, (list, tuple, np.ndarray)) or len(radius) != 2:
        raise ValueError("The 'radius' parameter must be a sequence of two values: (mean, std).")
    labeled_mask = measure.label(mask)
    circularized = np.zeros_like(mask)

    for region in measure.regionprops(labeled_mask):
        if region.area < 700 or region.area > 1800:  # skip very small objects
            for coord in region.coords:
                circularized[coord[0], coord[1]] = mask[coord[0], coord[1]]
            continue

        # Get center and approximate radius
        cy, cx = region.centroid
        if move_center:
            cy+= np.random.normal(0,1.5)
            cx+= np.random.normal(0,1.5)
        #radius = np.sqrt(region.area / np.pi)
        r = np.random.normal(radius[0], radius[1])
        # Create circular hole
        rr, cc = draw.disk((cy, cx), r , shape=mask.shape)
        circularized[rr, cc] = 1

    return circularized.astype(bool)




class tem_grid_image:
    def __init__(self, image_input):
        """ class representing a tem grid image performed at SEM with all the tools needed to measure its geometric transmission
        and graphene coverage. 
        
        Parameters
        -----------
        image_input: str or np.ndarray
            path to the image or numpy array of the image itself
        fiji_path: 
            relative path to the fiji installation you want to use to run fiji macros from inside the class

        Returns
        --------
        tem grid image: tem_grid_image
        """
        if isinstance(image_input, str):
            self.image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_input, np.ndarray):
            self.image = image_input
        else:
            raise ValueError("Input must be either a file path or a numpy array")
        self.filters = dict()
        self.holes = None
        self.grid = None

    
    def save_tiff(self, out_file_name):
        """Saves self.image in tif format
        
        Parameters:
        -----------
        out_file_name: str
            path for output file
        """
        tifffile.imwrite(out_file_name, self.image)


    def display(self,  ax = None , figsize=(5,5), cmap="gray",  filter_name = None):
        """ Plots self.image if filter_name is None or a filtered version of self.image if the filter name is specified
            
        Parameters: 
        -----------
        ax: matplotlib.axes, optional
            the ax where you want to put your histo. Default is None and creates a new ax
        cmap: str, optional
            the colormap you want to use (default = "gray")
        figsize: tuple, optional
            the figsize of the image (default = (5,5))
        filter_name: str, optional
            the filter you want to display self.image with
        
        Returns: nothing
        --------
        """
        with plt.rc_context(get_pytemgrid_style()):
            plt.figure(figsize=figsize)
            if ax is None:
                fig, ax = plt.subplots() 
            if filter_name is None:
                ax.imshow(self.image, cmap=cmap)
            else: 
                ax.imshow(self.filter(filter_name), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])

            # Optionally also hide the axis frame (spines)
            for spine in ax.spines.values():
                spine.set_visible(False)
                

    def hist(self, ax=None, label=None):
        """ Plots the histogram of the flattened self.image in 256 bins from 0 to 256
            
        Parameters: 
        ax: matplotlib.axes, optional
            the ax where you want to put your histo. Default is None and creates a new ax
        label: str (optional)
            the label to your histo. Default is None

        Returns:
        ---------
        Bins: numpy.ndarray
        Edges: numpy.ndarray
        ax: matplotlib.axes
        """
        with plt.rc_context(get_pytemgrid_style()):
            if ax is None:
                fig, ax = plt.subplots() 
            bins, edges, _ = ax.hist(self.image.flatten(), bins=256, range=(0, 256), alpha=0.5, label=label)
            
            ax.set_xlabel("pixel gray level values", fontsize = 20)
            ax.set_ylabel("counts", fontsize = 20)
            ax.set_yscale("log")
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
            ax.legend( frameon=False, fontsize = 20)
            return bins, edges, ax


    def filter(self, filter_name, inplace=False, **kwargs):
        """
        Applies a filter to the image for pre-processing.

        Parameters
        ----------
        filter_name : str
            The name of the filter to apply. Available options are: 
            "blur", "canny", "laplacian", "contrast", "median".
        inplace : bool, optional
            If True, overwrites `self.image` with the filtered image so that 
            subsequent segmentation methods use the processed version. 
            Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to customize the filter parameters. 
            Depending on the `filter_name`, the following arguments are accepted:
            
            - For "blur":
                ksize : tuple, default=(5, 5)
                    Gaussian kernel size.
            
            - For "canny":
                blur_ksize : tuple, default=(5, 5)
                    Gaussian kernel size applied before Canny edge detection.
                threshold1 : int, default=50
                    First threshold for the hysteresis procedure.
                threshold2 : int, default=150
                    Second threshold for the hysteresis procedure.
            
            - For "laplacian":
                blur_ksize : tuple, default=(5, 5)
                    Gaussian kernel size applied before the Laplacian operator.
                ksize : int, default=3
                    Aperture size used to compute the second-derivative filters.
            
            - For "contrast" (CLAHE):
                clipLimit : float, default=2.0
                    Threshold for contrast limiting.
                tileGridSize : tuple, default=(8, 8)
                    Size of the grid for histogram equalization.
            
            - For "median":
                ksize : int, default=9
                    Aperture linear size; it must be an odd integer greater than 1.

        Returns
        -------
        filtered_img : numpy.ndarray
            The filtered image array.
        """
        if filter_name == "blur":
            ksize = kwargs.get("ksize", (5, 5))
            filtered = cv2.GaussianBlur(self.image, ksize, 0)
            filtered = cv2.normalize(filtered, None, 0, 1, cv2.NORM_MINMAX)
            
        elif filter_name == "canny":
            blur_ksize = kwargs.get("blur_ksize", (5, 5))
            t1 = kwargs.get("threshold1", 50)
            t2 = kwargs.get("threshold2", 150)
            blur = cv2.GaussianBlur(self.image, blur_ksize, 0)
            canny = cv2.Canny(blur, threshold1=t1, threshold2=t2)
            filtered = cv2.normalize(canny, None, 0, 1, cv2.NORM_MINMAX)
            
        elif filter_name == "laplacian":
            blur_ksize = kwargs.get("blur_ksize", (5, 5))
            ksize = kwargs.get("ksize", 3)
            blur = cv2.GaussianBlur(self.image, blur_ksize, 0)
            laplacian = cv2.Laplacian(blur, cv2.CV_64F, ksize=ksize)
            laplacian = np.abs(laplacian)
            filtered = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
            
        elif filter_name == "contrast":
            clip_limit = kwargs.get("clipLimit", 2.0)
            grid_size = kwargs.get("tileGridSize", (8, 8))
            filtered = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(self.image)
            
        elif filter_name == "median":
            ksize = kwargs.get("ksize", 9)
            filtered = cv2.medianBlur(self.image, ksize)
            
        else:
            raise ValueError(f"Filter '{filter_name}' has not been implemented.") 

        # Save logic
        if inplace:
            self.image = filtered
        else:
            self.filters[filter_name] = filtered
            
        return filtered

        
    def analyze_particles(self, min_thresh = 1, min_size=1200, max_size=1800, min_circ=0.92, max_circ=1.0):
        """
        Replicate Fiji's 'Analyze Particles' tool on a binary image.

        Parameters
        ----------
        min_thresh : int, optional
            threshold value to be applied to the tem_grid_image object (default = 1).
            If to be applied to an holes image, keep default value.
        min_size : float, optional
            Minimum area (in pixels) of particles to keep (default = 1200).
        max_size : float, optional
            Maximum area (in pixels) of particles to keep (default = 1800).
        min_circ : float, optional
            Minimum circularity threshold (default = 0.92).
        max_circ : float, optional
            Maximum circularity threshold (default = 1).

        Returns
        -------
        results : pandas.DataFrame
            DataFrame with one row per particle containing:
            ['label', 'area', 'perimeter', 'circularity', 'centroid_x', 'centroid_y',
            'major_axis', 'minor_axis', 'bbox']
        labeled_image : np.ndarray
            Image with labeled regions (each particle has unique integer label).
        """

        # Label connected components
        _, binary_image = cv2.threshold(self.image, min_thresh, 255, cv2.THRESH_BINARY)
        labeled = measure.label(binary_image, connectivity=2)
        props = measure.regionprops(labeled)

        data = []
        for p in props:
            area = p.area
            perimeter = p.perimeter if p.perimeter > 0 else np.nan
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            #print(area, perimeter, circularity)

            if not (min_size <= area <= max_size):
                continue
            if not (min_circ <= circularity <= max_circ):
                continue

            data.append({
                "label": p.label,
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "centroid_x": p.centroid[1],
                "centroid_y": p.centroid[0],
                "major_axis": p.major_axis_length,
                "minor_axis": p.minor_axis_length,
                "bbox": p.bbox
            })

        df = pd.DataFrame(data)
        if len(data)==0:
            df = pd.DataFrame(columns=['label','area','perimeter','circularity','centroid_x','centroid_y','major_axis', 'minor_axis', 'bbox'])
        return df, labeled

    def compute_unit_cell_area(self, min_distance=20, pixel_size=1.0, show_plot=False):
        """
        Computes the area of the unit cell of the periodic grid using 2D autocorrelation.

        Parameters
        ----------
        min_distance : int, optional
            Minimum distance (in pixels) between peaks in the autocorrelation image.
            Adjust this based on the expected periodicity/hole spacing of your grid. (default = 20)
        pixel_size : float, optional
            Physical size of one pixel. The computed area will be multiplied by pixel_size**2. (default = 1.0)
        show_plot : bool, optional
            If True, displays the autocorrelation image with the identified vectors. (default = False)

        Returns
        -------
        area_physical : float
            The computed area of the unit cell in your physical units squared.
        vectors : tuple
            The two fundamental lattice vectors (v1, v2) in pixels.
        """
        # 1. Compute 2D Autocorrelation using FFT
        # Subtract mean to remove the zero-frequency DC component dominance
        img_mean_zero = self.image - np.mean(self.image)

        # FFT -> Power Spectrum -> IFFT -> Shift center
        f_transform = fft2(img_mean_zero)
        power_spectrum = np.abs(f_transform)**2
        autocorr = np.real(ifft2(power_spectrum))
        autocorr_shifted = fftshift(autocorr)

        # Normalize for easier visualization and peak finding
        autocorr_norm = autocorr_shifted / np.max(autocorr_shifted)

        # 2. Find peaks in the autocorrelation image
        peaks = peak_local_max(autocorr_norm, min_distance=min_distance)

        if len(peaks) < 3:
             raise ValueError("Could not find enough peaks to determine lattice vectors. Try decreasing min_distance.")

        # 3. Identify the central peak (due to fftshift, it should be in the middle)
        center_y, center_x = np.array(autocorr_norm.shape) // 2

        # 4. Calculate distances from the center to all peaks
        distances = np.linalg.norm(peaks - [center_y, center_x], axis=1)

        # 5. Sort peaks by distance (skip index 0, which is the center peak itself)
        sorted_indices = np.argsort(distances)
        
        v1 = None
        v2 = None

        # 6. Find two linearly independent vectors closest to the center
        for idx in sorted_indices[1:]:
            peak = peaks[idx]
            # Create vector from center to peak: (x, y) format
            vec = np.array([peak[1] - center_x, peak[0] - center_y]) 

            if v1 is None:
                v1 = vec
            elif v2 is None:
                # Check for linear independence using the cross product
                cross_prod = np.abs(np.cross(v1, vec))
                norm_v1 = np.linalg.norm(v1)
                norm_vec = np.linalg.norm(vec)
                
                # To prevent picking the parallel/anti-parallel vector (e.g., -v1), 
                # we ensure the angle between them is large enough (sin(theta) > 0.2, approx 11.5 deg)
                if (cross_prod / (norm_v1 * norm_vec)) > 0.2: 
                    v2 = vec
                    break

        if v1 is None or v2 is None:
            raise ValueError("Could not find two linearly independent lattice vectors. Check your image periodicity or min_distance.")

        # 7. Compute the unit cell area (in pixel squared)
        area_pixels = np.abs(np.cross(v1, v2))

        # Convert to physical units
        area_physical = area_pixels * (pixel_size ** 2)

        # Optional: Plotting for verification
        if show_plot:
            with plt.rc_context(get_pytemgrid_style()):
                fig, ax = plt.subplots(figsize=(6, 6))
                # Log scale often helps visualize the autocorrelation better
                ax.imshow(np.log1p(np.abs(autocorr_norm)), cmap='magma')
                
                # Draw the vectors
                ax.quiver(center_x, center_y, v1[0], v1[1], color='cyan', angles='xy', scale_units='xy', scale=1, label=r'$\vec{a}$')
                ax.quiver(center_x, center_y, v2[0], v2[1], color='lime', angles='xy', scale_units='xy', scale=1, label=r'$\vec{b}$')
                ax.scatter(center_x, center_y, color='red', marker='x', s=100) # Center
                
                ax.set_title(f"Autocorrelation\nArea: {area_physical:.2f}")
                ax.legend()
                ax.axis('off')
                plt.show()

        return area_physical, (v1, v2)
  
    
    
    
    def transmission_geom(self):
        """
        Computes the geometric transmission as \sum(pixels_holes != 0) / \sum(pixels_image!= 0). 
        If the image contains part of the rim, make sure to put the rim pixels to zero (you can use the cut function of Fiji, for example) 
        Parameters
        -----------

        Returns
        -----------
        transmission: float
            computed transmission value
         
        """
        if self.holes is None:
            raise ValueError("Holes mask not found. Please run .watershed() before calculating transmission.")
        bins, edges = np.histogram(self.image.flatten(), bins=256, range=(0, 256))
        bins_holes, edges_holes = np.histogram(self.holes.flatten(), bins=256, range=(0, 256))
        transmission = sum(bins_holes[1:])/sum(bins[1:])
        return transmission
        
 
       
    def watershed(self, k=0.3, radius = None, move = False):
        """
        Marker-based watershed routine to separate the holes from the grid. DOES NOT work if the grid rim or 
        anything else is cut from the image. This makes it more sensible to dirt and sample irregularities, but it
        is the most accurate algorithm developed up to now. If it does not work, turn to adaptive threshold that is more
        robust (in ancillary.py).

        Parameters
        -----------
            k: float in (0, 1], default = 0.3
                this parameter tells you how strict you want to be when identifying the sure foreground objects. The greater it is, the stricter you are.
            radius: sequence of float or ndarray of shape (2,), default = None
                if radius is provided, the circularize_holes_function is called with the values provided in radius and
                other arguments set to their default values. If it is None, no other operation is performed on the image
            move: bool, default = False
                if move is set to true, the move_center flag of circularize holes is activated if radius is not none

        Returns
        --------
            holes: ndarray
                an image containing just the holes with all the other pixels set to zero. Also stored in `self.holes`.
            grid: ndarray
                an image containing just the grid with all the other pixels set to zero. Also stored in `self.grid`.

        """
        #bin_img = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 0)
        ret, bin_img = cv2.threshold(self.image,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_CLOSE,
                           kernel,
                           iterations=1)
        sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
        dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist, k * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8) 
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR), markers)
        mask = markers != 1
        
        self.holes = np.zeros_like(self.image)  
        self.grid = np.zeros_like(self.image)
        
        
        if radius is not None:
            #normalized_mask = (mask / 255).astype(bool)
            #normalized_mask = (~normalized_mask).astype(int)
            mask = circularize_holes(mask, radius, move_center=move)
            self.grid[~mask] = self.image[~mask]  
            self.holes[mask] = self.image[mask]
        else:
            self.grid[~mask] = self.image[~mask]  
            self.holes[mask] = self.image[mask]
        
        
        self.holes = np.where(mask, self.image, 0)
        self.grid = np.where(mask, 0, self.image)

        return self.holes, self.grid



class holes_image(tem_grid_image):
    def __init__(self, image_input):
        super().__init__(image_input)
        self.covered = None
        self.uncovered = None
        

    def otsu_threshold(self):
        """
        Applies otsu threshold on the class object.

        Returns:
        ----------
            covered: np.ndarray
                image containing just the above otsu threshold pixels.  Also stored in `self.covered`
            uncovered: np.ndarray
                image containing just the below otsu threshold pixels. Also stored in `self.uncovered`
        
        """
        # Create new masks for covered and uncovered holes based on Otsu's threshold
        nonzero_mask = self.image > 0
        #nonzero_pixels = self.filter("median")[nonzero_mask]
        otsu_threshold, _ = cv2.threshold(self.image[nonzero_mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        above_mask = (self.image >= otsu_threshold) & nonzero_mask #covered holes
        below_mask = (self.image < otsu_threshold) & nonzero_mask #uncovered holes
        

        self.covered = np.zeros_like(self.image)
        self.uncovered = np.zeros_like(self.image)

        self.covered[above_mask] = self.image[above_mask]
        self.uncovered[below_mask] = self.image[below_mask]
        print("otsu", otsu_threshold)

        return self.covered, self.uncovered, otsu_threshold
    
     
    def histo_areas(self, areas, n_bins = 28, p0 = [1/np.sqrt(2*np.pi*50), 1480, 50, 0]):
        """ 
        plots the histogram of the measured areas and fits it with a gaussian distribution

        Parameters
        -----------
            areas: ndarray
                array containing the areas to measure
            n_bins: int (optional)
                number of bins of the histogram
        
        Returns
        ---------
            popt: ndarray
                array containing the parameters of the fit (amplitude, area, std, offset)
            pcov: ndarray
                array containing the covariance matrix of the fit
            av_hole_area: float
                mean value of the gaussian fit  
        """
        
        def gauss(x, A, mu, sigma, c):
            return A*np.exp(-(x-mu)**2/(2*sigma**2)) + c
        with plt.rc_context(get_pytemgrid_style()):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            weights = np.ones_like(areas) / len(areas)
            bins, edges, _ = ax.hist(areas, bins=n_bins, range=(np.min(areas), np.max(areas)),  weights = weights, alpha=0.6)
            binsize = edges[1]-edges[0]
            x = edges[1:]-0.5*binsize
            popt, pcov = curve_fit(gauss, x, bins, p0 = p0)
            x = np.arange(np.min(areas), np.max(areas), 1)
            ax.plot ( x, gauss(x, *popt), color='navy')
            plt.axvline(popt[1], -0.01, 1, linestyle="--", color='navy')
            ax.set_xlabel("hole areas [pixel]")
            ax.set_ylabel("normalized counts")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(60))  # labeled every 10
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(15))   # ticks every 2, unlabeled
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # labeled every 10
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            print("Average hole area", popt[1], "+-", np.sqrt(pcov[1][1]))  
            av_hole_area = popt[1]
            return popt, pcov, av_hole_area
    
    
 

    def triangle_threshold(self):
        """
        Applies triangle threshold on the class object.

        Returns:
        ----------
            covered: np.ndarray
                image containing just the above triangle threshold pixels.  Also stored in `self.covered`
            uncovered: np.ndarray
                image containing just the below triangle threshold pixels. Also stored in `self.uncovered`
            threshold value: int
                computed threshold value
        
        """

        nonzero_mask = self.image > 0
        #nonzero_pixels = self.filter("median")[nonzero_mask]
        #print(nonzero_pixels, np.shape(nonzero_pixels))
        #li_threshold = threshold_li(nonzero_pixels)
        triangle_threshold = threshold_triangle(self.image[nonzero_mask])
        above_mask = (self.image >= triangle_threshold) & nonzero_mask #covered holes
        below_mask = (self.image < triangle_threshold) & nonzero_mask #uncovered holes
        

        self.covered = np.zeros_like(self.image)
        self.uncovered = np.zeros_like(self.image)

        self.covered[above_mask] = self.image[above_mask]
        self.uncovered[below_mask] = self.image[below_mask]
        print("triangle thresh", triangle_threshold)

        return self.covered, self.uncovered, triangle_threshold
    

    def saddle_point_threshold(self):
        """
        Applies saddle point threshold on the class object. If less then two peaks are found, no object is returned and an error message is raised.

        Returns:
        ----------
            covered: np.ndarray
                image containing just the above saddle point threshold pixels.  Also stored in `self.covered`
            uncovered: np.ndarray
                image containing just the below saddle point threshold pixels. Also stored in `self.uncovered`
            threshold value: int
                computed threshold value
        
        """

        nonzero_mask = self.image > 0
        bins, edges = np.histogram(self.image.flatten(), bins=256, range=(0, 256))  
        peaks, _ = find_peaks(bins[1:], distance=5, prominence = 6000)

        if len(peaks) >= 2:
            # Sort peaks by prominence or height if needed
            p1, p2 = sorted(peaks[:2])  # take the first two peaks
            #print(p1, p2)
            #print(bins[p1+1:p2+2])
            # Find the index of the minimum value between the peaks
            saddle_idx = np.argmin(bins[p1+1:p2+2]) + p1+1
            #print(f"Saddle point threshold: {saddle_idx}")

            above_mask = (self.image >= saddle_idx) & nonzero_mask #covered holes
            below_mask = (self.image < saddle_idx) & nonzero_mask #uncovered holes
            

            self.covered = np.zeros_like(self.image)
            self.uncovered = np.zeros_like(self.image)

            self.covered[above_mask] = self.image[above_mask]
            self.uncovered[below_mask] = self.image[below_mask] 
            return self.covered, self.uncovered, saddle_idx   
        
        else:
            raise ValueError("Less than two peaks found. Cannot compute saddle point threshold.")

        
    
        
    def coverage(self, thresh = None):
        """
        Computes graphene coverage of the class object (holes image). Coverage is defined as covered_pixels/total_pixels.
        If threshold is provided, it is used to separate graphene and holes. Otherwise, the previously segmented image
        containing just graphene is used to identify graphene pixels. The latter can be built through any of the threshold 
        functions of this class.

        Parameters
        -----------
        thresh: int, optional
            threshold value to compute coverage. Default is None, that computes coverage assuming that graphene has already been
            separated from the holes image.
        
        Returns
        ---------
        coverage: int
            the coverage value
        """
        if self.covered is None:
            raise ValueError("Covered pixels mask not found. Please run a thresholding method (e.g., otsu_threshold) first, or provide a 'thresh' value.")
        if thresh == None:
            bins_holes, edges_holes = np.histogram(self.image.flatten(), bins=256, range=(0, 256))
            bins_covered, edges_covered = np.histogram(self.covered.flatten(), bins=256, range=(0, 256))
            return sum(bins_covered[1:])/sum(bins_holes[1:])
        else:
            bins_holes, edges_holes = np.histogram(self.image.flatten(), bins=256, range=(0, 256))
            return sum(bins_holes[thresh:])/sum(bins_holes[1:])
    

    

                

        