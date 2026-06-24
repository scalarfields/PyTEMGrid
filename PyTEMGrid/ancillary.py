# experimental.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from skimage.filters import threshold_li
from skimage.filters import threshold_yen
from skimage import measure, draw, morphology


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

def adaptive_thresh(grid_obj, block_size, shift, radius = None, move = False):
        """ 
        adaptive threshold routine that can be used for holes extraction from the grid. Works on images where the grid rim and noise or dirt are cut and put to zero. 
        (this has to be done before instantiating the tem_grid_image object, with fiji, for example). 
        WARNING: Comparison with experimental data revealed that
        it does not produce holes with an area comparable with the one measured with electrons. 
        Watershed segmentation works better. Anyway, it is a very robust method that always identifies the holes, even if not always correctly. 

        Parameters
        -----------
        grid_obj : tem_grid_image object
            object of the tem_grid_image class on which to perform the adaptive thresholding. 
        block_size: int
            side of the square where to compute the threshold for each pixel.
        shift: int
            offset to apply to the threshold computed in the block for each pixel

        radius: sequence of float or ndarray of shape (2,), default = None
            if radius is provided, the circularize_holes_function is called with the values provided in radius and
            other arguments set to their default values. If it is None, no other operation is performed on the image
        move: bool, default = False
            if move is set to true, the move_center flag of circularize holes is activated if radius is not none

        Returns
        --------
        holes: ndarray
            image containing the holes
        grid: ndarray
            image containing the grid
        mask:
            binary image representing the holes

        """
        mask = cv2.adaptiveThreshold(grid_obj.filter("median"), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, shift)
        grid_obj.holes = np.zeros_like(grid_obj.image)  
        grid_obj.grid = np.zeros_like(grid_obj.image)
        
        if radius is not None:
            normalized_mask = (mask / 255).astype(bool)
            normalized_mask = (~normalized_mask).astype(int)
            new_mask = circularize_holes(normalized_mask, radius, move_center=move)
            grid_obj.grid[~new_mask] = grid_obj.image[~new_mask]  
            grid_obj.holes[new_mask] = grid_obj.image[new_mask]
        else:
            normalized_mask = (mask / 255).astype(bool)
            grid_obj.grid[normalized_mask] = grid_obj.image[normalized_mask]  
            grid_obj.holes[~normalized_mask] = grid_obj.image[~normalized_mask]

        closure = cv2.morphologyEx(grid_obj.holes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))   
       
        new_mask = (closure!=0)
        grid_obj.adaptive_mask = new_mask
        grid_obj.holes = np.where(new_mask, grid_obj.image, 0)
        grid_obj.grid = np.where(new_mask,  0, grid_obj.image)
        return grid_obj.holes, grid_obj.grid, grid_obj.adaptive_mask



def li_threshold(holes_obj):
        """
        Applies li threshold on the class object.
        Parameters
        ------------
            holes_obj: tem_grid_image object
                object of the tem_grid_image class on which to perform the li thresholding. 

        Returns:
        ----------
            covered: np.ndarray
                image containing just the above li threshold pixels.  Also stored in `holes_obj.covered`
            uncovered: np.ndarray
                image containing just the below li threshold pixels. Also stored in `holes_obj.uncovered`
        
        """
        nonzero_mask = holes_obj.image > 0
        #nonzero_pixels = self.filter("median")[nonzero_mask]
        #print(nonzero_pixels, np.shape(nonzero_pixels))
        #li_threshold = threshold_li(nonzero_pixels)
        li_threshold = threshold_li(holes_obj.image[nonzero_mask])
        above_mask = (holes_obj.image >= li_threshold) & nonzero_mask #covered holes
        below_mask = (holes_obj.image < li_threshold) & nonzero_mask #uncovered holes
        

        holes_obj.covered = np.zeros_like(holes_obj.image)
        holes_obj.uncovered = np.zeros_like(holes_obj.image)

        holes_obj.covered[above_mask] = holes_obj.image[above_mask]
        holes_obj.uncovered[below_mask] = holes_obj.image[below_mask]
        print("li thresh", li_threshold)

        return holes_obj.covered, holes_obj.uncovered, li_threshold


def yen_threshold(holes_obj):
        """
        Applies yen threshold on the class object.
        Parameters
        ------------
            holes_obj: tem_grid_image object
                object of the tem_grid_image class on which to perform the yen thresholding. 

        Returns:
        ----------
            covered: np.ndarray
                image containing just the above yen threshold pixels.  Also stored in `holes_obj.covered`
            uncovered: np.ndarray
                image containing just the below yen threshold pixels. Also stored in `holes_obj.uncovered`
        
        """

        nonzero_mask = holes_obj.image > 0
        #nonzero_pixels = holes_obj.filter("median")[nonzero_mask]
        #print(nonzero_pixels, np.shape(nonzero_pixels))
        #li_threshold = threshold_li(nonzero_pixels)
        yen_threshold = threshold_yen(holes_obj.image[nonzero_mask])
        above_mask = (holes_obj.image >= yen_threshold) & nonzero_mask #covered holes
        below_mask = (holes_obj.image < yen_threshold) & nonzero_mask #uncovered holes
        

        holes_obj.covered = np.zeros_like(holes_obj.image)
        holes_obj.uncovered = np.zeros_like(holes_obj.image)

        holes_obj.covered[above_mask] = holes_obj.image[above_mask]
        holes_obj.uncovered[below_mask] = holes_obj.image[below_mask]
        print("yen thresh", yen_threshold)

        return holes_obj.covered, holes_obj.uncovered, yen_threshold