from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import skimage
from skimage import io, morphology
from skimage.feature import peak_local_max
from skimage.filters import sobel
import warnings
from skimage.filters import threshold_adaptive
from scipy import ndimage
from skimage.draw import circle_perimeter
#from skimage.filters import try_all_threshold

%matplotlib

rfp_fname = '../data/wQC60/wQC60_test_TexasRed.tif'
gfp_fname = '../data/wQC60/wQC60_test_wtGFP.tif'
gfp = io.imread_collection(gfp_fname)
rfp = io.imread_collection(rfp_fname)

def plot_gallery(images, n_row=3, n_col=4, fig_title=None):
    """
    Helper function to plot a gallery of portraits
    """
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i], cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    if fig_title: fig.suptitle(fig_title+'\n', fontsize=20)
    plt.tight_layout()
    return None

def fakeRGB2gray(im):
    """
    Check if an RGB image is grayscale and convert if necessary
    """
    try:
        # If the three channels are the same, it is grayscale, get only one
        if ((im[:,:,0] == im[:,:,1]) &  ( im[:,:,2]== im[:,:,0])).all():
            return im[:,:,0]
        else: raise ValueError('Channels are not identical')
    except IndexError: 
        warnings.warn('Image has only one channel')
        return im

def im_hist(im):
    """
    Plot image histogram
    """
    hist, bins = skimage.exposure.histogram(im)
    with sns.axes_style('darkgrid'):
        plt.fill_between(bins, hist, lw=0.25, alpha=0.4)
        plt.yscale('log')
        plt.xlabel('normalized pixel value')
        plt.ylabel('count')

def rectangleROI(im, thresh=None):
    """
    Identify coordinates of rectangular ROI
    which does not contain all continuous zero columns and rows after threshold
    """
    if not thresh:
        thresh = skimage.filters.threshold_li(im)
    # threshold to get boolean mask
    im_thresh = im > thresh
    # check that something remains after thresholding
    if not np.sum(im_thresh): raise RuntimeError('No pixels pass specified threshold')
    # get all zero rows and columns
    rows = np.nonzero(np.sum(im_thresh, 1))[0]
    cols = np.nonzero(np.sum(im_thresh, 0))[0]
    # remove only continuous zero rows and columns and store coordinates in
    # slice object
    roi_coords = (slice(min(rows),max(rows)+1), slice(min(cols),max(cols)+1))
    return roi_coords

def label_sizesel(im, im_mask, bounds=(100, 1500)):
    """
    Create and label markers from image mask, 
    filter by area and compute region properties
    """
    markers = morphology.label(im_mask)
    # delete markers of unusual area, likely not nuclei
    nuclei = skimage.measure.regionprops(markers, im)
    areas = [n.area for n in nuclei]
    # labels of markers within area bounds
    labels_ = [n.label for n in nuclei_rfp if bounds[0] < n.area < bounds[1]]
    # remove markers outside bounds
    ix = np.in1d(markers.ravel(), _labels).reshape(markers.shape)
    markers[~ix] = 0
    # update nuclei regionprops list
    nuclei = [n for n in nuclei if n.label in labels_]
    return markers, nuclei

def nuclei_int(im_r, im_g, plot=False):
    """
    get nuclei intensities
    """

    thresh = skimage.filters.threshold_li(im_r)
    # get nuclei markers 
    nuclei = peak_local_max(im_r, indices=True, threshold_abs=thresh, min_distance=25)

    # remove oversaturated nuclei
    nuclei = [tuple(n) for n in nuclei if im_r[tuple(n)] < 4000]

    # draw circle around non-oversaturated nuclei max for plotting
    circles = [circle_perimeter(r, c, 10) for (r,c) in nuclei]
    im_plot = im_r.copy()
    for circle in circles:
        im_plot[circle] = np.max(im_plot) 

    # get intensities of non-oversaturated nuclei
    int_r = np.array([im_r[n] for n in nuclei])
    int_g = np.array([im_g[n] for n in nuclei])
    int_ratio = int_r / int_g

    if plot: plot_nuceli_int(im_plot, int_ratio)

    return im_plot, int_ratio

def plot_nuceli_int(im_plot, int_ratio):
    fig, ax = plt.subplots(2)
    ax[0].imshow(im_plot, plt.cm.viridis)
    sns.stripplot(int_ratio, orient='v', size=10, alpha=0.5, cmap='viridis', ax=ax[1])
    ax[1].tick_params(axis='y', which='both', labelleft='off', labelright='on')
    ax[1].set_ylabel('intensity ratio', fontsize=20)
    ax[1].yaxis.set_label_position("right")
    return None

gfp, rfp = gfp[1:], rfp[1:]
for im_num in range(len(rfp)):
    im_r = rfp[im_num].copy()
    im_g = gfp[im_num].copy()

    # subregion histogram equalization to improve contrast
    #im = skimage.exposure.equalize_adapthist(im)
    # get roi from rfp and apply to both
    roi = rectangleROI(im_r, thresh)
    im_r = im_r[roi]
    im_g = im_g[roi]

    #im_thresh = threshold_adaptive(im, 15)
    #im_thresh = morphology.binary_opening(im_thresh, morphology.disk(3))
    #im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=200)

    #im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    #im_thresh = ndimage.morphology.binary_closing(im_thresh)
    #im_thresh = morphology.binary_opening(im_thresh, morphology.disk(5))

    #im_open = morphology.opening(im, morphology.disk(5))

#    # make the mask fit the markers
#    im_thresh = markers+im_thresh
    # reconstruct

    #selem = morphology.disk(300)
    #markers = morphology.reconstruction(markers, im, selem=selem)
    ##markers  = skimage.morphology.remove_small_objects(markers>0, min_size=100)

    ## uncomment below if finding peak local max on im_open
    ##markers = skimage.morphology.remove_small_objects(markers, min_size=50)
    ##markers = markers.astype(float)
    ##markers[markers>0] = 0.4

    #markers, nuclei = label_sizesel(im, im_thresh)
    #
    #intensities = [n.max_intensity for n in nuclei]
    im_plot, int_ratio = nuclei_int(im_r, im_g, plot=True)

