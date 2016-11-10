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

gfp, rfp = gfp[1:], rfp[1:]
for im in rfp:
    im = fakeRGB2gray(im)
    # remove oversaturated regions
    #im[im>4000] = 0
    # subregion histogram equalization to improve contrast
    #im = skimage.exposure.equalize_adapthist(im)
    thresh = skimage.filters.threshold_li(im)
    im = rectangleROI(im, thresh)
    im[im<thresh] = 0
    im_thresh = threshold_adaptive(im,15)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=200)
    # two options to get defined nuclei masks for morphological reconstruction
    # a)
    im_thresh = ndimage.morphology.binary_closing(im_thresh)
    # b)
    #im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    # then perform below operation on either output
    im_thresh = morphology.binary_opening(im_thresh, morphology.disk(5))

    # do morphological opening to enhance contrast between nuclei and background
    im_open = morphology.opening(im, morphology.disk(5))
    # get nuclei markers 
    markers = peak_local_max(im_open, indices=False, threshold_abs=0.3, min_distance=10)
    # make the mask fit the markers
    im_thresh = markers+im_thresh
    # reconstruct
    selem = morphology.disk(10)
    markers = morphology.reconstruction(markers,im_thresh,selem=selem)
    markers  = skimage.morphology.remove_small_objects(markers>0, min_size=100)

    # uncomment below if finding peak local max on im_open
    #markers = skimage.morphology.remove_small_objects(markers, min_size=50)
    #markers = markers.astype(float)
    #markers[markers>0] = 0.4
    markers = morphology.label(markers)
    # get nuclei
    assert markers.shape == im.shape
    nuclei_rfp = skimage.measure.regionprops(markers, im)
    nuclei_gfp = skimage.measure.regionprops(markers, im)
    # get intensities
    intensities = [n.max_intensity for n in nuclei]
    break
fig, ax = plt.subplots(1,3)
ax[0].imshow(im)
ax[1].imshow(im)
ax[1].imshow(markers, alpha=0.3, cmap='viridis')
sns.stripplot(intensities, orient='v', size=10, alpha=0.5, cmap='viridis', ax=ax[2])
ax[2].tick_params(axis='y', which='both', labelleft='off', labelright='on')
ax[2].set_ylabel('mean intensity (a.u.)', fontsize=20)
ax[2].yaxis.set_label_position("right")
