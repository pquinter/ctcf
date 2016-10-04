from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import skimage
from skimage import io, morphology
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.filters import try_all_threshold
x = 'test'

fname = '/Users/porfirio/Documents/research/sternberg_lab/EnhancerAssay/images/C2_myo2_R5/gblockr5c2r6.tiff'
im = io.imread_collection(fname)
def fakeRGB2gray(im):
    """
    Check if an RGB image is grayscale and convert
    """
    # If the three channels are the same, it is grayscale, get only one
    if ((im[:,:,0] == im[:,:,1]) &  ( im[:,:,2]== im[:,:,0])).all():
        return im[:,:,0]
    raise ValueError('Channels are not identical')

def im_hist(im):
    hist, bins = skimage.exposure.histogram(im)
    with sns.axes_style('darkgrid'):
        plt.fill_between(bins, hist, lw=0.25, alpha=0.4)
        plt.yscale('log')
        plt.xlabel('normalized pixel value')
        plt.ylabel('count')

def rectangleROI(im, thresh=None):
    """
    Removes all continuous zero columns and rows after threshold
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
    # remove only continuous zero rows and columns
    roi = im[min(rows):max(rows)+1,min(cols):max(cols)+1]
    return roi

im = fakeRGB2gray(im[11])
# subregion histogram equalization to improve contrast
im = skimage.exposure.equalize_adapthist(im)
thresh = skimage.filters.threshold_li(im)
im = rectangleROI(im, thresh)
im[im<thresh] = 0
im_thresh = threshold_adaptive(im,15)
im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=200)
# two options to get defined nuclei masks for morphological reconstruction
# a)
im_thresh = ndimage.morphology.binary_closing(im_thresh)
# b)
im_thresh = scipy.ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
# then perform below operation on either output
im_thresh = morphology.binary_opening(im_thresh, morphology.disk(5))

# do morphological opening to enhance contrast between nuclei and background
im_open = morphology.opening(im, morphology.disk(5))
# get nuclei markers 
markers = peak_local_max(im_open, indices=False, threshold_abs=0.3, min_distance=10)
# make the mask fit the markers
im_thresh = markers+im_thresh
# reconstruct
markers = morphology.reconstruction(markers,im_thresh,selem=selem)
markers  = skimage.morphology.remove_small_objects(markers>0, min_size=100)
# uncomment below if finding peak local max on im_open
#markers = skimage.morphology.remove_small_objects(markers, min_size=50)
#markers = markers.astype(float)
#markers[markers>0] = 0.4
markers = morphology.label(markers)
fig, ax = plt.subplots(2)
ax[0].imshow(im)
ax[0].imshow(markers, alpha=0.3, cmap='viridis')
ax[1].imshow(im)
# get nuclei
nuclei = skimage.measure.regionprops(markers, im)
# get intensities
intensities = [n.mean_intensity for n in nuclei]
sns.stripplot(intensities, orient='v', size=10, alpha=0.5, color='b')
