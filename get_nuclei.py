from matplotlib import pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
sns.set_style('white')

import pandas as pd
import numpy as np

import warnings

import os
import glob

import skimage
from skimage import io, morphology
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.filters import threshold_adaptive
from skimage.draw import circle_perimeter
from scipy import ndimage

from scipy.spatial.distance import pdist, squareform
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree

min_size=15
def split_project(im_col):
    """
    Split stack by channels and z-project each channel
    """
    # convert image collection to numpy array
    stack = np.copy(im_col)
    num = len(stack)/2
    # gfp channel first half, rfp second
    gfp = stack[:num]
    rfp = stack[num:]
    gfp = z_project(gfp)
    rfp = z_project(rfp)
    return gfp, rfp

def z_project(stack):
    """
    Z-project based on maximum value.
    """
    z_im = np.maximum.reduce([z for z in stack])
    return z_im

def plot_gallery(images, n_row=3, n_col=4, fig_title=None):
    """
    Helper function to plot a gallery of portraits
    """
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i], cmap=plt.cm.viridis)
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
    return None

def rectangleROI(im, thresh=None):
    """
    Identify coordinates of rectangular ROI in an image with mostly dark irrelevant pixels
    The returned ROI excludes all continuous zero columns and rows after thresholding

    Arguments
    ---------
    im: input image

    Returns
    ---------
    roi_coords: slice object containing roi coordinates. 
                Can be directly used as numpy slice: im[roi_coords]
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

def label_sizesel(im, im_mask, max_int, min_int, minor_ax_lim, major_ax_lim, max_area):
    """
    Create and label markers from image mask, 
    filter by area and compute region properties

    Arguments
    ---------
    im: input image

    Returns
    ---------
    markers: labeled image, where each object has unique value and background is 0
    nuclei: list of region properties of each labeled object
 
    """
    markers = morphology.label(im_mask)
    nuclei = skimage.measure.regionprops(markers, im)
    # get only markers within area bounds, above intensity thersh and 
    # not oversaturated
    all_labels = np.unique(markers)
    sel_labels = [n.label for n in nuclei if n.minor_axis_length > minor_ax_lim
                    and n.major_axis_length < major_ax_lim \
                    and n.area < max_area \
                    and min_int < n.max_intensity < max_int]
    rem_labels = [l for l in all_labels if l not in sel_labels]
    # remove unselected markers
    for l in rem_labels:
        markers[np.where(np.isclose(markers,l))] = 0

    nuclei = [n for n in nuclei if n.label in sel_labels]

    return markers, nuclei

def int_sel(nuclei, min_int):
    """
    Drop nuclei by intensity threshold

    Arguments
    ---------
    nuclei: list of region properties of each labeled object
    min_int: minimum intensity threshold

    Returns
    ---------
    markers: labeled image, where each object has unique value and background is 0
    nuclei: list of region properties of each labeled object

    """
    sel_labels = [n.label for n in nuclei if min_int < n.mean_intensity]
    nuclei = [n for n in nuclei if n.label in sel_labels]
    int_ = np.array([n.mean_intensity for n in nuclei])
    return nuclei, int_

def nuclei_int(im_g, im_r, plot=False, min_distance=10, manual_selection=False,
        harsh=True):
    """
    get ratio of nuclei intensities and show which nuclei are being measured
    """

    if harsh:
        max_nuclei = 6
    else:
        max_nuclei = 16

    thresh_r = skimage.filters.threshold_yen(im_r)
    thresh_g = skimage.filters.threshold_yen(im_g)
    # Get identical ROI in of both images
    roi = rectangleROI(im_r, thresh_r)
    im_r = im_r[roi]
    im_g = im_g[roi]

    mask_r = mask_image(im_r)
    mask_g = mask_image(im_g)

    # Nuclei area and intensity bounds
    max_int =  2**cam_bitdepth - 1
    min_int = min(thresh_r, thresh_g)
    minor_ax_lim, major_ax_lim, max_area = 5, 30, 200

    markers_r, nuclei_r = label_sizesel(im_r, mask_r, max_int, min_int, minor_ax_lim, major_ax_lim, max_area)
    markers_g, nuclei_g = label_sizesel(im_g, mask_g, max_int, min_int, minor_ax_lim, major_ax_lim, max_area)

    if manual_selection:
        nuclei_r, markers_r, nuclei_g, markers_g = manual_sel(im_r, markers_r, 
            nuclei_r, im_g, markers_g, nuclei_g)

    # get nuclei intensities
    int_r = np.array([n.mean_intensity for n in nuclei_r])
    int_g = np.array([n.mean_intensity for n in nuclei_g])


    # Drop lowest intensity nuclei if the channels don't have the same number
    # Also, if there are more than 10 nuclei, its probably gut granules, increase
    # threshold until they are dropped
    while len(int_r) != len(int_g) or len(int_r) + len(int_g) > max_nuclei:
        if len(nuclei_r) > len(nuclei_g):
            nuclei_r, int_r = int_sel(nuclei_r, np.mean(int_r))
        elif len(nuclei_g) > len(nuclei_r):
            nuclei_g, int_g = int_sel(nuclei_g, np.mean(int_g))
        elif len(int_r) + len(int_g) > max_nuclei:
            nuclei_g, int_g = int_sel(nuclei_g, np.mean(int_g))

    int_ratio = int_g / int_r

    # Draw circles around identified nuclei for plotting
    im_plot_r = circle_nuclei(nuclei_r, im_r, diam=(10,))
    im_plot_g = circle_nuclei(nuclei_g, im_g, diam=(10,))
    io.imshow(im_plot_r)

    if plot: 
        plot_nuclei_int(im_plot_r, im_plot_g, int_ratio)

    return int_ratio, int_r, int_g, im_plot_r, im_plot_g

def circle_nuclei(nuclei, im, diam=(10, 12, 14)):
    """
    Draw circles around identified segments for plotting
    Arguments are region props objects, image and circle diameter (more than one
    draws multiple circles around each centroid)

    Arguments
    ---------
    nuclei: list of region properties of each labeled object
    im: corresponding image
    diam: diameter of circles to be drawn around each object; also determines
            the number of circles.

    Returns
    ---------
    im_plot: copy of the image with circles around each object
 
    """
    nuclei_c = [n.centroid for n in nuclei]
    circles = []
    for d in diam:
        circles += [circle_perimeter(int(r), int(c), d, shape=im.shape) for (r,c) in nuclei_c]
    im_plot = im.copy()
    for circle in circles:
        im_plot[circle] = np.max(im_plot)
    return im_plot

def plot_nuclei_int(im_plot_r, im_plot_g, int_ratio):
    """
    Plot two channels and intensity ratio side by side
    """
    gs = gridspec.GridSpec(1, 7)
    ax1 = plt.subplot(gs[0:3])
    ax2 = plt.subplot(gs[3:6])
    ax3 = plt.subplot(gs[6])
    ax1.imshow(im_plot_r, plt.cm.viridis)
    ax1.set_title('red channel nuclei', fontsize=20)
    ax2.imshow(im_plot_g, plt.cm.viridis)
    ax2.set_title('green channel nuclei', fontsize=20)
    sns.stripplot(int_ratio, orient='v', size=10, alpha=0.5, cmap='viridis', ax=ax3)
    ax3.set_ylabel('intensity ratio (gfp/rfp)', fontsize=20)
    plt.tight_layout()
    return None

def mask_image(im):
    """
    Create a binary mask to segment nuclei using adaptive threshold

    Arguments
    ---------
    im: input image

    Returns
    ---------
    im_thresh: thresholded binary image 
 
    """
    im_thresh = threshold_adaptive(im, 15)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    im_thresh = morphology.binary_opening(im_thresh)
    return im_thresh

def manual_sel(im_r, markers_r, nuclei_r, im_g, markers_g, nuclei_g):
    """
    Manual (click) confirmatory selection of nuclei
    """
    coords_rg = []
    for i in (1,3):
        # click on nuclei
        fig, axes = plt.subplots(1,4, figsize=(25,10))
        axes[0].imshow(im_r, plt.cm.viridis)
        axes[1].imshow(markers_r, plt.cm.Paired)
        axes[2].imshow(im_g, plt.cm.viridis)
        axes[3].imshow(markers_g, plt.cm.Paired)
        axes[i].set_title('Select markers here\n(Press Alt+Click when done)', fontsize=20)
        coords = plt.ginput(100, show_clicks=True)
        coords = [(int(c1), int(c2)) for (c2, c1) in coords]
        plt.close('all')
        coords_rg.append(coords)
    coords_r, coords_g = coords_rg

    def update_sel(markers, nuclei, coords):
        all_labels = np.unique(markers)
        sel_labels = [markers[c] for c in coords]
        rem_labels = [l for l in all_labels if l not in sel_labels]

        # get selected nuclei
        nuclei = [n for n in nuclei if n.label in sel_labels]
        # remove unselected markers
        for l in rem_labels:
            markers[np.where(np.isclose(markers,l))] = 0
        return markers, nuclei

    markers_r, nuclei_r = update_sel(markers_r, nuclei_r, coords_r)
    markers_g, nuclei_g = update_sel(markers_g, nuclei_g, coords_g)

    return nuclei_r, markers_r, nuclei_g, markers_g


cam_bitdepth = 16
data_dir = '../data/leica_data/' 
# get directories ignoring hidden files (glob does by default) (i.e. .DS_Store)
data_dirs = glob.glob(data_dir + '*')
strains = [s.split('/')[-1] for s in data_dir]
data_cols = ['strain','line','rep', 'int_ratio','int_r', 'int_g']
intensities = pd.DataFrame(columns = data_cols)
for d in data_dirs:
    im_dirs = glob.glob(d + '/*.tif')
    for im_dir in im_dirs:
        print('Processing {0}'.format(im_dir))
        # load image and process
        im_stack = io.imread_collection(im_dir)
        im_g, im_r = split_project(im_stack)
        int_ratio, int_r, int_g, im_plot_r, im_plot_g = nuclei_int(im_g, im_r, plot=0, harsh=False)
        break
    break
        # More than 3 nuclei is a bit suspicious, better take a look
        #if len(int_ratio) > 3:
        #    plt.close('all')
        #    plot_nuclei_int(im_plot_r, im_plot_g, int_ratio)
        # fill intensities
        curr_data = pd.DataFrame()
        curr_data['int_r'] = int_r
        curr_data['int_g'] = int_g
        curr_data['int_ratio'] = int_ratio

        # fill worm ID info
        info = im_dir.split('/')[-2:] 
        curr_data['strain'] = info[0].split('_')[0]
        curr_data['line'] = info[0].split('_', 1)[-1]
        curr_data['rep'] = info[1].split('.')[0]

        intensities = pd.concat([intensities, curr_data], ignore_index=True)
        print('Found {0} nuclei'.format(curr_data.shape[0]))
        print("Processed line {0} image {1}".format(*info))

def plot_intensities(intensities, to_plot='int_ratio',
        ylabel='Intensity ratio\n (GFP/RFP)', yscale=None, save=False):
    plt.figure()
    sns.swarmplot(x='strain', y=to_plot, data=intensities, alpha=0.5, size=6)
    #sns.swarmplot(x='strain', y='int_g', hue='line', data=intensities)
    medianprops = dict(linewidth=2, color='black')
    sns.boxplot(x='strain', y=to_plot, data=intensities,
            showcaps=False, boxprops={'facecolor':'None'}, showfliers=False,
            whiskerprops={'linewidth':0}, showbox=False, notch=True,
            medianprops=medianprops)
    plt.xticks(np.arange(4),['CTCF (+)\nnative binding site',
        'CTCF (+)\nmutated binding site','CTCF (-)\nnative binding site',
        'CTCF (-)\nmutated binding site'], rotation=45)
    plt.ylabel(ylabel)
    if yscale: plt.yscale(yscale)
    plt.tight_layout()
    sns.despine()
    if save: plt.savefig('./output/'+to_plot+'.pdf')
    return None

#intensities = pd.read_csv('./output/intensities.csv', comment='#')
plot_intensities(intensities, save=True)
plot_intensities(intensities, to_plot='int_r', ylabel='RFP', yscale='log', save=True)
plot_intensities(intensities, to_plot='int_g', ylabel='GFP', yscale='log', save=True)

# example dataframe of region props
nuclei_df = pd.DataFrame()
areas_ = [n.area for n in nuclei_r]
perims_ = [n.perimeter for n in nuclei_r]
intensity_ = [n.mean_intensity for n in nuclei_r]
nuclei_df['intensity'] = intensity_
nuclei_df['area'] = areas_
nuclei_df['perimeter'] = perims_
nuclei_df = pd.melt(nuclei_df)
fig, axes = plt.subplots(1,3)
sns.swarmplot(x='variable', y='value', data=nuclei_df[nuclei_df['variable']=='intensity'], alpha=0.5, size=15, ax=axes[0])
sns.swarmplot(x='variable', y='value', data=nuclei_df[nuclei_df['variable']=='area'], alpha=0.5, size=15, ax=axes[1])
sns.swarmplot(x='variable', y='value', data=nuclei_df[nuclei_df['variable']=='perimeter'], alpha=0.5, size=15, ax=axes[2])
for ax in axes: 
    ax.set_xlabel('')
    ax.set_ylabel('')
sns.despine()
plt.tight_layout()
plt.savefig()
