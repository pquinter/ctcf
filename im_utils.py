"""
Utilities for C. elegans image analysis 

Author: Porfirio Quintero-Cadena
"""

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(font_scale=2)
sns.set_style('white')

import pandas as pd
import numpy as np

import warnings
import glob

import skimage
from skimage import io, morphology
from skimage.filters import threshold_adaptive
from skimage.draw import circle_perimeter
from scipy import ndimage


def split_project(im_col):
    """
    Split stack by channels and z-project each channel

    Arguments
    ---------
    im_col: skimage image collection

    Returns
    ---------
    gfp, rfp: array_like
        split image collection and z-projected based on maximum value
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

def z_project(stack, project='max'):
    """
    Z-project stack based on maximum value.

    Arguments
    ---------
    stack: array_like. 
        input image stack
    project: str
        which value to project: maximum (max), minimum (min) or mean

    Returns
    ---------
    z_im: z-projection of image
    """

    if project == 'max':
        z_im = np.maximum.reduce([z for z in stack])
    if project == 'min':
        z_im = np.minimum.reduce([z for z in stack])
    if project == 'mean':
        z_im = np.mean.reduce([z for z in stack])

    return z_im

def plot_gallery(images, n_row=3, n_col=4, fig_title=None):
    """
    Helper function to plot a gallery of images

    Arguments
    ---------
    images: tuple or list of array_like objects
            images to plot
    n_row, n_col: integer
            number of rows and columns in plot
            If less than number of images, truncates to plot n_row * n_col
    fig_title: string
            optional figure title

    Returns
    ---------
    None (only plots gallery)
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
    Check if an RGB image is grayscale and convert if necessary.
    Useful because sometimes grayscale images are saved as redundant RGB

    Returns an error if image is not grayscale.
    Returns the same image if already grayscale and with single channel.

    Arguments
    ---------
    im: array_like, shape (h,w,3)
        image to convert

    Returns
    ---------
    im: array_like, shape (h,w,1)
        grayscale image, if indeed grayscale.
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
    Plot image pixel intensity histogram

    Arguments
    ---------
    im: array_like
        image to plot

    Returns
    ---------
    None (only plots histogram)
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
    im: array_like
        input image
    thresh: int or float
        threshold to use to discard background (

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
    im: array_like
        input image

    Returns
    ---------
    markers: array_like
        labeled image, where each object has unique value and background is 0
    nuclei: list of region props objects
        list of region properties of each labeled object
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
    nuclei: list of region props objects
        list of region properties of each labeled object
    min_int: int or float
        minimum intensity threshold

    Returns
    ---------
    markers: array_like
        labeled image, where each object has unique value and background is 0
    nuclei: list of region props objects
        list of region properties of each labeled object

    """
    sel_labels = [n.label for n in nuclei if min_int < n.mean_intensity]
    nuclei = [n for n in nuclei if n.label in sel_labels]
    int_ = np.array([n.mean_intensity for n in nuclei])
    return nuclei, int_

def circle_nuclei(nuclei, im, diam=(10, 12, 14)):
    """
    Draw circles around identified segments for plotting
    Arguments are region props objects, image and circle diameter (more than one
    draws multiple circles around each centroid)

    Arguments
    ---------
    nuclei: list of region props objects
        list of region properties of each labeled object
    im: array_like
        corresponding image
    diam: tuple of int
        diameter of circles to be drawn around each object; 
        also determines the number of circles.

    Returns
    ---------
    im_plot: array_like
        copy of the image with circles around each object
 
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

    Arguments
    ---------
    im_plot_r, im_plot_g: array_like
        images to plot
    int_ratio: list of int or float
        intensity ratios

    Returns
    ---------
    None (only plots)


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

def mask_image(im, thresh=None, min_size=15):
    """
    Create a binary mask to segment nuclei using adaptive threshold.
    Useful to find nuclei of varying intensities.
    Remove small objects, fill holes and perform binary opening (erosion
    followed by a dilation. Opening can remove small bright spots (i.e. “salt”)
    and connect small dark cracks. This tends to “open” up (dark) gaps between
    (bright) features.)

    Arguments
    ---------
    im: array_like
        input image
    thresh: array_like, optional
        thresholded image
    min_size: float or int
        minimum size of objects to retain

    Returns
    ---------
    im_thresh: array_like
        thresholded binary image 
    """
    if not thresh:
        im_thresh = threshold_adaptive(im, 15)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    im_thresh = morphology.binary_opening(im_thresh)
    return im_thresh

def manual_sel(im_r, markers_r, nuclei_r, im_g, markers_g, nuclei_g):
    """
    Manual (click) confirmatory selection of nuclei
    After automatic segmentation, allows user to select objects of interest by
    single click on them.
    Useful if automatic segmentation is difficult.

    Arguments
    ---------
    im_r, im_g: array_like
        original reference images
    markers_r, markers_g, array_like
        segmented, labeled images
    nuclei_r, nuclei_g: region props object
        region properties of labeled images

    Returns
    ---------
    nuclei_r, markers_r, nuclei_g, markers_g: updated (clicked on) objects
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

def clickselect_plot(event, selected, fig, axes):
    """
    Event handler for button_press_event.
    Save index of image clicked on, useful to view multiple images in 
    subplots, and select which ones to keep.

    Has to be called as follows: 
    cid = fig.canvas.mpl_connect('button_press_event', 
                    lambda event: clickselect_plot(event, selected, fig, axes))

    Arguments
    ---------
    event: matplotlib event 
        from fig.canvas.mpl_connect('button_press_event', onclick)
    selected: empty set
        store unique indices of clicked images

    Returns
    ---------
    None
        Adds indices to selected in place
    """
    for i, ax in enumerate(axes):
        if ax == event.inaxes:
            # Print which image is selected
            ax.set_title('Image selected')
            # update plot and save selection
            fig.canvas.draw()
            selected.add(i)

def mult_im_selection(data_dir, project='max', ext='.tif', limit=100):
    """
    Widget for image selection, z_projection, ROI cropping, and DIC-GFP overlay 
    from multiple samples stored in different directories.
    Shows images from nested directories sequentially, and allows user to click
    on those to keep.

    Arguments
    ---------
    datadir: string
        parent directory containing image folders
    ext: string
        extension of image files to look for 
    limit: integer
        upper limit of the number of directories to look at 

    Returns
    ---------
    im_selection: dictionary
        Dictionary with selected images and respective directory name

    """
    data_dirs = glob.glob(data_dir + '*')
    # dictionary to store sample name and selected images as key:values
    all_im = {}
    im_selection = {}
    # counter to limit number of dictionaries
    n=1
    # show images by directory/strain
    for d in data_dirs:
        im_dirs = glob.glob(d + '/*' + ext)
        try:
            fig, axes = plt.subplots(3, int(len(im_dirs)/3))
        except IndexError:
            fig, axes = plt.subplots(len(im_dirs))
        # get strain number
        strain = int(d.split('/')[-1].split('_')[0])
        curr_ims = []
        for (im_dir, ax) in zip(im_dirs, np.ravel(axes)):
            # load image
            im_stack = io.imread_collection(im_dir)
            # Get channels, DIC is always first array
            dic = im_stack[0]
            gfp = im_stack[1:]
            # Project gfp based on maximum value
            gfp = z_project(gfp, project=project)
            # Crop image based on gfp channel
            roi = rectangleROI(gfp)
            dic = dic[roi]
            gfp = gfp[roi]
            # save it for later, DIC goes first
            curr_ims.append((dic, gfp))
            # Plot DIC and overlay GFP
            ax.imshow(dic)
            ax.imshow(gfp, alpha=0.7, cmap=plt.cm.viridis)

        # set to store indices of selected images
        selected = set()
        # select images by clicking on them
        cid = fig.canvas.mpl_connect('button_press_event', 
                        lambda event: clickselect_plot(event, selected, fig, np.ravel(axes)))
        # Stop after 100 clicks or until the user is done
        fig.suptitle('Click on images to keep and press Alt+click when done', fontsize=20)
        plt.ginput(100, timeout=0, show_clicks=True)
        fig.canvas.mpl_disconnect(cid)
        plt.close('all')
        im_selection[strain] = [im for (i, im) in enumerate(curr_ims) if i in selected]
        n+=1
        if n>limit: break
    return im_selection

def mult_im_plot(im_dict, n_row=3, n_col=4, fig_title=None, sort=False):
    """
    Helper function to plot a gallery of images stored in dictionary 
    (output from mult_im_selection function)

    Arguments
    ---------
    im_dict: dictionary
            Dictionary with images to plot. Values must be pairs of (DIC, GFP)
    n_row, n_col: integer
            number of rows and columns in plot
            If less than number of images, truncates to plot n_row * n_col
    fig_title: string
            optional figure title
    sort: boolean
            whether to plot images sorted by key

    Returns
    ---------
    None (only plots gallery)
    """

    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    # counter to add axes
    j=1
    if sort: keys = sorted(im_dict)
    else: keys = im_dict.keys()
    for sample in keys:
        for (i, im) in enumerate(im_dict[sample], start=j):
            dic, gfp = im
            ax = fig.add_subplot(n_row, n_col, i)
            # Plot DIC and overlay GFP
            ax.imshow(dic)
            ax.imshow(gfp, alpha=0.7, cmap=plt.cm.viridis)
            ax.set_title(sample)
            plt.xticks(())
            plt.yticks(())
            j+=1
    if fig_title: fig.suptitle(fig_title+'\n', fontsize=20)
    plt.tight_layout()
