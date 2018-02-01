"""
Utilities for image analysis 

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
import os
from tqdm import tqdm

import skimage
from skimage import io, morphology, segmentation
from skimage.filters import threshold_local
from skimage.draw import circle_perimeter
from skimage.external.tifffile import TiffFile
from scipy import ndimage

from numba import jit
from bebi103_legacy import ecdf

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
        which value to project: maximum (max), minimum (min)

    Returns
    ---------
    z_im: z-projection of image
    """

    if project == 'max':
        z_im = np.maximum.reduce([z for z in stack])
    if project == 'min':
        z_im = np.minimum.reduce([z for z in stack])
    # np.mean does not have reduce
    #if project == 'mean':
    #    z_im = np.mean.reduce([z for z in stack])

    return z_im

def plot_gallery(images, n_row=3, n_col=4, reshape=None,
        cmap='viridis', titles=None, fig_title=None):
    """
    Helper function to plot a gallery of images

    Arguments
    ---------
    images: tuple or list of array_like objects
            images to plot
    n_row, n_col: integer
            number of rows and columns in plot
            If less than number of images, truncates to plot n_row * n_col
    reshape: None or tuple of two height and width int
            reshape each image to h and w
    cmap: string
        name of matplotlib colormap
    titles, fig_title: string
            optional subaxes and figure title

    Returns
    ---------
    None (only plots gallery)
    """

    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        if reshape:
            h, w = reshape
            ax.imshow(images[i].reshape((h, w)), cmap=cmap)
        else:
            ax.imshow(images[i], cmap=cmap)
        if titles:
            ax.set_title(titles[i], size=12)
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

def label_sizesel(im, im_mask, maxint_lim, minor_ax_lim, 
        major_ax_lim, area_lim, watershed=False):
    """
    Create and label markers from image mask, 
    filter by area and compute region properties

    Arguments
    ---------
    im: array_like
        input image
    im_mask: boolean array
        image mask
    watershed: boolean
        whether to perform watershed on markers
    feature_lim: iterable of two
        minimum and maximum bounds for each feature, inclusive

    Returns
    ---------
    markers: array_like
        labeled image, where each object has unique value and background is 0
    nuclei: list of region props objects
        list of region properties of each labeled object
    """
    markers = morphology.label(im_mask)
    if watershed:
        # harsh erosion to get basins for watershed
        im_mask_eroded = skimage.measure.label(\
                skimage.morphology.binary_erosion(im_mask,
                selem=skimage.morphology.diamond(8)))
        # watershed transform using eroded cells as basins
        markers = skimage.morphology.watershed(markers,
                im_mask_eroded, mask=im_mask)
    nuclei = skimage.measure.regionprops(markers, im)
    # get only markers within area bounds, above intensity thersh and 
    # not oversaturated
    all_labels = np.unique(markers)
    sel_labels = [n.label for n in nuclei if \
                    minor_ax_lim[0] <= n.minor_axis_length <= minor_ax_lim[1]
                    and major_ax_lim[0] <= n.major_axis_length <= major_ax_lim[1] \
                    and area_lim[0] <= n.area <= area_lim[1] \
                    and maxint_lim[0] <= n.max_intensity <= maxint_lim[1]]
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

def mask_image(im, im_thresh=None, min_size=15, block_size=None, selem=skimage.morphology.disk(15),
        clear_border=True):
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
    block_size: odd int
        block size for adaptive threshold, must be provided if im_thresh is None

    Returns
    ---------
    im_thresh: array_like
        thresholded binary image 
    """
    if im_thresh is None:
        im_thresh = im>threshold_local(im, block_size)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    im_thresh = ndimage.morphology.binary_fill_holes(im_thresh, morphology.disk(1.8))
    im_thresh = morphology.binary_opening(im_thresh, selem=selem)
    im_thresh = skimage.morphology.remove_small_objects(im_thresh, min_size=min_size)
    if clear_border:
        im_thresh = skimage.segmentation.clear_border(im_thresh)
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

def clickselect_plot(event, selected, fig, axes, heart=True):
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
    figure: matplotlib figure
        figure containing axis to click on 
    ax: matplotlib axis
        axis to click on

    Returns
    ---------
    None
        Adds indices to selected in place
    """
    for i, ax in enumerate(axes):
        if ax == event.inaxes:
            # Show which image is selected
            ax.set_title('Liked!')
            if heart:
                draw_heart(ax)
            # update plot and save selection
            fig.canvas.draw()
            selected.add(i)

def draw_heart(ax):
    """
    Draw a red heart on ax.

    Arguments
    ---------
    ax: matplotlib axis
        axis to draw heart on 

    Returns
    ---------
    None
        Just plots heart for image selection tool (click_select_plot)
    """

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    xheart = np.mean(x_lim)
    yheart = np.mean(y_lim)
    t = np.linspace(0, 2 * np.pi, 200)
    x = -10*(16 * np.sin(t)**3) + xheart
    y = -10*(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) + yheart
    ax.fill_between(x, y, color='red', alpha=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def burn_scale_bar(im, width=6, white=True, zoom=40, pixel_dist='leica'):
    """
    Burn a horizontal scale bar
    Use interpixel_dist function to compute interpixel distance for other setup

    Arguments
    ---------
    im: array_like
        image to copy and draw scale bar on
    width: int
        width of the scale bar
    white: boolean
        color of scale bar (maximum or minimum)
    zoom: int
        optical zoom/lens used
    pixel_dist: dict or string (default only)
        dictionary with key, value pairs for scale bar dimensions
        as zoom:(length_pixels, length_microns)
        Default settings are for Leica microscope (Ca+2 imaging)

    Returns
    --------
    im_out: array_like
        copy of im with labeled scale bar
    legend_x, legend_y: int
        coordinates for scale bar legend
    legend: string
        label for scale bar (e.g. 10 microns, 20 microns...)
    """

    # Modify a copy of the image just in case
    im_out = im.copy()
    # Position of scale bar (top left corner)
    j_pos = legend_x = im.shape[1]*0.08
    i_pos = im.shape[0] * 0.9
    # Position of scale bar length label (above bar)
    legend_y = im.shape[0]*0.88

    # Dictionary with interpixel distance calibrated for Leica
    # key value pairs are: zoom:(length_pixels, length_microns)
    if pixel_dist == 'leica':
        pixel_dist = {10:(75,'100'), 20:(70,'50'), 40:(56,'20'), 63:(40, '10')}
    length, legend = pixel_dist[zoom]

    if white: pixel_val = np.max(im)
    else: pixel_val = np.min(im)

    # burn scale bar, not vector graphics, in case image is improperly reshaped
    im_out[i_pos-(width//2):i_pos+(width//2), j_pos:j_pos+length] = pixel_val

    return im_out, (legend_x, legend_y, str(legend))

def interpixel_dist(im, ref_length):
    """
    Compute interpixel distance by measuring an object of known length.
    Prompts the image and allows the user to click on the two ends of the object.

    Arguments
    ---------
    im: array_like
        image with reference object
    ref_length: int or float
        length of the object

    Returns
    ---------
    interpixel_distance: float
        number of pixels per unit of reference length
    """
    io.imshow(im)
    xy = plt.ginput(2)
    feature_length = np.sqrt((xy[1][1] - xy[0][1])**2 + (xy[1][0] - xy[0][0])**2)
    interpixel_distance = feature_length / ref_length
    return interpixel_distance

def mult_im_selection(data_dir, has_dic=True, project='max', ext='.tif', limit=100, 
        heart=True, save=False, crop_roi=False, overlay=0.9, plot=False):
    """
    Widget for image selection, z_projection, ROI cropping, and DIC-GFP overlay 
    from multiple samples stored in different directories.
    Shows images from nested directories sequentially, and allows user to click
    on those to keep.

    Arguments
    ---------
    datadir: string
        parent directory containing image folders
    has_dic: boolean
        whether image stacks have DIC, should be first in stack
    project: string
        project z-stacks based on max, mean or min value.
    ext: string
        extension of image files to look for 
    limit: integer
        upper limit of the number of directories to look at 
    heart: boolean
        whether to draw a heart on liked images
    save: boolean
        whether to save selected images to disk
    crop_roi: string or False
        whether to find rectangular ROI using dic, gfp, or nothing
    overlay: float
        alpha value to overlay GFP on DIC. 1 means completely block DIC
    plot: boolean
        whether to plot selection of images and save to pdf

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
    # start interactive mode
    plt.ion()

    # show images by directory/strain
    for d in data_dirs:

        im_dirs = glob.glob(d + '/*' + ext)
        try:
            fig, axes = plt.subplots(3, int(len(im_dirs)/3))
        except IndexError:
            fig, axes = plt.subplots(1)
        # get strain number
        strain = d.split('/')[-1].split('_')[0]
        curr_ims = []

        for (im_dir, ax) in zip(im_dirs, np.ravel(axes)):
            # get image name
            im_name = im_dir.split('/')[-1]
            # get zoom, if encoded in image name
            if 'x' in im_name:
                zoom = int(im_name.split('_')[-1].split('x')[0])
            else: zoom = 40
            # load image
            im_stack = io.imread_collection(im_dir)
            # Get channels, DIC is always first array
            dic = im_stack[0]
            if has_dic:
                gfp = im_stack[1:]
            else: gfp = im_stack[0:]
            # Project gfp based on maximum value
            gfp = z_project(gfp, project=project)
            # Find and crop ROI
            if crop_roi == 'gfp':
                roi = rectangleROI(gfp)
            elif crop_roi == 'dic':
                roi = rectangleROI(dic)
            if crop_roi:
                dic = dic[roi]
                gfp = gfp[roi]
            # save it for later, DIC goes first
            curr_ims.append((dic, gfp, zoom))
            # Plot DIC and overlay GFP
            ax.imshow(dic)
            ax.imshow(gfp, alpha=overlay, cmap=plt.cm.viridis)
            ax.set_title(im_name)
            ax.set_xticks([])
            ax.set_yticks([])

        # set to store indices of selected images
        selected = set()
        # select images by clicking on them
        cid = fig.canvas.mpl_connect('button_press_event', 
                        lambda event: clickselect_plot(event, selected, fig, np.ravel(axes), heart))
        # Stop after 100 clicks or until the user is done
        fig.suptitle('Click to like, right click (Alt+click) when done\nsample:{}\n'.format(strain), fontsize=15)
        #plt.tight_layout()
        plt.ginput(100, timeout=0, show_clicks=True)
        fig.canvas.mpl_disconnect(cid)
        plt.close('all')
        im_selection[strain] = [im for (i, im) in enumerate(curr_ims) if i in selected]
        n+=1
        if n>limit: break

    if save: save_imdict('./favorite_worms/', im_selection, has_dic)
    if plot: mult_im_plot(im_selection, save=True)

    return im_selection

def save_imdict(save_dir, im_selection, has_dic):
    """
    Save a dictionary of images to structured directory

    Arguments
    ---------
    save_dir: string
        root directory to save
    im_selection: dict
        dictionary with sample:(images, zoom)
    
    Returns
    --------
    None
        Saves images and prints save_dir
    """
    os.mkdir(save_dir)
    for sample in im_selection:
        save_subdir = save_dir + str(sample) + '/'
        os.mkdir(save_subdir)
        for i, image in enumerate(im_selection[sample], start=1):
            # each entry in im_selection is (dic, gfp, zoom)
            if has_dic:
                im_ = np.stack(image[:-1])
            else: im_ = image[1]
            zoom = image[-1]
            io.imsave(save_subdir + str(i) + '_' + str(zoom) + 'x.tif', im_)
    print('Images saved to {}'.format(save_dir))

def imdict_fromdir(data_dir):
    """
    Load images from multiple subdirs in data_dir to dictionary

    Arguments
    ---------
    data_dir: string
        root directory containing subirectories with images

    Returns
    ---------
    im_collection: dictionary
        dict containing sample:(dic, gfp, zoom)
    """
    data_dirs = glob.glob(data_dir + '*')
    im_collection = {}
    for d in data_dirs:
        strain = d.split('/')[-1].split('_')[0]
        im_dirs = glob.glob(d + '/*' + '.tif')
        ims = []
        for im_dir in im_dirs:
            im_name = im_dir.split('/')[-1]
            im_stack = io.imread_collection(im_dir)
            # Get channels, DIC is always first array
            dic = im_stack[0]
            gfp = im_stack[1]
            # Get zoom
            if 'x' in im_name:
                zoom = int(im_name.split('_')[-1].split('x')[0])
            else: zoom = 40
            ims.append((dic, gfp, zoom))
        im_collection[strain] = ims
    return im_collection

def mult_im_plot(im_dict, n_row='auto', n_col='auto', fig_title=None, sort=True, 
        overlay=0.7, scale_bar=True, scale_font=8, save=False):
    """
    Helper function to plot a gallery of images stored in dictionary 
    (output from mult_im_selection function)

    Arguments
    ---------
    im_dict: dictionary or str
            Dictionary with images to plot. Values must be pairs of (DIC, GFP)
            Also takes path to image directories that can be loaded with imdict_fromdir function
    n_row, n_col: integer or str
            number of rows and columns in plot
            If less than number of images, truncates to plot n_row * n_col
            If auto, automatically determine
    fig_title: string
            optional figure title
    sort: boolean
            whether to plot images sorted by key
    overlay: float
            alpha value for GFP channel (0-1). If 1, then completely hide DIC
    save: boolean
        whether to save plot to pdf

    Returns
    ---------
    None (only plots gallery)
    """

    # Load image dictionary if required
    if isinstance(im_dict, str):
        # add slash if not included in path
        if im_dict[-1] != '/': im_dict += '/'
        im_dict = imdict_fromdir(im_dict)

    # get appropiate number of rows and columns for plot
    if n_row and n_col == 'auto':
        num_ims = len([im for k in im_dict for im in im_dict[k]])
        if num_ims > 4:
            n_col = num_ims//2
        else: n_col = num_ims
        # compute number of rows required
        if num_ims % n_col ==0:
            n_row = (num_ims // n_col)
        else: n_row = (num_ims // n_col) + (num_ims % n_col)

    # whether to sort by name
    if sort: keys = sorted(im_dict)
    else: keys = im_dict.keys()

    plt.ion()
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    # counter to add axes
    j=1
    for sample in keys:
        for (i, im) in enumerate(im_dict[sample], start=j):
            try:
                # get zoom for scale bar; if not specified, use default
                # regardless of whether or not it is drawn (specified later)
                dic, gfp, zoom = im
            except ValueError:
                dic, gfp = im
                # default zoom
                zoom = 40

            if scale_bar:
                gfp, (scale_x, scale_y, scale_legend) = burn_scale_bar(gfp, zoom=zoom)

            # Create subplot, plot DIC and overlay GFP
            ax = fig.add_subplot(n_row, n_col, i)
            ax.imshow(dic, cmap=plt.cm.gray)
            ax.imshow(gfp, alpha=overlay, cmap=plt.cm.viridis)
            if scale_bar:
                # Add scale bar label (microns)
                ax.text(scale_x, scale_y,  r'$' +scale_legend + ' \mu m$', color='yellow', fontsize=scale_font)
            ax.set_title(sample)
            plt.xticks(())
            plt.yticks(())
            j+=1
    if fig_title: fig.suptitle(fig_title+'\n', fontsize=20)
    plt.tight_layout()
    if save: plt.savefig('./favorite_worms.pdf', transparent=True, bbox_inches='tight')

def zoom2roi(ax):
    """
    Identify coordinates of zoomed-in/moved axis in interactive mode

    Arguments
    ---------
    ax: matplotlib axis
        zoomed-in/moved axis

    Returns
    ---------
    zoom_coords: tuple of slice objects
        coordinates of zoomed in box, use as im[zoom_coords]

    """
    # get x and y limits
    xlim = [int(x) for x in ax.get_xlim()]
    ylim = [int(y) for y in ax.get_ylim()]

    # make and return slice objects
    return (slice(ylim[1],ylim[0]), slice(xlim[0],xlim[1]))

def show_movie(stack, delay=0.5, cmap='viridis', h=10, w=6, time=None,
        loop=False):
    """
    Show movie from stack

    Arguments
    ---------
    stack: numpy stack
        collection of 2D frames (movie)
    delay: float
        delay in between frames
    h, w: int
        height and width of movie display
    time: int or None
        inter-frame time to display on top of movie, in seconds
    loop: boolean
        whether to loop the movie

    Returns
    ---------
    None
        plays movie
    """
    fig = plt.figure(figsize=(w, h))
    while True:
        for n, frame in enumerate(stack):
            try:
                mov.set_data(frame)
            except NameError:
                mov = plt.imshow(frame, cmap=cmap)
                plt.tight_layout()
            plt.xticks(())
            plt.yticks(())
            plt.draw()
            plt.pause(delay)
            if time:
                plt.title('t {}s'.format(n*time))
            else:
                plt.title('frame {}'.format(n+1))
        if not loop: break

def resize_frame(frame, h, w, fillvalue='min'):
    """ 
    Resize frame to larger shape (h,w) by filling with 'fillvalue'

    Arguments
    ---------
    frame: numpy array
    h, w: int
        new height and width
    fillvalue: str or number
        'min', 'zeros' or number, value to fill empty spaces

    Returns
    ---------
    new_frame: numpy array
        resized frame
    """
    if fillvalue=='min':
        new_frame = np.full((h,w), np.min(frame))
    elif fillvalue=='zeros':
        new_frame = np.zeros((h,w))
    else:
        new_frame = np.full((h,w), fillvalue)
    new_frame[:frame.shape[0], :frame.shape[1]] = frame
    return new_frame

def concat_movies(movies, nrows=1):
    """
    Concatenate a set movies frame by frame to play multiple simultaneously

    Arguments
    ---------
    movies: iterable of numpy stacks
        movies to show
    rows: int
        number of rows to display movies into, 
        will add rows if necessary, i.e. nmovies%nrows!=0

    Returns
    ---------
    conc_mov: numpy stack
        concatenated movies, can be played using show_movie
    """
    # max number of frames per movie
    n_frames = max([len(l) for l in movies])
    # add black frames if necessary
    if not all(len(x) == n_frames for x in movies):
        _movies = []
        for m in movies:
            while len(m)< n_frames:
                m = np.append(m, [np.zeros_like(m[0])], axis=0)
            _movies.append(m)
    try: movies = _movies
    except NameError: pass
    # number of movies per column
    mpc = int(len(movies)/nrows)
    # maximum frame height and width
    max_h = max([m[0].shape[0] for m in movies])
    max_w = max([m[0].shape[1] for m in movies])
    # minimum pixel value to fill empty spaces
    min_val = (min([np.min(m) for m in movies])) 
    # new concatenated movie
    conc_mov = []
    # divide into sets of movies per row
    movs_byrow = [movies[i:i+mpc] for i in range(0, len(movies), mpc)]
    for f in range(n_frames):
        currframes = []
        for movrow in movs_byrow:
            while len(movrow) < mpc:
                # fill in empty columns with black frames if needed
                movrow.append(np.full_like(movrow[0], np.min(movrow[0])))
            # concatenate each (same sized) frame of movies in row
            currframes.append(np.concatenate([resize_frame(mov[f], 
                            max_h, max_w, min_val) for mov in movrow], axis=1))
        conc_mov.append(np.vstack(currframes))
    return np.stack(conc_mov)

def regionprops2df(regionprops, props = ('label','area','bbox',
    'intensity_image', 'mean_intensity','max_intensity','min_intensity')):
    """
    Convert list of region properties to dataframe

    Arguments
    ---------
    regionprops: list of skimage.measure._regionprops objects
    props: list of str, properties to store

    Returns
    ---------
    Pandas DataFrame with region properties
    """
    if not isinstance(regionprops, list): regionprops = [regionprops]
    return pd.DataFrame([[r[p] for p in props] for r in regionprops],columns=props)

def tracking_movie(movie, tracks, x='x', y='y'):
    """
    Label particles being tracked on movie for visualization

    Arguments
    ---------
    movie: array
    tracks: pandas dataframe
        containing columns `x`, `y` and `frame` for each particle being tracked

    Returns
    ---------
    movie_tracks: array
        copy of movie with circles around each identified particle

    """
    movie_tracks = np.empty_like(movie)
    # correct frame counter in case movie skips frames
    if len(movie) == len(tracks.frame.unique()):
        frame_counter = tracks.frame.unique()
    else: frame_counter = np.arange(len(movie))
    for f, im in enumerate(movie):
        coords = tracks[tracks.frame==frame_counter[f]][[x, y]].dropna()
        im_plot = im.copy()
        try:
            circles = [circle_perimeter(int(c[y]), int(c[x]), 10,
                        shape=im.shape) for (_, c) in coords.iterrows()]
        # if nan, no coordinates specified for frame, just put image
        except ValueError:
            movie_tracks[f] = im_plot
            continue
        for circle in circles:
            im_plot[circle] = np.max(im_plot)
        movie_tracks[f] = im_plot
    return movie_tracks

@jit(nopython=True)
def normalize(movie):
    """
    min-max scaler for a movie, to fix each frame in the range [0, 1]

    Arguments
    ---------
    movie: array
        movie to normalized

    Returns
    ---------
    scaled: array
        normalized copy of the movie
    """
    scaled = np.empty(movie.shape) # for numba it is necessary to use np.empty because float is default dtype
    for i in range(len(movie)):
        frame = movie[i]
        scaled[i] = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    return scaled

@jit(nopython=True)
def normalize_im(im):
    """
    min-max scaler for an image, to fix int values in the range [0, 1]

    Arguments
    ---------
    im: array
        im to normalize

    Returns
    ---------
    scaled: array
        normalized copy of the image
    """
    scaled = (im - np.min(im)) / (np.max(im) - np.min(im))
    return scaled

def load_zproject_STKcollection(load_pattern, savedir=None):
    """
    Load collection or single STK files and do maximum intensity projection

    Arguments
    ---------
    load_pattern: str
        pattern of file paths
    savedir: str
        directory to save projected images

    Returns
    ---------
    projected: nd array or np stack
        projected images

    """
    collection = io.ImageCollection(load_pattern, load_func=TiffFile)
    projected = [z_project(zseries.asarray()) for zseries in collection]
    if len(collection)>1:
        projected = np.stack(projected)
    else: projected = projected[0]
    if savedir:
        io.imsave(savedir, projected)
    return projected

def im_block(ims, cols, norm=True):
    """
    Construct block of images

    Arguments
    ---------
    ims: array or iterable of arrays
        images to concatenate in block
    cols: int
        number of columns in image block
    norm: bool
        whether to normalize/scale each image

    Returns
    ---------
    block: array

    """
    if norm:
        ims = normalize(ims)
    nrows = int(ims.shape[0]/cols)
    xdim, ydim = ims.shape[1:]
    block = []
    for c in np.arange(0, cols*nrows, cols):
        block.append(np.hstack(ims[c:c+cols]))
    block = np.vstack(block)
    return block

def get_bbox(center, size=9, im=None, return_im=True, pad=2, mark_center=False,
        size_z=None):
    """
    Get square bounding box around center

    Arguments
    ---------
    center: tuple
        x, y coordinates
    size: int
        size of the bounding box in pixels.
        Must be an odd number, or it will be round up.
    im: 2D array
        image to extract window from
    return_im: bool
        whether to return the bbox image or just coordinates
    pad: int
        size of padding around returned image
    mark_center: bool
        whether to draw a cross to mark image center

    Returns
    ---------
    im_bbox or bbox: array or numpy slice
        image bounding box

    """
    x, y = center
    x, y = int(x), int(y)
    # get bbox coordinates
    # if can't get full bbox, get partial and avoid negative values
    s = size//2
    ys = np.max((0, y-s))
    xs = np.max((0, x-s))
    bbox = np.s_[ys:y+s+1, xs:x+s+1]
    if return_im:
        # get bbox image
        im_bbox = im[bbox].copy()
        if mark_center:
            im_bbox[s-1:s+2,s] = np.min(im_bbox)
            im_bbox[s,s-1:s+2] = np.min(im_bbox)
        if pad:
            im_bbox = np.pad(im_bbox, pad, 'constant', constant_values=np.min(im_bbox))
        return im_bbox
    else: return bbox

def get_batch_bbox(bbox_df, ims_dict, size=9, movie=False,
        pad=0, mark_center=0, im3d=False, size_z=None, coords_col=None):
    """
    Get square bounding boxes in batch around center coords from dataframe

    Arguments
    ---------
    bbox_df: DataFrame
        df with object coordinates and corresponding image name.
        Must contain columns ['x','y','imname'] and 'frame' for movie
    ims_dict: dictionary
        dict of images. Keys must be the same as `imname`s in peaks_df
    size: int
        xy size of bounding box to get from image around object coordinates.
        If 3d, size in z is also this, unless Full is specified
    movie: bool
        True if ims_dict contains movies
    pad: int
        Padding to add to each image.
    mark_center: bool
        whether to mark center of each image
    im3d: bool
        whether image is 3D or not
    size_z: 'Full' or None
        If 'Full', get all z planes. Otherwise get same number as xy size.
    coords_col: list, str or None
        name of column(s) containing coordinates

    Returns
    ---------
    ims_array: stack or list
        stack of images if possible. If different sized arrays, returns list.

    """

    if im3d:
        _getbboxfunc = get_bbox3d
        coords = ['x','y','z']
    else:
        _getbboxfunc = get_bbox
        coords = ['x','y']
    if coords_col:
        coords = coords_col
    if movie:
        ims_df = bbox_df.apply(lambda x: [_getbboxfunc(x[coords], size,
                ims_dict[x.imname][int(x.frame)], mark_center=mark_center, pad=pad, size_z=size_z)], axis=1)
    else:
        ims_df = bbox_df.apply(lambda x: [_getbboxfunc(x[coords], size,
                ims_dict[x.imname], mark_center=mark_center, pad=pad, size_z=size_z)], axis=1)
    try:
        return np.stack([i[0] for i in ims_df])
    except ValueError:
        warnings.warn('Encountered images of different Z depths. Returning list of arrays.')
        return [i[0] for i in ims_df]

def get_bbox3d(center, size=9, im=None, return_im=True, pad=2, mark_center=False, size_z=None):
    """
    Get square bounding box around center

    Arguments
    ---------
    center: tuple
        x, y coordinates
    size: int
        size of the bounding box in pixels in xy dimensions.
        Get all z-stacks by default.
    im: 2D array
        image to extract window from
    return_im: bool
        whether to return the bbox image or just coordinates
    size_z: 'Full' or None
        If 'Full', get all z planes. Otherwise get same number as xy size.

    Returns
    ---------
    im_bbox or bbox: array or numpy slice
        image bounding box

    """
    x, y, z = center
    x, y, z = int(x), int(y), int(z)
    # get bbox coordinates
    s = size//2
    if size_z=='Full':
        bbox = np.s_[:,y-s:y+s+1, x-s:x+s+1]
    else:
        bbox = np.s_[z-s:z+s+1,y-s:y+s+1, x-s:x+s+1]
    if return_im:
        # get bbox image
        im_bbox = im[bbox].copy()
        if pad:
            im_bbox = np.pad(im_bbox, pad, 'constant', constant_values=np.min(im_bbox))
        return im_bbox
    else: return bbox

def check_borders(coords, im, s):
    """
    Check if coords are closer than `s` pixels to borders of `im`
    Return False if too close, convenient for indexing

    Arguments
    ---------
    coords: tuple
        x, y coordinates
    im: 2D array
        image to check
    s: int
        distance to border to check, in pixels

    Returns
    ---------
    boolean
        True if within border, False if not

    """
    try:
        dimx, dimy = im.shape
    # if it is a movie, get dimensions
    except ValueError:
        dimz, dimx, dimy = im.shape
    x, y = coords
    return (x>s)&(x+s<dimx)&(y>s)&(y+s<dimy)

def sel_training(peaks_df, ims_dict, s=9, ncols=10, cmap='viridis', scale=1,
        mark_center=True, movie=False, normall=False, figsize=(25.6, 13.6),
        step=None, title=''):
    """
    Manual click-selection of training set.
    Use a large screen if number of candidate objects is large!

    Arguments
    ---------
    peaks_df: DataFrame
        df with object coordinates and corresponding image name.
        Must contain columns ['x','y','imname'] and 'frame' for movie
    ims_dict: dictionary
        dict of images. Keys must be the same as `imname`s in peaks_df
    s: int
        size of bounding box to get from image around object coordinates
    cmap: str
        matplotlib colormap to display images with
    scale: float
        how to scale images to improve visibility for selection.
        Range is scaled from min to scale*max
    mark_center: bool
        whether to mark center of each image for selection
    movie: bool
        True ims_dict contains movies
    normall: bool
        Whether to normalize each image to [0,1] range.
        Might help visualization but hurt image comparison.
    figsize: tuple
        Pair of floats specifying figure size. Default is for big monitor.
    step: None or int
        if integer, make selection in steps of this size
    title: str
        instruction to print on top of selection plot

    Returns
    ---------
    sel_bool: boolean array
        Can be used to index original `peaks` dataframe. True for selected ims.
    all_ims: array
        Screened objects. To get selected images, index: all_ims[sel_bool]
    peaks: dataframe
        copy of original dataframe without image patches too close to border

    """

    if step:
        sel_bool, all_ims, peaks = [], [], pd.DataFrame()
        for n in np.arange(step, len(peaks_df)+step, step):
            _sel_bool, _all_ims, _peaks = sel_training(peaks_df.iloc[n-step:n],
                ims_dict, ncols=ncols, mark_center=mark_center, s=s, cmap=cmap,
                normall=normall, figsize=figsize, step=None, title=title)
            sel_bool.append(_sel_bool)
            all_ims.append(_all_ims)
            peaks = pd.concat((peaks, _peaks), ignore_index=True)
        return np.concatenate(sel_bool), np.concatenate(all_ims), peaks

    peaks = peaks_df.copy()
    # clear peaks too close to image border
    not_inborder = peaks.apply(lambda x: check_borders(x[['x','y']],
                                            ims_dict[x.imname], s), axis=1)
    peaks = peaks.loc[not_inborder]
    # add unique identifier
    peaks['uid'] = np.arange(len(peaks))
    # get s by s squares containing spots
    if movie: # also need to get frame
         peaks_ims = peaks.apply(lambda x: [get_bbox(x[['x','y']], s,
                ims_dict[x.imname][x.frame], mark_center=mark_center)], axis=1)
    else:
        peaks_ims = peaks.apply(lambda x: [get_bbox(x[['x','y']], s,
                ims_dict[x.imname], mark_center=mark_center)], axis=1)
    # append extra frames if necessary to make square array with ncols
    extra_frames, im_shape = len(peaks_ims)%ncols, peaks_ims.iloc[0][0].shape
    add_frames=0
    if extra_frames > 0:
        add_frames = ncols-extra_frames
        peaks_ims = peaks_ims.append(pd.Series([[np.zeros(im_shape)]\
                                for f in range(add_frames)]))
    # concatenate squares for selection (normalize each image if requested)
    peaks_ims = np.stack([i[0] for i in peaks_ims])
    peaks_imsconcat = im_block(peaks_ims, cols=ncols, norm=normall)
    # scale dynamic range to improve visibility, if scale<1
    peaks_imsconcat = np.clip(peaks_imsconcat, np.min(peaks_imsconcat),
            scale*np.max(peaks_imsconcat))
    # create s by s squares with labels to track selection and concatenate
    labels = np.stack([np.full(im_shape, l) for l in np.concatenate((peaks.uid.values,
                                np.full(add_frames, np.max(peaks.uid)+1)))])
    labels_concat = im_block(labels, cols=ncols, norm=0)
    # display for click selection
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_title(title+'\nclick to select; ctrl+click to undo last click; alt+click to finish')
    ax.imshow(peaks_imsconcat, cmap=cmap)# array of frames for visual sel
    ax.imshow(labels_concat, alpha=0.0)# overlay array of squares with invisible labels
    # yticks for guidance, take into account padding
    ax.set_yticks(np.arange(s+4, len(peaks_ims)/ncols*1.1*(s+4), 10))
    plt.tight_layout()
    # get labels by click
    coords = plt.ginput(10000, timeout=0, show_clicks=True)
    plt.close('all')
    if len(coords)>0:
        # filter selected labels
        selected = {labels_concat[int(c1), int(c2)] for (c2, c1) in coords}
    else: selected = []
    # get boolean array of selected for indexing original df
    sel_bool = peaks.uid.isin(selected).values
    # get selected images, without padding nor normalizing. Need to fetch originals again
    if movie: # also need to get frame
         peaks_ims = peaks.apply(lambda x: [get_bbox(x[['x','y']], s,
                ims_dict[x.imname][x.frame], pad=False)], axis=1)
    else:
        peaks_ims = peaks.apply(lambda x: [get_bbox(x[['x','y']], s,
                    ims_dict[x.imname], pad=False)], axis=1)
    all_ims  = np.stack([i[0] for i in peaks_ims])
    return sel_bool, all_ims, peaks

def classify_spots_from_df(spot_df, clf, im_dict, s, movie=False, norm=False):
    """
    Classify images from dataframe

    Arguments
    ---------
    spot_df: dataframe
        Must contain columns ['x','y','imname'] for images and 'frame' for movie
    clf: classifier object
    im_dict: dictionary
        with images to extract spots images frmo
    s: int
        size of bounding box to get from image around object coordinates
    movie: bool
        whether its a movie or not

    Returns
    -------
    spot_clf: dataframe
        copy of original dataframe with predicted labels, with cleared borders
    spot_ims: array
        all the images specified in spot_df

    """
    spot_clf = spot_df.copy()
    # clear spot_clf too close to borders
    spot_clf = spot_clf[spot_clf.apply(lambda x: check_borders(x[['x','y']],
                        im_dict[x.imname], s), axis=1)].reset_index(drop=True)
    spot_ims = get_batch_bbox(spot_clf, im_dict, size=s, movie=movie)
    if norm:
        spot_ims = normalize_im(spot_ims)
    # ravel for classif
    spot_ims_r = np.stack([np.ravel(i) for i in spot_ims])
    # classify
    labels_pred = clf.predict(spot_ims_r)
    # add labels
    spot_clf['svm_label'] = labels_pred
    return spot_clf, spot_ims

def filter_parts(spot_df, thresh=2):
    """
    Filter out trajectories less than 'thresh' number of points matching label

    Arguments
    ----------
    spot_df : DataFrame
        must include columns named 'pid' with unique particle id
        and 'plabel' with particle label (True or False)
    thresh: int
        minimum number of True labeled points in trajectory to survive

    Returns
    -------
    good_parts: DataFrame
        subset of trajectories

    """
    pids, counts = np.unique(spot_df[spot_df.plabel==True].pid.values, return_counts=True)
    good_parts = pids[counts>2]
    good_parts = spot_df[spot_df.pid.isin(good_parts)]
    return good_parts

def plot_ecdf(arr, ax=None, alpha=0.3, formal=0, label=''):
    if ax==None: fig, ax = plt.subplots(1)
    if formal:
        ax.plot(*ecdf(arr, conventional=formal), alpha=alpha, label=label)
    else:
        ax.scatter(*ecdf(arr, conventional=formal), s=15, alpha=alpha, label=label)

def load_ims(rdir, ext, channel=None):
    """ Load images to dictionary """
    ims = {}
    for _dir in tqdm(os.listdir(rdir)):
        if 'DS_Store' in _dir: continue
        ddir = rdir + _dir + '/'
        for fname in os.listdir(ddir):
            if ext=='tif' and ext in fname:
                im = io.imread(ddir + fname)
                if isinstance(channel, int):
                    im = im[:,:,channel]
            elif ext=='STK' and ext in fname:
                collection = io.ImageCollection(ddir+fname, load_func=TiffFile)
                im = np.stack([z.asarray() for z in collection])[0]
            else: continue
            _fname = fname.split('_')
            ims[_fname[1]+'_'+_fname[2].split('.')[0]] = im
    return ims

def segment_from_seeds(im, seed_markers, mask_params, dilate=False):
    """
    Segment cells by reconstructing from nuclei markers using watershed

    Arguments
    ---------
    im: array
        image to segment
    seed_markers: array
        integer labeled image of seeds to expand from for watershed
    mask_params: tuple of 3
        min_size, block_size, disk_size for mask_image func
    dilate: bool
        whether to perform morphological dilation.
        Useful to keep stuff close to edge

    Returns
    ---------
    markers, seed_markers: array
        only objects with seeds and seeds with objects
    """
    min_size, block_size, disk_size = mask_params
    mask = mask_image(im, min_size=min_size, block_size=block_size,
        selem=skimage.morphology.disk(disk_size))
    if dilate:
    # enlarge mask to keep particles close to edge. Doing this before watershed
    # prevents invasion into other cells and is faster, smart
        mask = skimage.morphology.binary_dilation(mask,
                                            selem=skimage.morphology.disk(10))
    markers = skimage.measure.label(mask)
    # watershed transform using nuclei as basins, also removes cells wo nucleus
    markers = skimage.morphology.watershed(markers,
            seed_markers, mask=mask)

    # ensure use of same labels for nuclei
    seed_mask = seed_markers>0
    seed_markers  =  seed_mask * markers

    return markers, seed_markers

def make_seg_im(markers, im):
    """
    Make a segmentation image with marker-based highlighted boundaries

    Arguments
    ---------
    markers: tuple of arrays
        labeled cell/nuclei markers
    im: array
        image to draw boundaries on

    Returns
    ---------
    seg_im: array
        image with segmentation boundaries
    """
    seg_im = im.copy()
    print('making segmentation image with highlighted boundaries...')
    for marker in markers:
        seg_im[skimage.segmentation.find_boundaries(marker)] = np.max(seg_im)
    return seg_im
