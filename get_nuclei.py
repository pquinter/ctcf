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

from im_utils import *

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
