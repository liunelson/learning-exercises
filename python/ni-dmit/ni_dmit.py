# Define function to parse `*.scan` files from Simon in Hamburg.

import numpy as np
import matplotlib.pyplot as plt
import re

def parse_scan(full_path):

    with open(full_path, 'r') as f:
        fcontent = f.read()
    
    # Trim starting '[' and ending ']'
    fcontent = fcontent[1:-1]

    # Remove all " (around NaN)
    fcontent = re.sub('(\")', '', fcontent)

    # Define data dict
    data = {'wave': None, 'time': None, 'abs': None, 'spec_bkg': None}

    # Find and split at '[[' and ']]'
    it1 = re.finditer('(\[{2})', fcontent)
    it2 = re.finditer('(\]{2})', fcontent)
    for m1, m2, k in zip(it1, it2, data.keys()):
        i = (m1.span())[1]
        j = (m2.span())[1] - 2
        data[k] = fcontent[i:j]

    # Clear data
    fcontent = None
    del fcontent

    # Remove extra '[' and ']'
    data['abs'] = re.sub('(\[)|(\])', '', data['abs'])

    # Split and convert string to array
    for k in data.keys():
        i = data[k].split(',')
        data[k] = np.asarray(i)

    for k in data.keys():
        print(f'{k} -> {data[k].shape}')

    # Convert 'NaN' strings to values
    data['abs'][(data['abs'][:] == 'NaN')] = np.nan

    # Convert strings to floats
    for k in data.keys():
        #print(f'{k} -> {data[k].shape}')
        data[k] = np.asarray([float(i) for i in data[k]])

    # Reshape absorbance data
    data['abs'] = np.reshape(data['abs'], (data['time'].shape[0], data['wave'].shape[0]))
    
    # Transpose such that x-axis is time, y-axis is wavelength
    data['abs'] = np.transpose(data['abs'])

    # Change to picoseconds
    data['time'] = np.divide(data['time'], 1000)

    return data

def plot_scan(data_plot, no_nan = False, vmin_max = 15, filename = None):

    # Define panel arrangement
    ndata = len(data_plot)
    ncols = int(np.floor(np.sqrt(ndata)))
    nrows = int(np.ceil(np.sqrt(ndata)))
    if ncols*nrows < ndata:
        ncols = ncols + 1

    # Define margins and dimensions
    lr_margin = 0.1
    ud_margin = 0.1
    spacing = 0.1
    width = (1 - 2*lr_margin - ncols*spacing)/ncols
    height = (1 - 2*ud_margin - nrows*spacing)/nrows

    # Define figure and axes
    fig = plt.figure(figsize = [4*ncols, 3*nrows])
    ax = np.empty([nrows, ncols], dtype = 'O')
    img = np.empty([nrows*ncols, 1], dtype = 'O')

    # Iterate
    for j in range(0, ncols, 1):
        for i in range(0, nrows, 1):

            # (i, j) index to linear index, Fortran/col-major style 
            k = np.ravel_multi_index([i, j], ax.shape, order = 'F')

            if k < ndata:

                # Create axis
                ax[i, j] = plt.axes([lr_margin + i*(width + spacing), ud_margin + (ncols - j)*(height + spacing), width, height])

                # Data
                data = data_plot[k]

                # Wavelength range
                if no_nan:
                    w = ~np.isnan(data['abs'][:, 0])
                else: 
                    w = np.ones(data['wave'].shape, dtype = bool)
                w1 = (np.where(w))[0][0]
                w2 = (np.where(w))[0][-1]
                
                # Reset time
                data['time'] = data['time'] - data['time'][0]

                # Plot
                img[k] = ax[i, j].imshow(1000*data['abs'][w, :], aspect = 'auto', extent = [data['time'][0], data['time'][-1], data['wave'][w2], data['wave'][w1]], cmap = 'RdBu_r', vmin = -vmin_max, vmax = vmin_max, interpolation = None)

                # Ticks
                ax[i, j].tick_params(which = 'both', direction = 'in', top = True, right = True)
                ax[i, j].ticklabel_format(style = 'plain')
                # ax[i, j].set_xticks([513.09])
                if i != 0:
                    ax[i, j].tick_params(labelleft = False)
                if j != (ncols - 1):
                    ax[i, j].tick_params(labelbottom = False)

                # Titles
                ax[i, j].set_title(f'Scan {k}')

    # Colorbar
    # cb = fig.colorbar(img[0, 0], ax = ax, orientation = 'vertical', fraction = 0.1)
    # cb.ax.tick_params(direction = 'in')
    # cb.ax.set_title('ΔA (mOD)', fontsize = 10)

    i = ax[0, -1].set_xlabel('Time Delay (ps)')
    ax[0, -1].set_ylabel('Wavelength (nm)')

    # Export figure
    if isinstance(filename, str):
        fig.savefig(filename, dpi = 300)
