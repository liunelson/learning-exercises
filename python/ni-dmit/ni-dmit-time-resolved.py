#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
import importlib
# from ni_dmit import ni_dmit.parse_scan, ni_dmit.plot_scan
import ni_dmit

# importlib.reload(ni_dmit)

#%% [markdown]
# Let's take a look at the time-resolved data (21-06-2019)
#
# Import data (sample 2-1, 500 Hz, 100 fs, 4 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample2_500Hz\\4mJcm-2_100fsSteps'
data_1 = []
for i in range(len(glob.glob(path + '\\*.scan'))):
    filename = 'CompiledScan_' + str(i + 1) + '.scan'
    data_1.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 2-2, 500 Hz, 10 fs, 3 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample2Spot2_500Hz\\3mJcm-2_10fsSteps'
data_2 = []
for i in range(len(glob.glob(path + '\\*.scan'))):
    filename = 'CompiledScan_' + str(i + 1) + '.scan'
    data_2.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 3-1, 500 Hz, 10 fs, 3 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample3_500Hz\\3mJcm-2_10fsSteps'
data_3 = []
for i in range(len(glob.glob(path + '\\*.scan'))):
    filename = 'CompiledScan_' + str(i + 1) + '.scan'
    data_3.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 3-2, 500 Hz, 300 fs, 3 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample3Spot2_500Hz\\3mjcm-2_300fsSteps'
data_4 = []
for i in range(len(glob.glob(path + '\\*.scan'))):
    filename = 'CompiledScan_' + str(i + 1) + '.scan'
    data_4.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 4, 500 Hz, 2.5 ps, 3 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample4_500Hz\\3mjcm-2_2500fsSteps'
data_5 = []
for i in range(len(glob.glob(path + '\\*.scan'))):
    filename = 'CompiledScan_' + str(i + 1) + '.scan'
    data_5.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 5, 500 Hz, 10 fs, 2 to 4 mJ/cm$^2$).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample5_500Hz\\2mjcm-2_10fsSteps_PowerDep'
data_6 = []
i = 0
filename = 'CompiledScan_' + str(i + 1) + '.scan'
data_6.append(ni_dmit.parse_scan(path + '\\' + filename))

path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample5_500Hz\\4mjcm-2_10fsSteps_PowerDep'
data_6.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Import data (sample 5, 500 Hz, 10 fs, 3 mJ/cm$^2$, 0 to 45 to 90 deg polarization).
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample5_500Hz\\3mjcm-2_10fsSteps_0Polarization'
data_7 = []
i = 0
filename = 'CompiledScan_' + str(i + 1) + '.scan'
data_7.append(ni_dmit.parse_scan(path + '\\' + filename))

path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample5_500Hz\\3mjcm-2_10fsSteps_45Polarization'
data_7.append(ni_dmit.parse_scan(path + '\\' + filename))

path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-21-NiDmit_Nice_Data\\Sample5_500Hz\\3mjcm-2_10fsSteps_90Polarization'
data_7.append(ni_dmit.parse_scan(path + '\\' + filename))

#%% [markdown]
# Plot `data_1`.
ni_dmit.plot_scan(data_1, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-1.png')

#%% [markdown]
# Plot `data_2`.
ni_dmit.plot_scan(data_2, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-2.png')

#%% [markdown]
# Plot `data_3`.
ni_dmit.plot_scan(data_3, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-3.png')

#%% [markdown]
# Plot `data_4`.
ni_dmit.plot_scan(data_4, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-4.png')

#%% [markdown]
# Plot `data_5`.
ni_dmit.plot_scan(data_5, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-5.png')

#%% [markdown]
# Plot `data_6`.
ni_dmit.plot_scan(data_6, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-6.png')

#%% [markdown]
# Plot `data_7`.
ni_dmit.plot_scan(data_7, no_nan = True, vmin_max = 15, filename = 'ni-dmit-time-resolved-7.png')

#%% [markdown]
# Plot `data_2[0]` slices.

data = data_2[0]

def plot_traces(data, w_range): 

    data['time'] = data['time'] - data['time'][0]

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = [10, 5])

    i = np.argmin(abs(data['wave'] - 400))
    ax[0].imshow(1000*data['abs'][i:, :], aspect = 'auto', extent = [data['time'][0], data['time'][-1], data['wave'][-1], data['wave'][i]], cmap = 'RdBu_r', vmin = -15, vmax = 15, interpolation = None)
    ax[0].set_ylabel('Wavelength (nm)')

    for i in w_range:
        j = np.argmin(abs(data['wave'] - i))

        ax[0].plot(data['time'], i*np.ones(data['time'].shape))
        ax[1].plot(data['time'], 1000*data['abs'][j, :], label = str(i) + ' nm')

    ax[0].set_ylabel('Wavelength (nm)')
    ax[0].set_xlabel('Time Delay (ps)')
    ax[0].tick_params(direction = 'in', top = True, left = True, right = True)

    ax[1].tick_params(direction = 'in', top = True, left = True, right = True, labelleft = False, labelright = True)
    ax[1].set_xlim([data['time'][0], data['time'][-1]])
    ax[1].set_xlabel('Time Delay (ps)')
    ax[1].legend()
    ax[1].yaxis.set_label_position('right')
    ax[1].set_ylabel('dA (mOD)')

    fig.savefig('ni-dmit-time-resolved-data-traces.png', dpi = 300)

# plot_traces(data_2[0], np.arange(450, 750, 50))
# plot_traces(data_4[0], np.arange(450, 750, 50))
plot_traces(data_5[0], np.arange(450, 750, 50))

#%% [markdown]
# Try continuous wavelet transform on this dataset.

import pywt

def try_cwt(data, w):

    # Probe wavelength
    j = np.argmin(abs(data['wave'] - w)) 

    # Scaling
    s = np.arange(1, 128)/5

    # Time step
    dt = data['time'][1] - data['time'][0] 

    # CWT
    [coef, freq] = pywt.cwt(data['abs'][j, :], s, 'cmor1-1', dt)

    power = np.abs(coef)**2
    period = 1./freq

    # Plot
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = [10, 5])
    ax[0].plot(data['time'], 1000*data['abs'][j, :])
    img = ax[1].imshow(power, extent = [data['time'][0], data['time'][-1], period[0], period[-1]], aspect = 'auto', cmap = 'viridis', interpolation = None)
    
    cb = fig.colorbar(img, ax = ax[1], orientation = 'vertical', fraction = 0.1)

    ax[0].set_xlim([data['time'][0], data['time'][-1]])
    ax[0].set_xlabel('Time Delay (ps)')
    ax[0].tick_params(direction = 'in', top = True, left = True, right = True)

    ax[1].tick_params(direction = 'in', top = True, left = True, right = True, labelleft = True)
    ax[1].set_xlabel('Time Delay (ps)')
    # ax[1].yaxis.set_label_position('right')
    ax[1].set_ylabel('CWT Scale')

    fig.suptitle('w = ' + str(w) + ' nm')

try_cwt(data_2[0], 450)


#%%





#%%
