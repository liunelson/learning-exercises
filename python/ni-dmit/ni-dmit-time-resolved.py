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

#%% [markdown]
# Since no substrate TA dataset is available, 
# GVD correction is not readily feasible for most of the datasets. 
# 
# Let's consider just the long-time datasets, i.e. `data_4`, `data_5`.

def svd_scan(data):

    # Break out variables
    wave = data['wave']
    time = data['time'] - data['time'][0]
    data = data['abs']

    # Trim NaN out
    no_nan = True
    i = data.sum(axis = 1)
    if no_nan:
        w = ~np.isnan(i)
    else: 
        w = np.ones(wave.shape, dtype = bool)
    w1 = (np.where(w))[0][0]
    w2 = (np.where(w))[0][-1]
    wave = wave[w1:w2]
    data = data[w1:w2, :]

    # Remove pump scatter
    pump = (data[:, 0:5]).mean(axis = 1)
    data_pump = np.add(data, -pump[:, np.newaxis])
    pump_ = data_pump.mean(axis = 1)

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = [8, 4])
    fig.suptitle('Pump Scatter Spectrum')
    ax.plot(wave, 1000*pump, color = 'tab:blue')
    ax.tick_params(direction = 'in', top = True, right = True)
    ax.tick_params(axis = 'y', color = 'tab:blue', labelcolor = 'tab:blue')
    ax.set_xlim([wave[0], wave[-1]])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('ΔA (mOD)')
    ax_ = ax.twinx()
    ax_.plot(wave, 1000*pump_, color = 'tab:red')
    ax_.tick_params(axis = 'y', direction = 'in', color = 'tab:red', labelcolor = 'tab:red')
    # w = np.argmin(pump)
    # ax[1].plot(time, data[w, :])
    # ax[1].tick_params(direction = 'in', top = True, right = True)
    # ax[1].set_xlabel('Time Delay (ps)')
    # ax[1].tick_params(labelleft = False)
    # ax[1].set_ylim(ax[0].get_ylim())


    # Trim pump scatter region
    w = np.argmin(np.abs(wave - 410))
    wave = wave[w:]
    data_pump = data_pump[w:, :]

    # Trim CPM region
    t = np.argmin(np.abs(time - 8))
    time = time[t:]
    data_pump = data_pump[:, t:]

    # SVD
    U, S, Vh = np.linalg.svd(data_pump, full_matrices = False)
    i = [0, 1, 2]
    U[:, i] = -U[:, i]
    Vh[i, :] = -Vh[i, :]

    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = [8, 4])
    ax[0].set_ylim([wave[-1], wave[0]])
    ax[0].set_xlabel('ΔA (a.u.)')
    ax[0].set_ylabel('Wavelength (nm)')
    ax[0].set_title('U')
    ax[1].plot(np.arange(1, S.shape[0] + 1), S, color = 'tab:blue')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_title('S')
    ax[2].set_xlabel('Time Delay (ps)')
    ax[2].set_ylabel('ΔA (a.u.)')
    ax[2].set_title('V')
    for i in range(2):
        ax[i].tick_params(direction = 'in', top = True, right = True)
        ax[i].grid(True)

    for i in range(3):
        ax[0].plot(U[:, i], wave, color = 'C' + str(i))
        ax[1].plot(i + 1, S[i], marker = 'o', color = 'C' + str(i))
        ax[2].plot(time, (Vh[i, :]).T)
        

    fig.tight_layout()


    # Low-rank approximation
    j = [0, 1]
    data_svd = np.dot(U[:, j], np.dot(np.diag(S[j]), Vh[j, :]))

    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = [8, 8])
    ext = [time[0], time[-1], wave[-1], wave[0]]
    img = ax[0].imshow(1000*data_pump, extent = ext, vmin = -15, vmax = 15, cmap = 'RdBu_r', aspect = 'auto', interpolation = None)
    img = ax[1].imshow(1000*data_svd, extent = ext, vmin = -15, vmax = 15, cmap = 'RdBu_r', aspect = 'auto', interpolation = None)
    img = ax[2].imshow(10*1000*(data_pump - data_svd), extent = ext, vmin = -15, vmax = 15, cmap = 'RdBu_r', aspect = 'auto', interpolation = None)
    for i in range(3):
        ax[i].tick_params(direction = 'in', top = True, right = True)

        if i != 2:
            ax[i].tick_params(labelbottom = False)

    ax[2].set_xlabel('Time Delay (ps)')
    ax[1].set_ylabel('Wavelength (nm)')
    ax[0].set_title('Trimmed Data')
    ax[1].set_title('Low-Rank Data (' + str(j[0]) + '-' + str(j[-1]) + ')')
    ax[2].set_title('Residuals (x 10)')
    cb = fig.colorbar(img, ax = ax, orientation = 'vertical', fraction = 0.1)
    cb.ax.tick_params(direction = 'in')
    cb.ax.set_title('ΔA (mOD)', fontsize = 10)



svd_scan(data_4[0])

#%%
