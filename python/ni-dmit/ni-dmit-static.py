#%%
import numpy as np
import matplotlib.pyplot as plt

#%% [markdown]
# Reference spectrum of white light.
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-20-NiDmit_Nice_Data\\Sample1_500Hz\\'
wave_WL, spec_WL = np.loadtxt((path + 'WL_ref.txt'), delimiter = '\t', unpack = True)

# Sample transmission.
wave_trans, spec_trans = np.loadtxt((path + 'sample_transmission.txt'), delimiter = '\t', unpack = True)

#%%
fig, ax = plt.subplots(2, 1)
ax[0].plot(wave_WL, spec_WL, color = 'tab:blue')
ax[0].tick_params(axis = 'y', labelcolor = 'tab:blue')
ax[0].set_ylabel('Whitelight (counts)')
ax[0].set_ylim(0, 35e3)
ax[0].set_xlim(wave_trans[0], wave_trans[-1])
ax[0].set_xticklabels([])
ax[0].grid(True)
ax_ = ax[0].twinx()
ax_.plot(wave_trans, spec_trans, color = 'tab:red')
ax_.tick_params(axis = 'y', labelcolor = 'tab:red')
ax_.set_ylabel('Sample Transmission (counts)')
ax_.set_ylim(ax[0].get_ylim())

# Sample absorbance.

abs_ref = -np.log10(np.divide(spec_trans, spec_WL))
abs_ref[spec_WL < 1000] = np.nan

ax[1].plot(wave_trans, abs_ref)
ax[1].set_xlabel('Wavelength (nm)')
ax[1].set_ylabel('Sample Absorbance (OD)')
ax[1].grid(True)
ax[1].set_ylim(-0.25, 0.75)
ax[1].set_xlim(wave_trans[0], wave_trans[-1])

# Weird negative $\Delta A$ between 550 nm and 700 nm. Scattering?

#%% [markdown]
# Read background data from `\ta-data-from-simon\2019-06-24_NiDmit_Static`
#
# Sample #2
wave_bkg, abs_bkg = np.loadtxt('D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-24_NiDmit_Static\\bkg_2048x1p7ms.txt', delimiter = '\t', unpack = True)

#%%
fig, ax = plt.subplots(1, 1)
ax.plot(wave_bkg, abs_bkg)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Measurement')
ax.set_title('Background')

#%% [markdown]
# Read absorption spectra from `\ta-data-from-simon\2019-06-24_NiDmit_Static\reference spectra`.
# 
# Unpolarized data.
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-24_NiDmit_Static\\reference spectra\\sample2_HT'
filename = '\\reference_unpolarized.txt'
wave_pol, abs_data_HT_unpol = np.loadtxt((path + filename), delimiter = '\t', unpack = True)

path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-24_NiDmit_Static\\reference spectra\\sample2_LT'
filename = '\\reference_unpolarized.txt'
wave_pol, abs_data_LT_unpol = np.loadtxt((path + filename), delimiter = '\t', unpack = True)

#%%
fig, ax = plt.subplots()
ax.plot(wave_pol, abs_data_HT_unpol, label = 'HT', color = 'tab:red')
ax.plot(wave_pol, abs_data_LT_unpol, label = 'LT', color = 'tab:blue')
legend = ax.legend(loc='upper right')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Absorption')

#%% [markdown]
# Polarisation dependence of sample #2.
path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-24_NiDmit_Static\\reference spectra\\sample2_HT'
abs_data_HT = np.zeros((2048, 19))
abs_pol = np.arange(0, 190, 10)
for i in range(19):
    filename = ('reference_%ddeg.txt' % (10*i))
    wave_pol, abs_data_HT[:, i] = np.loadtxt((path + '\\' + filename), delimiter = '\t', unpack = True)

#%%
fig, ax = plt.subplots()
ax.pcolor(abs_pol, wave_pol, abs_data_HT)
ax.set_xlabel('Polarization (deg)')
ax.set_ylabel('Wavelength (nm)')
ax.set_title('Polarization Dependence - Ni(dmit) Sample #2 HT')

#%%
U, S, Vh = np.linalg.svd(abs_data_HT, full_matrices = False)

#%%
fig, ax = plt.subplots()
ax.plot(S[1:10], marker = 'o')
ax.set_yscale('log')
ax.set_title('Singular Values')

fig, ax = plt.subplots()
ax.plot(wave_pol, U[:, 0])
ax.set_xlabel('Wavelength (nm)')
ax.set_title('Left Singular Vectors')
ax_ = ax.twinx()
ax_.plot(wave_pol, U[:, 1], color = 'tab:red')
ax_.tick_params(axis = 'y', labelcolor = 'tab:red')

fig, ax = plt.subplots()
ax.plot(abs_pol, Vh[0, :])
ax.set_xlabel('Polarization (deg)')
ax.set_title('Right Singular Vectors')
ax_ = ax.twinx()
ax_.plot(abs_pol, Vh[1, :], color = 'tab:red')
ax_.tick_params(axis = 'y', labelcolor = 'tab:red')

#%% [markdown]
# Repeat for Sample #2 at LT

path = 'D:\\Projects (Miller)\\Ni-dmit\\ta-data-from-simon\\2019-06-24_NiDmit_Static\\reference spectra\\sample2_LT'
abs_data_LT = np.zeros((2048, 19))
abs_pol = np.arange(0, 190, 10)
for i in range(19):
    filename = ('reference_%ddeg.txt' % (10*i))
    wave_pol, abs_data_LT[:, i] = np.loadtxt((path + '\\' + filename), delimiter = '\t', unpack = True)

#%%
fig, ax = plt.subplots()
ax.pcolor(abs_pol, wave_pol, abs_data_LT)
ax.set_xlabel('Polarization (deg)')
ax.set_ylabel('Wavelength (nm)')
ax.set_title('Polarization Dependence - Ni(dmit) Sample #2 LT')

#%%
U, S, Vh = np.linalg.svd(abs_data_LT, full_matrices = False)
S = np.divide(S, S[0])

#%%
fig, ax = plt.subplots()
ax.plot(np.arange(1, S.shape[0] + 1, 1), S, color = 'tab:gray')
ax.plot(1, S[0], marker = 'o', color = 'tab:blue')
ax.plot(2, S[1], marker = 'o', color = 'tab:red')
ax.plot(3, S[2], marker = '+', color = 'tab:red')
ax.set_yscale('log')
ax.set_title('Singular Values')
ax.set_xticks(np.arange(1, S.shape[0] + 1, 2))

fig, ax = plt.subplots(2, 1)

ax[0].plot(wave_pol, U[:, 0], color = 'tab:blue')
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_title('Left Singular Vectors')
ax[0].tick_params(axis = 'y', labelcolor = 'tab:blue')
ax[0].set_xlim(wave_pol[0], wave_pol[-1])
ax_ = ax[0].twinx()
ax_.plot(wave_pol, U[:, 1], color = 'tab:red')
ax_.plot(wave_pol, U[:, 2], linestyle = ':', color = 'tab:red')
ax_.tick_params(axis = 'y', labelcolor = 'tab:red')

ax[1].plot(abs_pol, Vh[0, :], color = 'tab:blue')
ax[1].set_xlabel('Polarization (deg)')
ax[1].set_title('Right Singular Vectors')
ax[1].tick_params(axis = 'y', labelcolor = 'tab:blue')
ax[1].set_xlim(abs_pol[0], abs_pol[-1])
ax[1].set_xticks(np.arange(0, 181, 45))
ax_ = ax[1].twinx()
ax_.plot(abs_pol, Vh[1, :], color = 'tab:red')
ax_.plot(abs_pol, Vh[2, :], linestyle = ':', color = 'tab:red')
ax_.tick_params(axis = 'y', labelcolor = 'tab:red')

fig.tight_layout()

#%%
