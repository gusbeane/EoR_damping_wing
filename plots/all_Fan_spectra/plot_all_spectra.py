import numpy as np

from spectra.spectra import spec
from spectra.spectra import read_spectrum

import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib as mpl
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

zlist = ['521', '531', '541', '582', '582_hres', '637']

def plot_spectra(lambda_min=7500, lambda_max=9300, observed=True):

    fig, ax = plt.subplots(len(zlist), 1, figsize=(6, len(zlist)*1.5), sharex=True)

    for z, x in zip(zlist,ax):
        fname = 'data/QUASAR_spec_FAN/z' + z + '.npy'
        z, wv, flx, flx_n = read_spectrum(fname)

        if not observed:
            wv /= 1. + z

        keys = np.where(np.logical_and(wv > lambda_min, wv < lambda_max))[0]

        x.plot(wv[keys], flx[keys])
        x.set_ylabel(r'$\text{flux}$')

    ax[-1].set_xlim(lambda_min, lambda_max)

    ax[-1].set_xlabel(r'$\lambda_{\text{obs}}\,[\,A\,]$')

    fig.tight_layout()

    if observed:
        fig.savefig('all_fan_spectra_observed.pdf')
    else:
        fig.savefig('all_fan_spectra_rest.pdf')


if __name__ == '__main__':
    plot_spectra()
    plot_spectra(1000, 1350, False)
