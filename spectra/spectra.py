import numpy as np
import sys

import astropy.units as u
from astropy.table import Table

import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

class spec(object):
    """Spectrum class with some useful methods.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, wavelength, flux, flux_noise, redshift, continuum=None, logwavelength=False, cosmo=None):
        """Initialize the spectrum object.

        Args:
            wavelength (:obj:`list` of :obj:`float`): Array of wavelength with astropy units
            flux (:obj:`list` of :obj:`float`): Array of flux (units unnecessary)
            flux_noise (:obj:`list` of :obj:`float`): Array of flux noise (units unnecessary)
            redshift (:obj:`float`): Redshift of emission for the quasar
            continuum (:obj:`list` of :obj:`float`, optional): Array of continuum values (same units as flux)
            logwavelength(:obj:`bool`, optional): If wavelength column is log10(wavelength) (default: False)
            cosmo(colossus cosmo object, optional): Cosmology object from colossus (default: 'planck18')
        """
        try:
            if logwavelength:
                wavelength = np.power(10, wavelength)
            self.data = np.c_[wavelength, flux, flux_noise]
        except:
            print('Cant combine wavelength and flux data, make sure they have the same size')
            sys.exit(-1)

        try:
            self.redshift = float(redshift)
        except:
            print('Redshift:', redshift, ' cant be cast to a float')
            sys.exit(-1)

        if continuum is not None:
            self.has_continuum = True
            if len(continuum) != len(wavelength):
                print('Continuum is not same shape as wavelength column')
                sys.exit(-1)
            self.continuum = continuum
        else:
            self.has_continuum = False

        self.lymanseries = {'Lyalpha': 1215.67 * u.AA,
                            'Lybeta': 1025.73 * u.AA,
                            'Lygamma': 972.54 * u.AA}

        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = cosmology.setCosmology('planck18')

    def comoving_extent(self, wavelength_lower, wavelength_upper, lyman='Lyalpha'):
        """Returns the comoving extent between wavelength_lower and wavelength_upper.

        Args:
            wavelength_lower (:obj:`float`): Lower wavelength (observed)
            wavelength_upper (:obj:`float`): Upper wavelength (observed)
            lyman (:obj:`string`, optional): Which lyman series line to use, either 'Lyalpha',
                'Lybeta', or 'Lygamma' (default: 'Lyalpha')
        """
        # first convert the wavelengths into the redshift at which they equal the correct lyman line
        z_lower = wavelength_lower / self.lymanseries[lyman] - 1
        z_upper = wavelength_upper / self.lymanseries[lyman] - 1

        # check and make sure we're good
        if z_lower < 0 or z_upper < 0:
            print('Ahh!!! It looks like the wavelength was never at the correct lyman line, z < 0!')
            sys.exit(-1)
        if z_lower > self.redshift or z_upper > self.redshift:
            print('Ahh!!! It looks like the wavelength was never at the correct lyman line, z > zquasar!')
            sys.exit(-1)
        if z_lower > z_upper:
            print('z_lower > z_upper: ', z_lower, ' > ', z_upper, ', uh oh!')
            sys.exit(-1)

        comoving_distance = self.cosmo.comovingDistance(z_lower, z_upper, transverse=False)
        comoving_distance /= self.cosmo.h

        return comoving_distance



    def plot(self, show=True):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[:,0], self.data[:,1], lw=0.2)
        ax.set_xlabel('wavelength [ angstrom ]')
        ax.set_ylabel('flux')
        if show:
            plt.show()
            return None
        else:
            return fig, ax


if __name__ == '__main__':
    z = 5.41
    dat = np.genfromtxt('../data/QUASAR_spec_FAN/z541.spec.tex')
    s = spec(dat[:,0], dat[:,1], z, logwavelength=True)
