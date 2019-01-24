import numpy as np
import sys

import astropy.units as u
from astropy.table import Table

class spec(object):
    """Spectrum class with some useful methods.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, wavelength, flux, redshift, continuum=None, logwavelength=False):
        """Initialize the spectrum object.

        Args:
            wavelength (:obj:`list` of :obj:`float`): Array of wavelength with astropy units
            flux (:obj:`list` of :obj:`float`): Array of flux (units unnecessary)
            redshift (:obj:`float`): Redshift of emission for the quasar
            continuum (:obj:`list` of :obj:`float`, optional): Array of continuum values (same units as flux)
            logwavelength(:obj:`bool`, optional): If wavelength column is log10(wavelength) (default: False)
        """
        try:
            if logwavelength:
                wavelength = np.power(10, wavelength)
            self.data = np.c_[wavelength, flux]
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

if __name__ == '__main__':
    z = 5.41
    dat = np.genfromtxt('../data/QUASAR_spec_FAN/z541.spec.tex')
    s = spec(dat[:,0], dat[:,1], z, logwavelength=True)
