import numpy as np
import sys

import astropy.units as u
from astropy.table import Table

import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from astropy import constants as const
import more_itertools as mit

def read_spectrum(fname):
    """Function which reads in file name and returns spectra

    Returns z, wavelength, signal, noise

    Args:
        fname (:obj:`str`): file name where spectrum is stored
    """
    t = np.load(fname)
    data = t.item().get('data')
    redshift = t.item().get('redshift')
    return redshift, data[:,0], data[:,1], data[:,2]

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

        self.lymanseries = {'Lyalpha': 1215.67,
                            'Lybeta': 1025.73,
                            'Lygamma': 972.54}

        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = cosmology.setCosmology('planck18')

        self._speed_of_light_kms_ = const.c.to_value(u.km/u.s)

        self._compute_smoothed_spectra_()
        self._compute_dark_gaps_()

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

    def velocity_offset(self, wavelength_lower, wavelength_upper):
        """Compute the velocity offset between wavelength_lower and wavelength_upper.

        Args:
            wavelength_lower (:obj:`list` of :obj:`float`): lower wavelengths
            wavelength_upper (:obj:`list` of :obj:`float`): upper wavelengths
        """
        avg_wavelength = np.divide(np.add(wavelength_lower, wavelength_upper), 2.0)
        delta_wavelength = np.subtract(wavelength_upper, wavelength_lower)
        delta = np.divide(delta_wavelength, avg_wavelength)
        return np.multiply(delta, self._speed_of_light_kms_)

    def _compute_smoothed_spectra_(self, width=100):
        """Computes the smoothed spectra in bins of width km/s

        Args:
            width (:obj:`float`, optional): bin width [km/s] (default: 100)
        """
        wavelength = self.data[:,0]
        flux = self.data[:,1]
        flux_noise = self.data[:,2]

        wvtensor = np.array([ wavelength for _ in range(len(wavelength)) ])
        block = np.subtract(np.transpose(wvtensor), wavelength) # block[i][j] = wavelength[i] - wavelength[j]
        block = np.divide(block, wavelength) # / wavelength[i]
        lim = width / const.c.to_value(u.km/u.s)
        limbool = np.less(np.abs(block), lim)
        keys = [np.where(l)[0] for l in limbool]

        smoothed_flux = np.array([ np.mean(flux[k]) for k in keys ])
        smoothed_noise = np.array([ np.sum(np.square(flux_noise[k]))/len(k) for k in keys ])

        self.smoothed_data = np.c_[wavelength, smoothed_flux, smoothed_noise]

        return None

    def _compute_dark_gaps_(self, n=3, vel_extent=100):
        """Computes all dark gaps in the spectra.

        Assumes that dark gaps are all pixels in the spectra with smoothed_flux < n * smoothed_sigma
        Min_extent is the minimum velocity offset for each gap.

        Args:
            n (:obj:`float`, optional): dark gap criteria (default: 3)
            vel_extent (:obj:`float`, optional): min velocity offset to be considered dark gap [km/s] (default: 1)
        """
        wavelength = self.data[:,0]
        flux = self.data[:,1]
        noise = self.data[:,2]

        smoothed_wavelength = self.smoothed_data[:,0]
        smoothed_flux = self.smoothed_data[:,1]
        smoothed_noise = self.smoothed_data[:,2]

        dark_keys = np.where(np.less(smoothed_flux, n * smoothed_noise))[0]

        dark_gaps = [list(group) for group in mit.consecutive_groups(dark_keys)]
        wavelength_lower = np.array( [smoothed_wavelength[gap[0]] for gap in dark_gaps] )
        wavelength_upper = np.array( [smoothed_wavelength[gap[-1]] for gap in dark_gaps] )
        velocity_offset = self.velocity_offset(wavelength_lower, wavelength_upper)
        gap_bool = np.greater(velocity_offset, vel_extent)

        self.dark_gaps = [gap for gap,b in zip(dark_gaps, gap_bool) if b]
        self.dark_gaps_wavelength = [wavelength[gap] for gap in self.dark_gaps]
        self.dark_gaps_flux = [flux[gap] for gap in self.dark_gaps]
        self.dark_gaps_noise = [noise[gap] for gap in self.dark_gaps]

        wavelength_lower = np.array( [wavelength[gap[0]] for gap in self.dark_gaps] )
        wavelength_upper = np.array( [wavelength[gap[-1]] for gap in self.dark_gaps] )
        self.dark_gaps_velocity_offset = self.velocity_offset(wavelength_lower, wavelength_upper)

        return None

    def get_dark_gaps_extent(self, vel_extent, greater=True, alpha_only=False, beta_only=False):
        """Returns all dark gaps with extent greater than (or less than) vel_extent.

        Note that if alpha_only and beta_only are True, then an error will be raised.

        Args:
            vel_extent (:obj:`float`): velocity offset of dark gap
            greater (:obj:`bool`, optional): All dark gaps greater than extent, False for less than (default: True)
            alpha_only (:obj:`bool`, optional): All dark gaps must lie between lyman alpha and lyman beta
            beta_only
        """

        wavelength = self.data[:,0]
        wavelength_rest = np.divide(wavelength, 1.+self.redshift)

        if alpha_only and beta_only:
            raise Exception('alpha_only and beta_only cannot both be set to True')

        if alpha_only:
            w_upper = self.lymanseries['Lyalpha']
            w_lower = self.lymanseries['Lybeta']
        elif beta_only:
            w_upper = self.lymanseries['Lybeta']
            w_lower = self.lymanseries['Lygamma']
        else:
            w_upper = np.inf
            w_lower = 0

        wbig = np.greater(wavelength_rest, w_lower)
        wless = np.less(wavelength_rest, w_upper)
        keys = np.where(np.logical_and(wbig, wless))[0]
        k_min = keys[0]
        k_max = keys[-1]

        these_gaps = []
        for gap, wave, vel in zip(self.dark_gaps, self.dark_gaps_wavelength, self.dark_gaps_velocity_offset):
            # check to see if we in the correct window
            kbool = gap[0] > k_min and gap[-1] < k_max

            # check to see if we match the necessary size
            if greater:
                vbool = vel >= vel_extent
            else:
                vbool = vel < vel_extent

            # cool
            if kbool and vbool:
                these_gaps.append(gap)

        return these_gaps

    def get_dark_gaps_in_alpha_and_beta(self, alpha_vel_extent, beta_vel_extent):
        """Returns all dark gaps that are dark in both lyman alpha and lyman beta.

        Args:
            alpha_vel_extent (:obj:`float`): minimum velocity extent of lyman alpha dark gaps [km/s]
            beta_vel_extent (:obj:`float`): minimum velocity extent of lyman beta dark gaps [km/s]
        """

        wavelength = self.data[:,0]
        wavelength_emitted = np.divide(wavelength, 1+self.redshift)

        lyalpha = self.lymanseries['Lyalpha']
        lybeta = self.lymanseries['Lybeta']

        # first gather all dark gaps
        lymanalpha_gaps = self.get_dark_gaps_extent(alpha_vel_extent, alpha_only=True)
        lymanbeta_gaps = self.get_dark_gaps_extent(beta_vel_extent, beta_only=True)

        # (lambdae/lambdao) * ( 1 + ze) - 1
        # determine the redshift at which each wavelength in each gap was at emission wavelength
        lymanalpha_gaps_wavelength = [ wavelength_emitted[gap] for gap in lymanalpha_gaps ]
        lymanbeta_gaps_wavelength = [ wavelength_emitted[gap] for gap in lymanbeta_gaps ]

        lymanalpha_gaps_z = [ np.divide(w, lyalpha) * (1+self.redshift) - 1 
                              for w in lymanalpha_gaps_wavelength]
        lymanbeta_gaps_z = [ np.divide(w, lybeta) * (1+self.redshift) - 1 
                              for w in lymanbeta_gaps_wavelength]

        lymanalpha_gaps_zextent = [[z[0], z[-1]] for z in lymanalpha_gaps_z]
        lymanbeta_gaps_zextent = [[z[0], z[-1]] for z in lymanbeta_gaps_z]

        # now, require that the entire lyman alpha gap is in lyman beta
        these_gaps_alpha = []
        these_gaps_beta = []
        for zextent, gap_alpha in zip(lymanalpha_gaps_zextent, lymanalpha_gaps):
            for zextent_beta, gap_beta in zip(lymanbeta_gaps_zextent, lymanbeta_gaps):
                if zextent[0] > zextent_beta[0] and zextent[1] < zextent_beta[1]:
                    these_gaps_alpha.append(gap_alpha)
                    these_gaps_beta.append(gap_beta)
                    continue

        return these_gaps_alpha, these_gaps_beta

    def convert_gap_to_redshift(self, gap, line):
        """Returns the redshift of a gap at which it was in resonance with line.

        Args:
            gap (:obj:`list` of :obj:`int`): list of integers corresponding to data keys of gap
            line (:obj:`str`): line which to compute ('Lyalpha', 'Lybeta', or 'Lygammma')
        """
        try:
            line_wave = self.lymanseries[line]
        except:
            raise Exception('cant recognize line:', line)

        wavelength = self.data[:,0] / (1+self.redshift) # want emitted redshift
        wave = wavelength[gap]

        z = (wave / line_wave) * (1 + self.redshift) - 1
        return z

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
    invar = np.genfromtxt('../data/QUASAR_spec_FAN/z541.var.tex')[:,1]
    noise = 1/np.sqrt(invar)
    s = spec(dat[:,0], dat[:,1], noise, z, logwavelength=True)
    Lyalpha_gaps, Lybeta_gaps = s.get_dark_gaps_in_alpha_and_beta(100, 500)
