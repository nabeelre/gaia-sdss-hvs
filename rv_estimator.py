import numpy as np
import matplotlib.pyplot as plt

import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

from astroquery.sdss import SDSS

import warnings
warnings.filterwarnings("ignore", category=AstropyWarning)

BOLD = "\033[1m"
END  = "\033[0m"


def extract_coords(SDSS_name):
    """
    Extract RA and Dec coordinates in degrees from an SDSS object name
    ex: "SDSS J013655.91+242546.0" yields ("01h36m55.91s", "+24d25m46.0s")
    """
    if "SDSS " in SDSS_name:
        SDSS_name = SDSS_name[6:]
    
    decsign = ''
    if '+' in SDSS_name:
        decsign = '+'
    elif '-' in SDSS_name:
        decsign = '-'
    else:
        print("couldn't find declination sign")
        
    RA = SDSS_name[0:SDSS_name.index(decsign)]
    RA = RA[0:2] + 'h' + RA[2:4] + 'm' + RA[4:] + 's'
    
    Dec = SDSS_name[SDSS_name.index(decsign):]
    Dec = Dec[0:3] + 'd' + Dec[3:5] + 'm' + Dec[5:] + 's'
    
    return RA, Dec


def remove_digits(string):
    """
    Remove any numbers from a given string
    """
    digits = '0123456789'

    table = str.maketrans('', '', digits)
    return string.translate(table)

# ---------------------------------------------
# Cross correlation helper functions
def rolling_quant(x, y, N_pix, stat=np.median):
    """
    Calculate a rolling quantity (e.g. median) over data y(x) with window size
    given by N_pix 
    """
    rm = np.zeros(len(x))
    for i in range(len(x)):
        if i < int(N_pix/2):
            rm[i] = stat(y[0:i+int(N_pix/2)])
        elif i <= len(x)-int(N_pix/2):
            rm[i] = stat(y[i-int(N_pix/2):i+int(N_pix/2)])
        else:  # i > len(x)-int(N_pix/2)
            rm[i] = stat(y[i-int(N_pix/2):-1])
    return rm


def doppler_shift(wavelens, radvel):
    """
    Perform a doppler shift on provided wavelengths by a provided radial vel
    radvel is specified as an astropy.Quantity with units of velocity
    """
    # radvel in km/s
    return wavelens*(1+(radvel/(3e5*u.km/u.s)).to(u.dimensionless_unscaled))


def CCF(shift, spec_wls, spec_dat, tmpl_wls, tmpl_dat):
    """
    calculate the cross correlation of a spectrum with a tempalte spectrum for a
    provided radial velocity shift
    """
    shifted_data = np.interp(spec_wls, doppler_shift(tmpl_wls, shift), tmpl_dat)
    return np.correlate(shifted_data, spec_dat)[0]


def rv_from_spec(name):
    print(f"{BOLD}Estimating radial velocity of {name} from SDSS spectra{END}")
    RA, Dec = extract_coords(name)
    pos = coord.SkyCoord(RA + ' ' + Dec, frame='icrs')

    search_radius = 1*u.arcmin
    xid = SDSS.query_region(pos, spectro=True, radius=search_radius)

    if xid is None:
        print("Could not find any spectral matches.")
        return

    found_pos = coord.SkyCoord(xid['ra'] *u.deg, xid['dec']*u.deg)
    closest = np.argmax(-pos.separation(found_pos))
    
    print(f"Found {len(xid)} match(es).\nChoosing closest match at",
          f"{BOLD}{found_pos[closest].ra:.4f}, {found_pos[closest].dec:.4f}{END}",
          f"from query on {BOLD}{pos.ra:.4f}, {pos.dec:.4f}{END} at separation",
          f"of {BOLD}{pos.separation(found_pos[closest]).to(u.arcsec):.4f}{END}")

    sp = SDSS.get_spectra(matches=xid)
    
    star_dat = sp[0][1].data
    star_wavelens = 10**star_dat['loglam']  # * u.Angstrom
    star_flux = star_dat['flux']  # * 1e-17 * u.erg / u.cm**2 / u.s / u.Angstrom

    star_continuum = rolling_quant(star_wavelens, star_flux, 100, lambda x: np.percentile(x, 90))
    star_normed = star_flux/star_continuum
    
    # Fetch template data and prepare continuum normed template spectrum
    spec_type = remove_digits(sp[0][2].header['TFORM1'])
    template = SDSS.get_spectral_template(f'star_{spec_type}')

    # borrowed from https://github.com/rpmunoz/PentaUC
    spec_hdr = template[0][0].header
    spec_dat = template[0][0].data
    wcs = WCS(spec_hdr)  

    index = np.arange(spec_hdr['NAXIS1'])
    tmpl_wavelens = 10**wcs.wcs_pix2world(index, np.zeros(len(index)), 0)[0]
    tmpl_flux = spec_dat[0]

    tmpl_continuum = rolling_quant(tmpl_wavelens, tmpl_flux, 100, lambda x: np.percentile(x, 90))
    tmpl_normed = tmpl_flux/tmpl_continuum

    # Cross correlation to find radial velocity
    shifts = np.linspace(-1000, 1000, int(1e4))*u.km/u.s
    XCs = [CCF(shift, star_wavelens, star_normed, tmpl_wavelens, tmpl_normed) for shift in shifts]    

    RV = shifts[np.argmax(XCs)]

    SN = np.median((spec_hdr['SN_G'], spec_hdr['SN_I'], spec_hdr['SN_R']))
    sig_RV = 500/SN*u.m/u.s

    print(f"RV = {RV.value:.2e} +/- {sig_RV.to(u.km/u.s):.2e}")

    # plt.figure(figsize=(15,7))
    # plt.plot(shifts, XCs)
    # plt.axvline(RV.value)
    # plt.title("Cross correlation")
    # plt.xlabel('doppler shift [km/s]', size=16)
    # plt.ylabel('XC', size=16)
    # plt.show()

    # plt.figure(figsize=(15,5))
    # plt.plot(tmpl_wavelens, tmpl_normed)
    # plt.plot(doppler_shift(star_wavelens, RV), star_normed)
    # plt.title("Cross correlation")
    # plt.xlabel('wavelength [angstrom]', size=16)
    # plt.ylabel('flux [erg/cm^2/s/Angstrom]', size=16)
    # plt.show()

    return RV, sig_RV