#!usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

from astroquery.sdss import SDSS
from astroquery.gaia import Gaia

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


def tbl_ext(table, key):
    """
    Extract value (with unit) from an astropy table or row
    """
    if table[key].unit is None:
        return table[key].data[0]
    return table[key].data[0] * table[key].unit


def pms_parallax_from_gaia(star_coords):
    width = 5 * u.arcsec
    height = 5 * u.arcsec

    res = Gaia.query_object_async(coordinate=star_coords, width=width, height=height)

    return tbl_ext(res, 'source_id'), tbl_ext(res, 'parallax'), tbl_ext(res, 'parallax_error'), tbl_ext(res, 'ra'), tbl_ext(res, 'dec'), tbl_ext(res, 'pmra'), tbl_ext(res, 'pmra_error'), tbl_ext(res, 'pmdec'), tbl_ext(res, 'pmdec_error')


def rv_from_sdss_spec(star_coords):
    search_radius = 25*u.arcsec
    xid = SDSS.query_region(star_coords, spectro=True, radius=search_radius)

    if xid is None:
        print("Could not find any spectral matches.")
        return None, None, None, None, None, None, None, None

    found_pos = coord.SkyCoord(xid['ra'] *u.deg, xid['dec']*u.deg)
    closest = np.argmax(-star_coords.separation(found_pos))
    
    print(f"Found {len(xid)} match(es).\nChoosing closest match at",
          f"{BOLD}{found_pos[closest].ra:.4f}, {found_pos[closest].dec:.4f}{END}",
          f"from query on {BOLD}{star_coords.ra:.4f}, {star_coords.dec:.4f}{END} at separation",
          f"of {BOLD}{star_coords.separation(found_pos[closest]).to(u.arcsec):.4f}{END}")

    sp = SDSS.get_spectra(matches=xid)
    
    star_dat = sp[closest][1].data
    star_wavelens = 10**star_dat['loglam']  # * u.Angstrom
    star_flux = star_dat['flux']  # * 1e-17 * u.erg / u.cm**2 / u.s / u.Angstrom

    star_continuum = rolling_quant(star_wavelens, star_flux, 100, lambda x: np.percentile(x, 90))
    star_normed = star_flux/star_continuum
    
    # Fetch template data and prepare continuum normed template spectrum
    spec_type = remove_digits(sp[closest][2].header['TFORM1'])
    # print(sp[0][2].header['TTYPE105'], sp[0][2].header['TFORM105'])
    print(f"Spectral type {spec_type}")
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
    shifts = np.linspace(-1200, 1200, int(1e5))*u.km/u.s
    XCs = [CCF(shift, star_wavelens, star_normed, tmpl_wavelens, tmpl_normed) for shift in shifts]    

    RV = shifts[np.argmax(XCs)]

    SN = np.median((spec_hdr['SN_G'], spec_hdr['SN_I'], spec_hdr['SN_R']))
    sig_RV = 500/SN*u.m/u.s

    print(f"RV = {RV.value:.2e} +/- {sig_RV.to(u.km/u.s):.2e}\n\n")

    plt.figure(figsize=(15,7))
    plt.plot(shifts, XCs)
    plt.axvline(RV.value)
    plt.title("Cross correlation")
    plt.xlabel('doppler shift [km/s]', size=16)
    plt.ylabel('XC', size=16)
    plt.savefig("CC.pdf")
    plt.close()
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(tmpl_wavelens, tmpl_normed)
    plt.plot(doppler_shift(star_wavelens, RV), star_normed)
    plt.title("with shift")
    plt.xlabel('wavelength [angstrom]', size=16)
    plt.ylabel('flux [erg/cm^2/s/Angstrom]', size=16)
    plt.savefig("with_RV_shift.pdf")
    plt.close()
    plt.show()

    plt.figure(figsize=(15,5))
    plt.plot(tmpl_wavelens, tmpl_normed)
    plt.plot(star_wavelens, star_normed)
    plt.title("without shift")
    plt.xlabel('wavelength [angstrom]', size=16)
    plt.ylabel('flux [erg/cm^2/s/Angstrom]', size=16)
    plt.savefig("without_RV_shift.pdf")
    plt.close()
    plt.show()

    return RV, sig_RV, spec_type, found_pos[closest].ra, found_pos[closest].dec, xid[closest]['plate'], xid[closest]['fiberID'], xid[closest]['specobjid']


if __name__ == "__main__":
    names = [
        "SDSS J013655.91+242546.0",
        "SDSS J090745.0+024507", 
        "SDSS J091301.00+305120.0",
        "SDSS J091759.42+672238.7",
        "SDSS J110557.45+093439.5",
        "SDSS J113312.12+010824.9",
        "SDSS J094214.04+200322.1",
        "SDSS J102137.08-005234.8",
        "SDSS J120337.85+180250.4"
    ]
    # for name in names:
    #     rv_from_spec(name)

    HVSs = pd.DataFrame(columns=["sdss_name", "query_ra", "query_dec", "RV", "RV_unc", "spec_type", "sdss_ra", "sdss_dec", "plate", "fiberid", "specobjid", "gaia_sourceid", "parallax", "parallax_error", "gaia_ra", "gaia_dec", "pmra", "pmra_error", "pmdec", "pmdec_error"])

    for name in names[0:3]:
        RA, Dec = extract_coords(name)
        star_coords = coord.SkyCoord(ra=RA, dec=Dec, frame='icrs')

        RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid = rv_from_sdss_spec(star_coords)

        gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error = pms_parallax_from_gaia(star_coords)

        # dist from gaia parallax

        HVSs.loc[len(HVSs.index)] = [name, RA, Dec, RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid, gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error]

    HVSs.to_csv("HVSs.csv")