#!usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas as pd

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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
})
plt.rcParams['axes.linewidth'] = 3


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


# Misc. data manipulation helper functions
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


def sigfig(num):
    if type(num) == u.quantity.Quantity:
        return np.float64('{:0.3g}'.format(num.value)) * num.unit
    return np.float64('{:0.3g}'.format(num))


# ------------------------------------------------------------------------------
# Gaia query (PMs, parallax) and analysis (distance from parallax) functions
def pms_parallax_from_gaia(star_coords):
    width = 1 * u.arcsec
    height = 1 * u.arcsec

    res = Gaia.query_object_async(coordinate=star_coords, width=width, height=height, verbose=False)

    found_pos = coord.SkyCoord(res['ra'], res['dec'])
    closest = np.argmax(-star_coords.separation(found_pos))

    print(f"Queried Gaia and found {len(res)} match(es).\nChoosing closest",
          f"match with separation of{BOLD}",
          f"{star_coords.separation(found_pos[closest]).to(u.arcsec):.4f}{END}")

    match = res[closest]

    return match['source_id'], match['parallax']*u.mas, match['parallax_error']*u.mas, match['ra']*u.deg, match['dec']*u.deg, match['pmra']*u.mas/u.yr, match['pmra_error']*u.mas/u.yr, match['pmdec']*u.mas/u.yr, match['pmdec_error']*u.mas/u.yr


def dist_from_parallax(parallax, parallax_error):
    dist = (1*u.pc*u.arcsec/parallax.to(u.arcsec)).to(u.kpc)
    dist_err = (1*u.pc*u.arcsec/parallax.to(u.arcsec)**2 * parallax_error.to(u.arcsec)).to(u.kpc)
    print(f"Estimated distance from parallax = {dist.value:.2e} +/- {dist_err:.2e}\n\n")
    return sigfig(dist), sigfig(dist_err)


# ------------------------------------------------------------------------------
# SDSS query (spectrum) and analysis (RV estimation) functions
def rv_from_sdss_spec(star_coords, name):
    search_radius = 1*u.arcsec
    xid = SDSS.query_region(star_coords, spectro=True, radius=search_radius)

    if xid is None:
        print("Could not find any spectral matches.")
        return None, None, None, None, None, None, None, None

    found_pos = coord.SkyCoord(xid['ra'] *u.deg, xid['dec']*u.deg)
    closest = np.argmax(-star_coords.separation(found_pos))
    
    print(f"Queried SDSS and found {len(xid)} match(es).\nChoosing closest",
          f"match with separation of{BOLD}",
          f"{star_coords.separation(found_pos[closest]).to(u.arcsec):.4f}{END}")
    try:
        sp = SDSS.get_spectra(matches=xid)
    except:
        return None, None, None, None, None, None, None, None

    star_dat = sp[closest][1].data
    star_wavelens = 10**star_dat['loglam']  # * u.Angstrom
    star_flux = star_dat['flux']  # * 1e-17 * u.erg / u.cm**2 / u.s / u.Angstrom

    star_continuum = rolling_quant(star_wavelens, star_flux, 100, lambda x: np.percentile(x, 90))
    star_normed = star_flux/star_continuum
    
    # Fetch template data and prepare continuum normed template spectrum
    spec_type = remove_digits(sp[closest][2].header['TFORM1'])
    # print(sp[0][2].header['TTYPE105'], sp[0][2].header['TFORM105'])
    # print(f"Spectral type {spec_type}")
    template = SDSS.get_spectral_template(f'star_{spec_type}')

    # borrowed from https://github.com/rpmunoz/PentaUC
    spec_hdr = template[0][0].header
    spec_dat = template[0][0].data
    wcs = WCS(spec_hdr)  

    index = np.arange(spec_hdr['NAXIS1'])
    tmpl_wavelens = 10**wcs.wcs_pix2world(index, np.zeros(len(index)), 0)[0]
    tmpl_flux = spec_dat[0]

    # tmpl_flux = tmpl_flux[tmpl_wavelens < 7200]
    # tmpl_wavelens = tmpl_wavelens[tmpl_wavelens < 7200]

    tmpl_continuum = rolling_quant(tmpl_wavelens, tmpl_flux, 100, lambda x: np.percentile(x, 90))
    tmpl_normed = tmpl_flux/tmpl_continuum

    # Cross correlation to find radial velocity
    shifts = np.linspace(-1200, 1200, int(1e5))*u.km/u.s
    XCs = [CCF(shift, star_wavelens, star_normed, tmpl_wavelens, tmpl_normed) for shift in shifts]    

    RV = shifts[np.argmax(XCs)]

    dlambda = np.median(np.diff(star_wavelens))*u.angstrom
    sig_RV = ((3e18*u.angstrom/u.s/6563/u.angstrom)**2*dlambda**2)**0.5

    print(f"Spectrum inferred radial velocity:",\
          f"RV = {RV.value:.3g} +/- {sig_RV.to(u.km/u.s):.3g}\n")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13,10))
    ax1.plot(shifts, XCs, c='k', linewidth=3)
    ax1.axvline(RV.value, c='k', linewidth=2, linestyle='dashed', label=fr"$v_l$$_o$$_s$={RV.value:.1f}$\pm${sig_RV.to(u.km/u.s):.2f}")
    ax1.set_xlabel('Shift [km/s]')
    ax1.set_ylabel('Cross correlation')
    ax1.legend(frameon=False, loc='upper right', prop={'size': 22})

    ax1.xaxis.set_major_locator(MultipleLocator(500))
    ax1.xaxis.set_minor_locator(MultipleLocator(100))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(2))

    ax1.tick_params(axis='both', which='major', direction='out', length=6, width=3, left=True, labelsize='medium', pad=10)
    ax1.tick_params(axis='both', which='minor', direction='out', length=3, width=3, left=True, labelsize='medium', pad=10)

    ax2.plot(star_wavelens, star_normed, c='k', label='Observed spectrum', linewidth=2.5)
    ax2.plot(tmpl_wavelens, tmpl_normed, c='red', label='A-type star template', linewidth=2.5, alpha=1)
    ax2.plot(doppler_shift(tmpl_wavelens, RV), tmpl_normed, c='blue', label='Shifted tempalte', linewidth=2.5, alpha=1, zorder=3)

    ax2.xaxis.set_major_locator(MultipleLocator(50))
    ax2.xaxis.set_minor_locator(MultipleLocator(25))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax2.yaxis.set_major_locator(MultipleLocator(0.25))

    ax2.tick_params(axis='both', which='major', direction='out', length=6, width=3, left=True, labelsize='medium', pad=10)
    ax2.tick_params(axis='both', which='minor', direction='out', length=3, width=3, left=True, labelsize='medium', pad=10)

    ax2.set_xlabel(r'Wavelength [$\AA$]')
    ax2.set_ylabel('Normalized flux')
    ax2.set_xlim([6500,6670])
    ax2.set_ylim([0.5,1.1])
    ax2.legend(frameon=False, loc='lower right', prop={'size': 22})
    plt.tight_layout()
    plt.savefig(f"rv_fits/{name}.pdf")
    plt.close()

    return sigfig(RV), sigfig(sig_RV), spec_type, found_pos[closest].ra.to(u.deg), found_pos[closest].dec.to(u.deg), xid[closest]['plate'], xid[closest]['fiberID'], xid[closest]['specobjid']


if __name__ == "__main__":
    matches = [
        #    name  ,     RA     ,    Dec
        ["Hivel15" , 250.9464124, 43.60734652],
        ["Hivel117", 207.0853222, 40.94939004],
        ["Hivel141", 234.1466978, 17.88953893],
        ["Hivel216", 197.2074332, 50.02354171],
        ["Hivel217", 201.9403245, 40.67826605],
        ["Hivel275", 206.8596394, 23.68693109],
        ["Hivel321", 198.1116608, 31.2129322 ],
        ["Hivel329", 166.2455078, 5.23116676 ],
        ["Hivel361", 219.5165922, 42.98527958],
        ["Hivel385", 221.4877243, 25.87594721],
        ["Hivel389", 223.9402804, 33.5384934 ],
        ["Hivel425", 176.681224 , 12.1402108 ],
        ["Hivel432", 191.6696044, 13.42992853],
        ["Hivel468", 1.6897984  , 27.09330209],
        ["Hivel500", 224.6887569, 4.45115226 ],
        ["Hivel563", 168.2815741, 48.02222718],
        ["Hivel567", 258.8817557, 27.2626295],
        ["Hivel588", 210.7395489, 37.80888577]
    ]

    HVSs = pd.DataFrame(columns=["name", "query_ra", "query_dec", "rv", "rv_unc", "spec_type", "sdss_ra", "sdss_dec", "plate", "fiberid", "specobjid", "gaia_sourceid", "parallax", "parallax_unc", "gaia_ra", "gaia_dec", "pmra", "pmra_unc", "pmdec", "pmdec_unc", "dist", "dist_unc"])

    for match in matches:
        name, ra, dec = match
        star_coords = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid = rv_from_sdss_spec(star_coords, name)
        gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error = pms_parallax_from_gaia(star_coords)
        dist, dist_err = dist_from_parallax(parallax, parallax_error)
        HVSs.loc[len(HVSs.index)] = [name, ra, dec, RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid, gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error, dist, dist_err]
    
    HVSs.to_csv("HVSs_lim.csv", index=False)

    # Find matches in LAMOST DR7 data
    # lamostdr7 = pd.read_csv("lamostdr7_gaiadr2_hvs_591.csv")
    # for name, ra, dec in zip(lamostdr7['ID'], lamostdr7['R.A.'], lamostdr7['decl.']):
    #     star_coords = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    #     RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid = rv_from_sdss_spec(star_coords)
    #     gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error = pms_parallax_from_gaia(star_coords)
    #     dist, dist_err = dist_from_parallax(parallax, parallax_error)
    #     HVSs.loc[len(HVSs.index)] = [name, ra, dec, RV, sig_RV, spec_type, sdss_ra, sdss_dec, plate, fiberid, specobjid, gaia_sourceid, parallax, parallax_error, gaia_ra, gaia_dec, pmra, pmra_error, pmdec, pmdec_error, dist, dist_err]
    
    # df[df['parallax'] > df['parallax_error']*5]
    # df[~df['RV'].isna()]