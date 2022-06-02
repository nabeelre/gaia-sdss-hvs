#!usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import pandas as pd

import astropy.coordinates as coord
import astropy.units as u

import gala.dynamics as gd
import gala.potential as gp

BOLD = "\033[1m"
END  = "\033[0m"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
})
plt.rcParams['axes.linewidth'] = 3


def str_to_astropy(vals):
    return [u.Quantity(val) for val in vals]


def colorline(
    ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha, rasterized=True)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_orbit(pos, axs, cmap, alpha):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    
    ax1, ax2, ax3 = axs
    
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    xx, yy = verts[:, 0], verts[:, 1]
    _ = colorline(ax1, xx, yy, cmap=plt.get_cmap(cmap), linewidth=4, alpha=alpha)
    
    path = mpath.Path(np.column_stack([x, z]))
    verts = path.interpolated(steps=3).vertices
    xx, zz = verts[:, 0], verts[:, 1]
    _ = colorline(ax2, xx, zz, cmap=plt.get_cmap(cmap), linewidth=4, alpha=alpha)
    
    path = mpath.Path(np.column_stack([y, z]))
    verts = path.interpolated(steps=3).vertices
    yy, xx = verts[:, 0], verts[:, 1]
    _ = colorline(ax3, yy, zz, cmap=plt.get_cmap(cmap), linewidth=4, alpha=alpha)


def plot_integration(star_samples, name, forward_orbit_samples, reverse_orbit_samples):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(25,9), sharey=True)

    for i, star in enumerate(star_samples):
        alpha = 1 if i == 0 else 0.05
        alpha2 = 1 if i == 0 else alpha*0.25
        
        fwd_pos = [forward_orbit_samples.x.value[:,i], forward_orbit_samples.y.value[:,i], forward_orbit_samples.z.value[:,i]]
        rev_pos = [reverse_orbit_samples.x.value[:,i], reverse_orbit_samples.y.value[:,i], reverse_orbit_samples.z.value[:,i]]
        
        plot_orbit(fwd_pos, (ax1, ax2, ax3), 'autumn', alpha)
        plot_orbit(rev_pos, (ax1, ax2, ax3), 'winter', alpha)

        ax1.scatter(star.x, star.y, marker='*', s=900, c='green', zorder=5, alpha=alpha2, rasterized=True)
        ax1.arrow(star.x.value, star.y.value, dx=star.v_x.value/25, dy=star.v_y.value/25, width=0.9, ec='green', fc='green', zorder=5, alpha=alpha2, rasterized=True)

        ax2.scatter(star.x, star.z, marker='*', s=900, c='green', zorder=5, alpha=alpha2, rasterized=True)
        ax2.arrow(star.x.value, star.z.value, dx=star.v_x.value/25, dy=star.v_z.value/25, width=0.9, ec='green', fc='green', zorder=5, alpha=alpha2, rasterized=True)

        ax3.scatter(star.y, star.z, marker='*', s=900, c='green', zorder=5, alpha=alpha2, rasterized=True)
        ax3.arrow(star.y.value, star.z.value, dx=star.v_y.value/25, dy=star.v_z.value/25, width=0.9, ec='green', fc='green', zorder=5, alpha=alpha2, rasterized=True)

    plt.suptitle(name, y=0.96, size=40)
    ax1.set_xlabel('X [kpc]')
    ax1.set_ylabel('Y [kpc]')
    ax2.set_xlabel('X [kpc]')
    ax2.set_ylabel('Z [kpc]')
    ax3.set_xlabel('Y [kpc]')
    ax3.set_ylabel('Z [kpc]')

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(20))

        ax.tick_params(axis='both', which='major', direction='out', length=6, width=3, left=True, labelsize='medium', pad=10)
        ax.tick_params(axis='both', which='minor', direction='out', length=3, width=3, left=True, labelsize='medium', pad=10)

        ax.scatter([0],[0],c='k', s=200)
        ax.set_xlim([-50,50])
        ax.set_ylim([-50,50])
    return fig, (ax1, ax2, ax3)


if __name__ == "__main__":
    np.random.seed(2)
    HVS = pd.read_csv("HVSs_lim.csv")

    ra = str_to_astropy(HVS['gaia_ra'])
    dec = str_to_astropy(HVS['gaia_dec'])
    dist = str_to_astropy(HVS['dist'])
    dist_unc = str_to_astropy(HVS['dist_unc'])

    pmra = str_to_astropy(HVS['pmra'])
    pmra_unc = str_to_astropy(HVS['pmra_unc'])
    pmdec = str_to_astropy(HVS['pmdec'])
    pmdec_unc = str_to_astropy(HVS['pmdec_unc'])
    rv = str_to_astropy(HVS['rv'])
    rv_unc = str_to_astropy(HVS['rv_unc'])

    pmra_cosdec = [pr*np.cos(d) for (pr, d) in zip(pmra, dec)]
    pmra_cosdec_unc = [pr_u*np.cos(d) for (pr_u, d) in zip(pmra_unc, dec)]

    names = HVS['name']
    stars = coord.SkyCoord(ra=ra, dec=dec, distance=dist, 
                           pm_ra_cosdec=pmra_cosdec, pm_dec=pmdec, 
                           radial_velocity=rv)

    star_uncs = coord.SkyCoord(ra=np.zeros(len(stars))*u.deg, 
                                dec=np.zeros(len(stars))*u.deg, 
                                distance=dist_unc, pm_ra_cosdec=pmra_cosdec_unc,
                                pm_dec=pmdec_unc, radial_velocity=rv_unc)

    potential = gp.MilkyWayPotential()
    galcen_frame = coord.Galactocentric()
    _ = coord.galactocentric_frame_defaults.set('v4.0')

    for star, unc, name in zip(stars, star_uncs, names):
        n_samples = 100
        
        dist = np.random.normal(star.distance.value, 
                                unc.distance.value,
                                n_samples) * star.distance.unit
        dist = np.concatenate(([star.distance], dist))

        pm_ra_cosdec = np.random.normal(star.pm_ra_cosdec.value,
                                        unc.pm_ra_cosdec.value,
                                        n_samples) * star.pm_ra_cosdec.unit
        pm_ra_cosdec = np.concatenate(([star.pm_ra_cosdec], pm_ra_cosdec))

        pm_dec = np.random.normal(star.pm_dec.value,
                                unc.pm_dec.value,
                                n_samples) * star.pm_dec.unit
        pm_dec = np.concatenate(([star.pm_dec], pm_dec))

        rv = np.random.normal(star.radial_velocity.value,
                            unc.radial_velocity.value,
                            n_samples) * star.radial_velocity.unit
        rv = np.concatenate(([star.radial_velocity], rv))

        ra = np.full(n_samples+1, star.ra.degree) * u.degree
        dec = np.full(n_samples+1, star.dec.degree) * u.degree

        star_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                            pm_ra_cosdec=pm_ra_cosdec,
                            pm_dec=pm_dec, radial_velocity=rv)

        star_samples = star_samples.transform_to(galcen_frame)

        w0_samples = gd.PhaseSpacePosition(star_samples.data)
        forward_orbit_samples = potential.integrate_orbit(w0_samples, dt=0.25*u.Myr, n_steps=4000)
        reverse_orbit_samples = potential.integrate_orbit(w0_samples, dt=-0.25*u.Myr, n_steps=4000)

        plot_integration(star_samples, name, forward_orbit_samples, reverse_orbit_samples)
        plt.savefig(f"orbit_figs/{name}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Finished {name} orbit plot")

    fig, ax = plt.subplots(figsize=(10,7))
    for i, (star, unc, name) in enumerate(zip(stars, star_uncs, names)):
        n_samples = 250
        
        dist = np.random.normal(star.distance.value, 
                                unc.distance.value,
                                n_samples) * star.distance.unit

        pm_ra_cosdec = np.random.normal(star.pm_ra_cosdec.value,
                                        unc.pm_ra_cosdec.value,
                                        n_samples) * star.pm_ra_cosdec.unit

        pm_dec = np.random.normal(star.pm_dec.value,
                                unc.pm_dec.value,
                                n_samples) * star.pm_dec.unit

        rv = np.random.normal(star.radial_velocity.value,
                            unc.radial_velocity.value,
                            n_samples) * star.radial_velocity.unit

        ra = np.full(n_samples, star.ra.degree) * u.degree
        dec = np.full(n_samples, star.dec.degree) * u.degree

        samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                            pm_ra_cosdec=pm_ra_cosdec,
                            pm_dec=pm_dec, radial_velocity=rv)

        samples = samples.transform_to(galcen_frame)
        sample_radii = [np.sqrt(x**2 + y**2 + z**2).value for x, y, z in zip(samples.x, samples.y, samples.z)]
        sample_vel3d = [np.sqrt(vx**2 + vy**2 + vz**2).value for vx, vy, vz in zip(samples.v_x, samples.v_y, samples.v_z)]
    
        star = star.transform_to(galcen_frame)
        star_radius = np.sqrt(star.x**2 + star.y**2 + star.z**2).value
        star_vel3d = np.sqrt(star.v_x**2 + star.v_y**2 + star.v_z**2).value

        radial_grid = np.linspace(0,20,100)
        vesc = np.sqrt(-2*potential.energy([radial_grid, np.zeros(100), np.zeros(100)])).to(u.km/u.s)

        colors = ["darkred", "darkolivegreen", "salmon", "springgreen", "saddlebrown", "orange", "gold", "olive", "red", "chartreuse", "orangered", "turquoise", "teal", "darkorchid", "dodgerblue", "slategray", "blue", "blueviolet", "crimson", "magenta", "cyan"]

        ax.plot(radial_grid, vesc, linestyle='dashed', linewidth=2, c='k')
    
        ax.scatter(star_radius, star_vel3d, s=200, c=colors[i], marker='*', edgecolor='k', zorder=5)
        ax.scatter(sample_radii, sample_vel3d, s=10, c=colors[i], marker='.', alpha=0.5)

    v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()
    v3_sun = (v_sun.x.value**2 + v_sun.y.value**2 + v_sun.z.value**2)**0.5  # km/s
    r_sun = coord.Galactocentric().galcen_distance.value  # kpc
    ax.scatter(r_sun, v3_sun, s=200, c='darkorange', marker='*')
    ax.text(x=r_sun+0.09, y=v3_sun-10, s=r"Sun", size=16)

    ax.set_xlim([6,11])
    ax.set_ylim([200,800])

    ax.set_xlabel(r"Galactocentric radius [kpc]")
    ax.set_ylabel(r"$v_{\mathrm{3D}}$ [km s$^{-1}$]")

    ax.tick_params(axis='y', which='major', direction='out', length=6, width=3, left=True, labelsize='medium', pad=10)
    ax.tick_params(axis='y', which='minor', direction='out', length=3, width=3, left=True, labelsize='medium', pad=10)

    ax.tick_params(axis='x', which='major', direction='out', length=6, width=3, left=True, right=True, labelsize='medium', pad=10)
    ax.tick_params(axis='x', which='minor', direction='out', length=3, width=3, left=True, right=True, labelsize='medium', pad=10)

    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(200))
    plt.savefig("unbounds.pdf", bbox_inches='tight')
    plt.close()
    print("Finished bound/unbound plot\n")
