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
                              linewidth=linewidth, alpha=alpha)

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


def plot_orbit(orbit, axs, cmap):
    x = orbit.x.value
    y = orbit.y.value
    z = orbit.z.value

    ax1, ax2, ax3 = axs
    
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    xx, yy = verts[:, 0], verts[:, 1]
    _ = colorline(ax1, xx, yy, cmap=plt.get_cmap(cmap), linewidth=4)
    
    path = mpath.Path(np.column_stack([x, z]))
    verts = path.interpolated(steps=3).vertices
    xx, zz = verts[:, 0], verts[:, 1]
    _ = colorline(ax2, xx, zz, cmap=plt.get_cmap(cmap), linewidth=4)
    
    path = mpath.Path(np.column_stack([y, z]))
    verts = path.interpolated(steps=3).vertices
    yy, xx = verts[:, 0], verts[:, 1]
    _ = colorline(ax3, yy, zz, cmap=plt.get_cmap(cmap), linewidth=4)


def plot_integration(star, name, forward_orbit, reverse_orbit):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(25,9), sharey=True)
    
    plot_orbit(forward_orbit, (ax1, ax2, ax3), 'autumn')
    plot_orbit(reverse_orbit, (ax1, ax2, ax3), 'winter')

    plt.suptitle(name, y=0.96, size=40)
    
    ax1.scatter(star.x, star.y, marker='*', s=900, c='green', zorder=5)
    ax1.arrow(star.x.value, star.y.value, dx=star.v_x.value/25, dy=star.v_y.value/25, width=0.9, ec='green', fc='green', zorder=5)
    ax1.set_xlabel('X [kpc]')
    ax1.set_ylabel('Y [kpc]')
    
    
    ax2.scatter(star.x, star.z, marker='*', s=900, c='green', zorder=5)
    ax2.arrow(star.x.value, star.z.value, dx=star.v_x.value/25, dy=star.v_z.value/25, width=0.9, ec='green', fc='green', zorder=5)
    ax2.set_xlabel('X [kpc]')
    ax2.set_ylabel('Z [kpc]')
    
    
    ax3.scatter(star.y, star.z, marker='*', s=900, c='green', zorder=5)
    ax3.arrow(star.y.value, star.z.value, dx=star.v_y.value/25, dy=star.v_z.value/25, width=0.9, ec='green', fc='green', zorder=5)
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
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def plot_unbound(rgrid, vesc, r, v3, names):
    srt = np.argsort(v3)
    v3 = np.asarray(v3)[srt]
    r = np.asarray(r)[srt]
    names = names.to_numpy()[srt]

    fig, ax = plt.subplots(figsize=(10,7))
    plt.plot(rgrid, vesc, linestyle='dashed', linewidth=2, c='k')
    
    plt.scatter(r[:-1], v3[:-1], s=20, c='k')
    plt.scatter(r[-1], v3[-1], c='r')
    plt.text(x=r[-1]+0.11, y=v3[-1]-10, s=names[-1], size=16)

    v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()
    v3_sun = (v_sun.x.value**2 + v_sun.y.value**2 + v_sun.z.value**2)**0.5  # km/s
    r_sun = coord.Galactocentric().galcen_distance.value  # kpc
    plt.scatter(r_sun, v3_sun, s=20, c='darkorange')
    plt.text(x=r_sun+0.11, y=v3_sun-10, s=r"Sun", size=16)
    
    plt.xlim([5,15])
    plt.ylim([200,800])

    ax.set_xlabel(r"Galactocentric radius [kpc]")
    ax.set_ylabel(r"$v_{\mathrm{esc}}$ [km s$^{-1}$]")

    ax.tick_params(axis='y', which='major', direction='out', length=6, width=3, left=True, labelsize='medium', pad=10)
    ax.tick_params(axis='y', which='minor', direction='out', length=3, width=3, left=True, labelsize='medium', pad=10)

    ax.tick_params(axis='x', which='major', direction='out', length=6, width=3, left=True, right=True, labelsize='medium', pad=10)
    ax.tick_params(axis='x', which='minor', direction='out', length=3, width=3, left=True, right=True, labelsize='medium', pad=10)

    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(200))

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
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
    # pmra_cosdec_unc = [pr_u*np.cos(d) for (pr_u, d) in zip(pmra_unc, dec)]

    names = HVS['name']

    stars = coord.SkyCoord(ra=ra, dec=dec, distance=dist, 
                           pm_ra_cosdec=pmra_cosdec, pm_dec=pmdec, 
                           radial_velocity=rv)

    # stars_uncs = coord.SkyCoord(ra=np.zeros(len(stars))*u.deg, 
    #                             dec=np.zeros(len(stars))*u.deg, 
    #                             distance=dist_unc, pm_ra_cosdec=pmra_cosdec_unc,
    #                             pm_dec=pmdec_unc, radial_velocity=rv_unc)

    galcen_frame = coord.Galactocentric()
    stars = stars.transform_to(galcen_frame)
    radii = [np.sqrt(x**2 + y**2 + z**2).value for x, y, z in zip(stars.x, stars.y, stars.z)]
    vel3d = [np.sqrt(vx**2 + vy**2 + vz**2).value for vx, vy, vz in zip(stars.v_x, stars.v_y, stars.v_z)]
    
    _ = coord.galactocentric_frame_defaults.set('v4.0')

    potential = gp.MilkyWayPotential()

    radial_grid = np.linspace(0,20,100)
    vesc = np.sqrt(-2*potential.energy([radial_grid, np.zeros(100), np.zeros(100)])).to(u.km/u.s)

    fig, ax = plot_unbound(radial_grid, vesc.value, radii, vel3d, HVS['name'])
    plt.savefig("unbounds.pdf", bbox_inches='tight')
    plt.close()
    print("Finished bound/unbound plot\n")

    for (star, name) in zip(stars, names):
        w0 = gd.PhaseSpacePosition(pos=star.data, vel=star.velocity)
        forward_orbit = potential.integrate_orbit(w0, dt=0.25*u.Myr, n_steps=3000)
        reverse_orbit = potential.integrate_orbit(w0, dt=-0.25*u.Myr, n_steps=3000)

        plot_integration(star, name, forward_orbit, reverse_orbit)
        plt.savefig(f"orbit_figs/{name}_orbit.pdf", bbox_inches='tight')
        plt.close()
        print(f"Finished {name} orbit plot")
    print()
