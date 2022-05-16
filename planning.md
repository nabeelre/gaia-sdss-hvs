# Orbital integration and boundedness of Gaia stars



+ Start: SDSS names

+ extract coordinates from name

+ get SDSS RV from spectrum
  + fetch spectrum
  + unpack fits file to plot spectrum
  + get template for spectral type of star
  + cross correlation? to get shift of template -> RV

+ fetch Gaia PMs, parallax, sky position
  + calculate distance from parallax 
  + compare SDSS sky position with Gaia sky position
+ put 6D coords into astropy skyCoord object
+ set up galpy and MW potential model
+ plug star coords into galpy - orbital integration?
+ do once: calculate escape velocity curve from MW potential 
+ compare 3D velocity to escape velocity at galactocentric radius of star
  + V>Vesc unbound; V<Vesc bound



