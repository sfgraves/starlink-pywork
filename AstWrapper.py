from astropy.wcs.wcsapi import BaseLowLevelWCS, wcs_info_str, HighLevelWCSMixin
import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries

import astropy.coordinates as coords

from itertools import combinations
import collections

import logging
logger = logging.getLogger(__name__)


from astropy.coordinates import SkyCoord

class AstWCSLowLevel(BaseLowLevelWCS):

    def __init__(self, frameset, arrayshape=None):
        """
        This assumes you've passed in an Ast frameset with a base Grid
        frame and a current WCS frame.

        I.e. Nin is the number of pixel axes, Nout is the number of
        WCS axes.

        NDF Grid indices start at 1, Python pixel indices
        start at 0.

        Python is ordered [z,y,x], Ast is ordered [x,y,z], but this
        interface has methods for both, named Array (z,y,x) and
        Pixel(x,y,z)

        arrayshape, if given, is in python order.

        """

        self._ast = frameset
        if arrayshape is not None:
            self._arrayshape = list(arrayshape)
        else:
            self._arrayshape = None
        self._pixelshape = None

        # Sky Information: we need to know which wcs axis is part of a sky frame.
        self._wcsaxis_is_sky = [frameset.get('Domain({})'.format(i)).upper()=='SKY'
                                               for i  in range(1, self._ast.Nout + 1)]

        # Save the title. If its a component frame, try decomposing it
        # (once) and get the underlying types.
        self.domain = frameset.Domain
        self.system = frameset.System
        self.title = '{} ({})'.format(frameset.Title, frameset.Domain)
        self.subtitle1 = None
        self.subtitle2 = None
        if self.system.lower() == 'compound':
            frame1, frame2, _, _, _ = frameset.getframe(frameset.Current).decompose()
            if frame2:
                self.subtitle1 = '{} ({}-{})'.format(frame1.Title, frame1.Domain, frame1.System)
                self.subtitle2 = '{} ({}-{})'.format(frame2.Title, frame2.Domain, frame2.System)






    @property
    def pixel_n_dim(self):
        return self._ast.Nin

    @property
    def world_n_dim(self):
        return self._ast.Nout


    # Pixel and Array properties of initial data.
    @property
    def pixel_shape(self):
        if not self._pixelshape and self._arrayshape:
            pixel_shape = self._arrayshape.copy()
            pixel_shape.reverse()
            self._pixelshape = pixel_shape
        return self._pixelshape

    @property
    def array_shape(self):
        return self._arrayshape


    @property
    def pixel_bounds(self):
        """
        From astropy docs:
        The bounds (in pixel coordinates) inside which the WCS is defined,
        as a list with ``pixel_n_dim`` ``(min, max)`` tuples (optional).

        The bounds should be given in ``[(xmin, xmax), (ymin, ymax)]``
        order. WCS solutions are sometimes only guaranteed to be accurate
        within a certain range of pixel values, for example when defining a
        WCS that includes fitted distortions. This is an optional property,
        and it should return `None` if a shape is neither known nor relevant.
        """
        # Ask DSB how to identify regions that frame is valid over?
        # SUN/211 section 5.9 suggests that mappings will return bad
        # values for coordinates that don't have a valid value -- not
        # sure how to turn this into bounds. In general this probably
        # isn't simple -- area may well not be a simple rectangle
        # after all. Perhaps require user to provide it if wanted?
        return None




    # These 3 should 1) cache this and 2)only do it once.
    @property
    def _astropy_axis_information(self):
        return _astropy_axis_information(self._ast)

    @property
    def world_axis_physical_types(self):
        return self._astropy_axis_information[0]


    # This interface has astropy objects instead of serialized
    # classes.
    @property
    def serialized_classes(self):
        return False

    @property
    def world_axis_names(self):
        names = [self._ast.get('Label({})'.format(i))
                 for i in range(1, self.world_n_dim + 1)]
        return names


    @property
    def world_axis_object_components(self):
        return self._astropy_axis_information[1]

    @property
    def world_axis_object_classes(self):
        return self._astropy_axis_information[2]

    @property
    def world_axis_units(self):
        """
        Returns an iterable of strings given the units of the world
        coordinates for each axis. The strings should follow the recommended
        VOUnit standard (though as noted in the VOUnit specification
        document, units that do not follow this standard are still allowed,
        but just not recommended).
        """
        # What order should this be in? Assume world order for now.
        # Use the internal units.
        world_axes_units = []
        for i in range(1, self.world_n_dim + 1):
            unit = self._ast.get('InternalUnit({})'.format(i))

            # If we're  in a sky frame, we want to use degrees instead of radians.
            if self._wcsaxis_is_sky[i-1]:
                unit = 'deg'
            try:
                unit = u.Unit(unit)
                unit = unit.to_string(format='VOUnit')
            except u.UnitsError:
                unit = ''
            world_axes_units.append(unit)
        return world_axes_units



    @property
    def axis_correlation_matrix(self):
        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        return _axis_correlation_matrix(mapping)

    def pixel_to_world_values(self, *pixel_arrays):
        """Convert pixel coordinates to world coordinates.

        This method takes n_pixel scalars or arrays as input, and pixel
        coordinates should be zero-based. Returns n_world scalars or
        arrays in units given by ``world_axis_units``. Note that pixel
        coordinates are assumed to be 0 at the center of the first pixel
        in each dimension. If a pixel is in a region where the WCS is
        not defined, NaN can be returned. The coordinates should be
        specified in the ``(x, y)`` order, where for an image, ``x`` is
        the horizontal coordinate and ``y`` is the vertical coordinate.

        Expects an input of ([xvalues], [yvalues], [zvalues])

        *Sky coords special case: Ast uses radians as the internal
        unit. In practice everything will work better if we use degrees
        for input output, as users' will almost never want
        radians. Therefore there will be an annoying hack where this
        wrapper will convert skyframe values from radians to degrees.
        """
        # Deal with case where you passed in scalars for each axis instead of array.
        try:
            len(pixel_arrays[0])
            pixel_arrays = np.asarray(pixel_arrays)
        except TypeError:
            pixel_arrays = np.asarray([[i] for i in pixel_arrays])

        grid_arrays = pixel_arrays + 1
        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        world_values = mapping.tran(grid_arrays, 1)


        # If SkyFrame, convert from radians to deg.
        for ax in range(0, self.world_n_dim):
            if self._wcsaxis_is_sky[ax]:
                world_values[ax]  = u.rad.to(u.deg)*world_values[ax]
        return world_values[0] if self.world_n_dim == 1 else world_values



    def world_to_pixel_values(self, *world_arrays):
        """Convert world coordinates to pixel coordinates.

        This method takes n_world scalars or arrays as input in units
        given by ``world_axis_units``.  Returns n_pixel scalars or
        arrays. Note that pixel coordinates are assumed to be 0 at the
        center of the first pixel in each dimension.  to be 0 at the
        center of the first pixel in each dimension. If a world
        coordinate does not have a matching pixel coordinate, NaN can be
        returned.  The coordinates should be returned in the ``(x, y)``
        order, where for an image, ``x`` is the horizontal coordinate
        and ``y`` is the vertical coordinate.

        *Sky coords special case: Ast uses radians as the internal unit. In
        practice everything will work better if we use degrees, as users'
        will almost never want radians. Therefore there will be an annoying
        hack where this wrapper will expect 'degrees' for any SkyFrame
        defined in radians, and this interface will convert the input
        appropriately before passing it into Ast.
        """
        try:
            len(world_arrays[0])
            world_arrays = np.asarray(world_arrays)

        except TypeError:
            world_arrays = np.asarray([[i] for i in world_arrays])

        # If  its a sky  frame, convert from  degrees to radians.
        for ax in range(0, self.world_n_dim):
            if self._wcsaxis_is_sky[ax]:
                world_arrays[ax]  = u.deg.to(u.rad) *world_arrays[ax]

        mapping = self._ast.getmapping(self._ast.Current, self._ast.Base)
        grid_values = mapping.tran(world_arrays, 1)
        pixel_values = grid_values - 1
        # Do these need to be converted to integers?
        return pixel_values[0] if self.pixel_n_dim == 1 else pixel_values


    def world_to_array_index_values(self, *world_arrays):
        """Convert world coordinates to array indices.

        This is the same as ``world_to_pixel_values`` except that the
        indices should be returned in ``(i, j)`` order, where for an
        image ``i`` is the row and ``j`` is the column (i.e. the opposite
        order to ``pixel_to_world_values``).  The indices should be
        returned as rounded integers.
        """
        index_arrays = self.pixel_to_world_values(*world_arrays)[::-1]
        return index_arrays


    def array_index_to_world_values(self, *index_arrays):
        """Convert array indices to world coordinates.

        This is the same as ``pixel_to_world_values`` except that the
        indices should be given in ``(i, j)`` order, where for an image
        ``i`` is the row and ``j`` is the column (i.e. the opposite
        order to ``pixel_to_world_values``).
        """

        world = self.world_pixel_values(*index_arrays[::-1])
        return world


    def __repr__(self):
        return wcs_info_str(self)


_AST_ALIASES = {
    'ENER': 'ENERGY',
    'WAVN': 'WAVENUM',
    'WAVE': 'WAVELEN',
    'AWAV': 'AIRWAVE',
    'VRAD': 'VRADIO',
    'VOPT': 'VOPTICAL',
    'ZOPT': 'REDSHIFT',
    'VELO': 'VREL',
    'GAPPT': 'GEOCENTRIC',
    'APPARENT': 'GEOCENTRIC',

    }
_PHYSICAL_TYPE = {
    "FREQ" : 'em.freq',
    "ENERGY" : 'em.energy',
    "WAVENUM" : 'em.wavenumber',
    "WAVELEN" : 'em.wl',
    "AIRWAVE" : 'custom:em.wl.air',
    "VRADIO" : 'spect.dopplerVeloc.radio',
    "VOPTICAL" : 'spect.dopplerVeloc.opt',
    "REDSHIFT" : 'src.redshift',
    "BETA" : 'custom:em.beta',
    "VREL" :'spect.dopplerVeloc',
    'RA': 'pos.eq.ra',
    'DEC': 'pos.eq.dec',
    'GLAT': 'pos.galactic.lat',
    'GLON': 'pos.galactic.lon',
    'AZEL': ('pos.az.alt','pos.az.azi'),
}



_RADECSYSTEMS = ['ICRS', 'FK5', 'FK4', 'FK4-NO-E']



_ASTROPY_FRAME_NAMES = {
    'ICRS': 'icrs',
    'FK5': 'fk5',
    'FK4': 'fk4',
    'FK4-NO-E': 'fk4noeterms',
    'EQUATORIAL': 'fk5',
}


def _astropy_axis_information(frame):

    """
    Get types, components and classes for Astropy APE14 serialisation

    frame: an Ast frame/frameset

    returns: types, components and classes following APE-14 spec.

    Known limitations:
    This definitely will fail if there are two of the same type of Sky
    frame in the current frame (i.e an FK5-SKY-FK5-SKY Cmp frame).

    AZEL/geocentri/equatorial frames not yet implemented.

    Doesn't properly handle time frames.

    Spec frames just converted to units as no spectral coordinate
    frame in astropy yet.

    Physical types from ucd dictionary pretty limited.
    """

    object_components = ['']*frame.Nout
    object_classes = {}
    physical_types = ['']*frame.Nout


    # Go through each output value
    for axis in range(1, frame.Nout+1):
        domain = frame.get('Domain({})'.format(axis))
        system = frame.get('System({})'.format(axis))
        unit = frame.get('InternalUnit({})'.format(axis))
        system =  _AST_ALIASES.get(system, system)
        IDstring = '{}-{}'.format(domain, system)

        if domain == 'SKY':
            result = _create_astropy_sky_frame(frame, axis,
                                      distinguish_idstring_withaxis=False)

        elif domain == 'TIME':
            result = _create_astropy_time_frame(frame, axis)

        else:
            if frame.Nout > 1:
                IDstring += '-{}'.format(axis-1)
            object_comp = (0, 'value')
            object_class = (Quantity,
                            (),
                            {'unit':unit})
            if system in _PHYSICAL_TYPE:
                physical_type = _PHYSICAL_TYPE[system]
            else:
                axislabel = frame.get('Label({})'.format(axis))
                physical_type = 'custom:{}-{}.{}'.format(
                    domain, system, axislabel)
            result = (IDstring, physical_type, object_comp, object_class)



        (IDstring, physical_type, object_comps, object_class) = result
        object_classes[IDstring] = object_class
        physical_types[axis-1] = physical_type
        object_components[axis-1] = tuple([IDstring] + list(object_comps))


    return physical_types, object_components, object_classes



# SkyFrame can have the following system: AZEL, ECLIPTIC, FK4, FKR_NO_E/FK4_NO_E,
# FK5/EQUATORIAL (requires equinox), GALACTIC, GAPPT/GEOCENTRIC/APPARENT, HELIOECLIPTIC,
# ICRS, J2000 (no equinox!), SUPERGALACTIC, UNKNOWN.

#SpecFrame systems.: FREQ, ENERGY, WAVENUM/WAVN, WAVE/WAVELEN, AWAV/AIRWAVE, VRAD/VRADIO, VOPT/VOPTICAL, ZOPT/REDSHIFT, BETA, VELO/VREL.


#TimeFrame systems:
# MJD, JD, JEPOCH, BEPOCH

#FluxFrame systems:
#FLXDN, FLXDNW, SFCBR, SFCBRW


def _get_observer_lat_or_lon_asfloat(value):
    if value[0] in ('N', 'E'):
        mult = 1.0
    elif value[0] in ('S', 'W'):
        mult = -1.0

    # strip off leading letter.
    value = value[1:]
    values = [float(i) for i in value.split(':')]
    output = values[0] + values[1]/60.0 + values[2]/(60.0*60.0)
    return output * mult

def _get_ast_epochequinox(epoch):
    """
    Format an Ast epoch/equinox attribute for astropy.

    From SUN/211: if < 1984.0 its in Besselian, if greater its in
    Julian. If blank string return None.

    Doesn't yet handle weird values sensibly (should it raise an error?)
    """
    if epoch == '':
        epoch = None
    elif float(epoch) < 1984.0:
        epoch = 'B' + str(epoch)
    elif epoch:
        epoch = 'J' + str(epoch)

    return epoch


_TIMEFORMAT_LOOKUP = {
    'JEPOCH': 'jyear',
    'BEPOCH': 'byear',
    'MJD': 'jd',
}


def _create_astropy_time_frame(frame, axis):
    system = frame.get('System({})'.format(axis))
    unit = frame.get('InternalUnit({})'.format(axis))
    unit = u.Unit(unit)

    # Options for system are MJD, JD, JEPOCH and BEPOCH.
    astropyformat = _TIMEFORMAT_LOOKUP.get(system, system.lower())
    IDstring = '{}-{}'.format('TIME', system)

    # LTOffset, TimeOrigin, TimeSCale
    timescale = frame.get('TimeScale({})'.format(axis))

    # Create the astropy earth location information.
    earthlocation = _get_observer_earth_location(frame, axis)


    # Handle all non-sidereal times:
    if timescale not in SIDEREALTIMES:
        object_comps = (0, 'value')
        physical_type = 'time'


        # If the origin is non 0, create a time delta object, and warn
        # that the information about the origin has been lost.
        timeorigin = float(frame.get('TimeOrigin({})'.format(axis))) * unit
        if timeorigin.value != 0:
            object_comps = ('time_delta', 'value')
            timeclass = (TimeSeries,
                         (),
                         {'format': astropyformat,
                          'scale': timescale.lower(),
                          'timestart': timeorigin,
                          }
                        )

        else:
            if timescale == 'LT':
                timescale = 'local'
            timeclass = (Time,
                         (),
                         {'format':astropyformat,
                          'scale':timescale.lower(),
                          'location': earthlocation,
                      })

    # Deal with sidereal times.
    else:
        raise NotImplementedError

        # Unclear how to do this. Astropy doesn't handle sidereal times as a
        # time, but instead as a Longitude?

        # This would mean having to provide the values in angular units
        # Will definitely need the earth location though.
        # earthlocation = _get_observer_earth_location(frame, axis)

    return IDstring, physical_type, object_comps, timeclass


def _create_astropy_sky_frame(frame, axis,
                              distinguish_idstring_withaxis=False):
    """
    Create an astropy Sky frame (astropy.coords) from an Ast SkyFrame.
    """
    system = frame.get('System({})'.format(axis))
    system =  _AST_ALIASES.get(system, system)
    IDstring = '{}-{}'.format('SKY', system)
    if distinguish_idstring_withaxis:
        IDstring = IDstring + '-{}'.format(axis)

    # SKy frames: always have a lat and lon axis. Always 2
    # axes. Internal unit is always radians, but we want the output to
    # be in degrees..  Check if axis is latitude or longitude.

    #Set up each of Longitude and Latitude Axes.
    lataxis = int(frame.get('LatAxis'))
    lonaxis = int(frame.get('LonAxis'))
    if axis==lataxis:
        AXISTYPE = 'LATITUDE'
    elif axis==lonaxis:
        AXISTYPE = 'LONGITUDE'
    else:
        raise NotImplementedError

    # Get the physical type (as ucd), and the object component
    # information (as a tuple) for this axis.
    systemtype= _GET_SYSTEM_TYPE.get(system, system)
    try:
        physical_type, object_comps = _GET_LATLONG_PHYS_OBJCOMP[AXISTYPE][systemtype]
        object_comps = object_comps
    except KeyError:
        raise NotImplementedError

    # Create the astropy skyclass information.
    earthlocation = _get_observer_earth_location(frame, axis)
    epoch = frame.get('Epoch({})'.format(axis))
    epoch = _get_ast_epochequinox(epoch)
    equinox = frame.get('Equinox({})'.format(axis))
    equinox = _get_ast_epochequinox(equinox)
    astropy_framename =  _ASTROPY_FRAME_NAMES.get(system, system.lower())
    skyclass = (SkyCoord,
                (),
                {'frame': astropy_framename,
                 'unit': u.deg,
                 'equinox': equinox,
                 'obstime': epoch,
             })
    return IDstring, physical_type, object_comps, skyclass


_GET_SYSTEM_TYPE = {
    'ICRS' : 'RADEC_TYPE',
    'FK5' : 'RADEC_TYPE',
    'FK4' : 'RADEC_TYPE',
    'FK4-NO-E': 'RADEC_TYPE',
}

_GET_LATLONG_PHYS_OBJCOMP = {
    'LONGITUDE': {
        'RADEC_TYPE': ('pos.eq.ra', (0, 'ra.deg')),
        'GALACTIC': ('pos.galactic.lon', (0, 'b.deg')),
        'AZEL': ('pos.az.azi', (0, 'azi.deg')),
    },
    'LATITUDE': {
        'RADEC_TYPE': ('pos.eq.dec',(1, 'dec.deg')),
        'GALACTIC': ('pos.galactic.lat',(1, 'l.deg')),
        'AZEL': ('pos.az.alt.', (1, 'alt.deg')),
    }
}


def _get_observer_earth_location(frame, axis):
    earthlon = _get_observer_lat_or_lon_asfloat(
                frame.get('ObsLon({})'.format(axis)))*u.deg

    earthlat =  _get_observer_lat_or_lon_asfloat(
        frame.get('ObsLat({})'.format(axis)))*u.deg

    earthheight = float(frame.get('ObsAlt({})'.format(axis))) * u.m
    earthlocation = coords.EarthLocation(lon=earthlon.value,
                                         lat=earthlat.value,
                                         height=earthheight.value,
                                         )
    return earthlocation


# Timescales lookup

{'TAI': 'tai',
'UTC': 'utc',
'UT1': 'ut1',
'TT': 'tt',
'TDB': 'tdb',
'TCB': 'tcb',
'TCG': 'tcg',
'LT': 'local',
}

SIDEREALTIMES = ('LAST', 'LMST', 'GMST')



# def _convert_to_asttran_shape(n_dim, coord_arrays):
#     """
#     Convert set of n_dim scalars/arrays into shape for mapping.tran
#     """
#     # Convert to a numpy array for convenience.
#     coord_arrays = np.array(coord_arrays)

#     # If you're given a set of n_dim scalars, ensure these
#     # are in a sensible array form.
#     if coord_arrays.shape == (n_dim,):
#         coord_arrays = coord_arrays.reshape(n_dim, 1)
#     return coord_arrays

# def _convert_from_asttran_shape(n_dim, coord_arrays):
#     """
#     Convert the output from mapping.tran into format for astropy.
#     """
#     # Convert to a numpy array for convenience.
#     coord_arrays = np.array(coord_arrays)

#     if coord_arrays.shape == (n_dim, 1):
#         coord_arrays.reshape(n_dim)
#     return coord_arrays


def _axis_correlation_matrix(mapping):

    """
    Given an Ast.mapping, return the axis correlation matrix.

    It will be of shape (Nout,Nin). It uses the mapping.mapsplit to
    separate out component mappings. Assumes everything eles links
    to all output axes. Only tries to decompose at the top leve.

    TODO: Check this properly against complicated WCS set ups? Not
    sure if this is right.
    """

    n_input = mapping.Nin
    n_output = mapping.Nout
    matrix = np.zeros((n_output, n_input), dtype=int)

    # Create a flat list of all possible combinations of input pixel
    # axes It is important these are in order from fewest to largest.
    inputs =  range(1, n_input+1)
    combos = [j for i in range(1, n_input+1)
                for j in list(combinations(inputs, i))]


    #Keep track of input axes that have already been linked to an
    # output axis.
    already_linked_input_axes = []


    # Go through all input combinations, check if any of the input
    # dimensions have already been included, and then see if they lead
    # to a an output axis by using mapping.mapsplit
    for i in combos:
        intersection = set(i).intersection(already_linked_input_axes)
        if len(intersection) == 0:
            linking, outmap = mapping.mapsplit(i)
            if outmap is not None:
                for inax in i:
                    already_linked_input_axes.append(inax)
                    for outax in linking[linking!=0]:
                        matrix[outax -1][inax -1] = True

    matrix = matrix.astype(bool)
    return matrix



from starlink import hds
import numpy as np
from starlink import Ast

def get_data_wcs(hdsfile):
    hdsloc = hds.open(hdsfile, 'READ')
    astloc = hdsloc.find('WCS').find('DATA')
    astwcs = [i.decode() for i in astloc.get()]
    astwcs = ''.join([i[1:] if i.startswith('+') else '\n'+i for i in astwcs]).split('\n')
    chan = Ast.Channel(astwcs)
    frameset = chan.read()
    dataloc = hdsloc.find('DATA_ARRAY').find('DATA')
    badvalue = hds.getbadvalue(dataloc.type)
    data = dataloc.get()
    data = np.ma.masked_array(data, data==badvalue)
    hdsloc.annul()
    return data, frameset


# Create a high level object, either from a file or from an ast frameset/mapping.
class AstWCS(HighLevelWCSMixin):
    def __init__(self, input_, component=None, filetype='sdf', getshape=False):

        # TODO: this should be using python NDF!
        # If input is a string, assume it is either an NDF file or a
        # FITS file, and read in the object
        frameset = None
        shape = None
        # TODO: (Should useNDF interface once that is easy to install)
        if isinstance(input_, str) and filetype=='sdf':
            try:
                hdsloc = hds.open(input_, 'READ')
                if component:
                    comps = component.split('.')
                    for c in comps:
                        hdsloc = hdsloc.find(c)

                astwcs = hdsloc.find('WCS').find('DATA').get()
                astwcs = [i.decode() for i in astwcs]
                astwcs = ''.join(
                    [i[1:] if i.startswith('+') else '\n'+i
                         for i in astwcs]).split('\n')
                chan = Ast.Channel(astwcs)
                frameset = chan.read()
            except:
                logger.error('Could not get a frameset from %s', input_)
                hdsloc.annul()
                raise

            # Get the shape of the data.
            if getshape:
                try:
                    data = hdsloc.find('DATA_ARRAY').find('DATA').get()
                    shape = data.shape
                except:
                    logger.debug('No data component found in input file.')
                    shape = None


            hdsloc.annul()

        # If its a fits file, try and read a
        elif isinstance(input_, str) and filetype=='fits':
            from astropy.io import fits
            from starlink import Atl
            hdulist = fits.open(input_)
            if not component:
                component = 0
            (frameset,encoding) = Atl.readfitswcs( hdulist[component] )
            shape = hdulist[component].shape
            print('shape is',shape)
            hdulist.close()

        # Create low level object from frameset and shape
        print(type(frameset), type(shape))
        self._low_level_wcs = AstWCSLowLevel(frameset, arrayshape=shape)
        self._info_string = None

    @property
    def info_string(self):
        if not self._info_string:
            astropystr = wcs_info_str(self.low_level_wcs)
            lines = astropystr.split('\n')
            lines.insert(1, '')
            lines.insert(2, 'WCS frame: ' +  self.low_level_wcs.title)
            if self.low_level_wcs.subtitle1:
                lines.insert(3, '')
                lines.insert(4, 'WCS subframe1: ' + self.low_level_wcs.subtitle1)
                lines.insert(5, 'WCS subframe2: ' + self.low_level_wcs.subtitle2)
            self._info_string = '\n'.join(lines)
        return self._info_string

    def __repr__(self):
        return self.info_string

    @property
    def low_level_wcs(self):
        return self._low_level_wcs

    @property
    def pixel_n_dim(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`
        """
        return self.low_level_wcs.pixel_n_dim

    @property
    def world_n_dim(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`
        """
        return self.low_level_wcs.world_n_dim

    @property
    def world_axis_physical_types(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_physical_types`
        """
        return self.low_level_wcs.world_axis_physical_types

    @property
    def world_axis_units(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_units`
        """
        return self.low_level_wcs.world_axis_units

    @property
    def array_shape(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape`
        """
        return self.low_level_wcs.array_shape

    @property
    def pixel_shape(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_shape`
        """
        return self.low_level_wcs.pixel_shape

    @property
    def pixel_bounds(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_bounds`
        """
        return self.low_level_wcs.pixel_bounds

    @property
    def axis_correlation_matrix(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.axis_correlation_matrix`
        """
        return self.low_level_wcs.axis_correlation_matrix
