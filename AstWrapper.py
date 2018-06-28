"""
Quick attempt at an implementation of the low-level WCS object
defined in astropy-APE14, to wrap Starlink's PyAst.

This is definitely not ready for proper use, i.e. buyer beware, here be
dragons etc. This was mostly written as an excuse to understand Ast
better.

Sarah Graves, June 2018

Testing, against test astropy HighlevelWCS implementation (not yet
finalised) worked for a SKY-SPECTRUM frame, and POLANAL-SPECTRUM
frame for the [0,0,0] positions.

astwrapper = astropy_astwrapper(indf.wcs)
high = HighLevelWCS(astwrapper)
worldcoords = high.pixel_to_world(0,0,0)
pixelcoords = high.world_to_pixel(*worldcoords)

worldcoords = high.pixel_to_world([0,1,2],[0,1,2],[0,1,2])
pixelcoords = high.world_to_pixel(*worldcoords)

And check what is returned.
"""


import astropy.units as u
from astropy.units.format import VOUnit
from itertools import combinations
import numpy as np
import astropy.coordinates as coords

class astropy_astwrapper(object):
    def __init__(self, frameset, datashape=None):

        """
        This assumes you've passed in an Ast frameset with a base Grid frame
        and a current WCS frame.

        I.e. Nin is the number of pixel axes, Nout is the number of
        WCS axes.

        NDF Grid indices start at 1, Python pixel indices
        start at 0.

        Python is ordered [z,y,x], Ast is ordered [x,y,z],
        but this interface has methods for both.
        """

        self._ast = frameset
        self._datashape = datashape


    @property
    def pixel_n_dim(self):
        return self._ast.Nin

    @property
    def world_n_dim(self):
        return self._ast.Nout

    @property
    def pixel_shape(self):
        return self._datashape


    @property
    def pixel_bounds(self):
        # Ask DSB how to identify regions that frame is valid over?
        # SUN/211 section 5.9 suggests that mappings will return bad
        # values for coordinates that don't have a valid value -- not
        # sure how to turn this into bounds. In general this probably
        # isn't simple -- area may well not be a simple rectangle
        # after all. Perhaps require user to provide it if wanted?
        raise NotImplementedError


    @property
    def world_axis_physical_types(self):
        return _astropy_axis_information(self._ast)[0]


    @property
    def world_axis_object_components(self):
        return _astropy_axis_information(self._ast)[1]

    @property
    def world_axis_object_classes(self):
        return _astropy_axis_information(self._ast)[2]


    @property
    def world_axis_units(self):
        # What order should this be in? Assume world order for now.
        # Use the internal units.
        world_axes_units = []
        for i in range(1, self.world_n_dim + 1):
            unit = self._ast.get('InternalUnit({})'.format(i))
            try:
                unit = u.Unit(unit)
                unit = unit.to_string(format=VOUnit)
            except u.UnitsError:
                unit = ''
            world_axes_units.append(unit)
        return world_axes_units

    @property
    def axis_correlation_matrix(self):
        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        return _axis_correlation_matrix(mapping)


    # Lot of repeated code in these methods, should consolidate...
    def pixel_to_world_values(self, *pixel_arrays):

        pixel_arrays = _convert_to_asttran_shape(self.pixel_n_dim,
                                                 pixel_arrays)

        # Convert to Ast GRID pixels (center of first pixel =1):
        grid_arrays = pixel_arrays + 1

        #Now use the mapping.tran method to look at this. We use '1'
        #to indicate the correct direction of the transformation.
        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        world_values = mapping.tran(grid_arrays, 1)
        return world_values

    def index_to_world_values(self, *index_arrays):

        pixel_arrays = _convert_to_asttran_shape(self.pixel_n_dim,
                                                      pixel_arrays)

        # Convert to Ast GRID pixels (center of first pixel =1):
        grid_arrays = pixel_arrays + 1

        # Now flip to correct order (x,y,z) instead of (z,y,x):
        cartesian_ordered = np.flipud(grid_arrays)

        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        world_values = mapping.tran(cartesian_ordered, 1)
        return np.flipud(world_values)

    def world_to_pixel_values(self, *world_arrays):
        world_arrays = _convert_to_asttran_shape(self.world_n_dim,
                                                 world_arrays)

        # Need to convert to the internal unit, from the specified
        # unit which is degrees.
        mapping = self._ast.getmapping(self._ast.Current, self._ast.Base)
        grid_values = mapping.tran(world_arrays, 1)
        return grid_values - 1

    def world_to_index_values(self, *world_arrays):
        world_arrays = _convert_to_asttran_shape(self.world_n_dim,
                                                 world_arrays)
        cartesian_ordered = np.flipud(world_arrays)
        mapping = self._ast.getmapping(self._ast.Base, self._ast.Current)
        grid_values = mapping.tran(cartesian_ordered, 1)
        return np.flipud(grid_values - 1)




def _convert_to_asttran_shape(n_dim, coord_arrays):
    """
    Convert set of n_dim scalars/arrays into shape for mapping.tran
    """
    # Convert to a numpy array for convenience.
    coord_arrays = np.array(coord_arrays)

    # If you're given a set of n_dim scalars, ensure these
    # are in a sensible array form.
    if coord_arrays.shape == (n_dim,):
        coord_arrays = coord_arrays.reshape(n_dim, 1)
    return coord_arrays

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
    matrix = np.zeros((n_output, n_input))

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
    return matrix




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

            # SKy frames: always have a lat and lon axis. Always 2
            # axes. Internal unit is always radians.
            try:
                islat = bool(int(frame.get('IsLatAxis({})'.format(axis))))
            except Ast.BADAT:
                islat = False
            try:
                islon = bool(int(frame.get('IsLonAxis({})'.format(axis))))
            except Ast.BADAT:
                islon = False
            if islon:
                if system in _RADECSYSTEMS:
                    physical_types[axis-1] = _PHYSICAL_TYPE['RA']
                    object_components[axis-1] = (IDstring, 0, 'ra.rad')
                elif system == 'GALACTIC':
                    physical_types[axis-1] = _PHYSICAL_TYPE['GLON']
                    object_components[axis-1] = (IDstring, 0, 'b.rad')
                elif system == 'AZEL':
                    # Latitutde is elevation, long is azimuth
                    physical_types[axis-1] = 'pos.az.azi'
                    object_components[axis-1] = (IDstring, 0, 'azi.rad')

            elif islat:
                if system in _RADECSYSTEMS:
                    physical_types[axis-1] = _PHYSICAL_TYPE['DEC']
                    object_components[axis-1] = (IDstring, 1, 'dec.rad')
                elif system == 'GALACTIC':
                    physical_types[axis-1] = _PHYSICAL_TYPE['GLAT']
                    object_components[axis-1] = (IDstring, 1, 'l.rad')
                elif system == 'AZEL':
                    physical_types[axis-1] = 'pos.az.alt'
                    object_components[axis-1] = (IDstring, 1, 'alt.rad')

            if IDstring not in object_classes:
                if system == 'AZEL':

                    # # Need to handle the earth location.
                    # earthlon = frame.get('ObsLon')
                    # earthlat = frame.get('ObsLat')
                    # earthheight = frame.get('ObsAlt')
                    # # Need a parser to convert these strings to
                    # # floats
                    # earthlocation = coords.EarthLocation(
                    #     lon=earthlon, lat=earthlat, height=earthheight)
                    # # Ensure time is in correct system??
                    # epoch = frame.get('Epoch({})'.format(axis))
                    # epoch = _get_ast_epochequinox(epoch)
                    # skyclass = ('astropy.coordinates.AltAz',
                    #             (),
                    #             {'obstime':epoch,
                    #              'location': earthlocation})
                    raise NotImplementedError(system)
                elif system in ('ECLIPTIC', 'GEOCENTRIC', 'HELIOECLIPTIC',
                                'SUPERGALACTIC', 'J2000.0' 'UNKNOWN'):
                    raise NotImplementedError(system)
                else:
                    # Add the system to objctclass dict if its not already
                    # there.

                    # This assumes that no one is creating a frameset with
                    # 2 or more instances of the same skyframe -- e.g. a
                    # 4-D array with FK5 ra,dec vs a different FK5
                    # ra,dec. Not sure how to check for that, unless I go
                    # back to the decompose into component frames
                    # aporach. Or could check first how many there are,
                    # and handle them differently. I.e. iterate through
                    # all axes first?
                    epoch = frame.get('Epoch({})'.format(axis))
                    epoch = _get_ast_epochequinox(epoch)
                    equinox = frame.get('Equinox({})'.format(axis))
                    equinox = _get_ast_epochequinox(equinox)
                    skyclass = ('astropy.coordinates.SkyCoord',
                                (),
                                {'frame': _ASTROPY_FRAME_NAMES[system],
                                 'unit': u.rad,
                                 'equinox': equinox,
                                 'obstime': epoch,
                             })
                    object_classes[IDstring] = skyclass

        else:
            label = IDstring
            if frame.Nout > 1:
                label+= '-{}'.format(axis-1)
            unit = frame.get('InternalUnit({})'.format(axis))
            object_components[axis-1] = (label, 0, 'value')
            object_classes[label] = ('astropy.units.Quantity',
                                     (),
                                     {'unit':unit})
            system = frame.get('System({})'.format(axis))
            system = _AST_ALIASES.get(system, system)
            if system in _PHYSICAL_TYPE:
                physical_types[axis-1] = _PHYSICAL_TYPE[system]
            else:
                axislabel = frame.get('Label({})'.format(axis))
                if axislabel.strip() == '':
                    axislabel = '{}'.format(axis)
                physical_types[axis-1] = 'custom:{}-{}.{}'.format(
                    domain, system, axislabel)


    return physical_types, object_components, object_classes


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


def _get_ast_epochequinox(epoch):
    """
    Format an Ast epoch/equinox attribute for astropy.

    From SUN/211: if < 1984.0 its in Besselian, if greater its in
    Julian. If blank string return None.

    Doesn't yet handle weird values sensibly (should raise an error?)
    """
    if epoch == '':
        epoch = None
    elif float(epoch) < 1984.0:
        epoch = 'B' + str(epoch)
    elif epoch:
        epoch = 'J' + str(epoch)

    return epoch
