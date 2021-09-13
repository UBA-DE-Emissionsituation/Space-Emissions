import numpy
from pyproj import Proj

# For Base Module
class TransformEqEeasy:
    @staticmethod
    def easytrafo(long:numpy.ndarray,lat:numpy.ndarray) -> numpy.ndarray:
        toproj=Proj("epsg:8857")
        x,y=toproj(long,lat)#TODO check if lat is not UpsideDown in geographic Terms -> reason for fliped rasters in datasets
        deltax = numpy.abs(x - numpy.roll(x, 1,axis=1))
        deltay = numpy.abs(y - numpy.roll(y, 1,axis=0))
        area = deltax*deltay# TODO in meters otherwise km:/1000000
        return area

    @staticmethod
    def grid(startx:float, stopx:float, stepx:float, starty:float, stopy:float, stepy:float)->(numpy.ndarray,numpy.ndarray):
        x=numpy.arange(startx, stopx, stepx)
        y=numpy.flip(numpy.arange(starty, stopy, stepy))#TODO: Fix for Geographic_COORDINATES
        return numpy.meshgrid(x,y)