import geopandas as gpd
import numpy,copy
from shapely.geometry import Polygon
from fiona.crs import from_epsg
from osgeo import gdalnumeric
from methods import data_handle as dhdl

class VectorTools:

    @staticmethod
    def geometries(lon,lat,xul,yul,xur,yur,xlr,ylr,xll,yll,xuu,yuu):
        x=[numpy.ndarray.flatten(lon) + xul,numpy.ndarray.flatten(lon) + xur,numpy.ndarray.flatten(lon) + xlr,numpy.ndarray.flatten(lon) + xll,numpy.ndarray.flatten(lon) + xuu]
        y=[numpy.ndarray.flatten(lat) + yul,numpy.ndarray.flatten(lat) + yur,numpy.ndarray.flatten(lat) + ylr,numpy.ndarray.flatten(lat) + yll,numpy.ndarray.flatten(lat) + yuu]
        ulpts=[x[0],y[0]]
        urpts=[x[1],y[1]]
        lrpts = [x[2], y[2]]
        llpts = [x[3], y[3]]
        return x,y,ulpts,urpts,lrpts,llpts

    @staticmethod
    #make sure input grids are regular and square/rectangular shape for each pixel!
    def generate_polygon_from_lolat(lon,lat,position="ct"):
        dx=lon[:,0]-lon[:,1]#TODO all kinds of grids and vectors
        dy =lat[0,:] - lat[1,:]#TODO all kinds of grids and vectors
        a=numpy.unique(dx)
        b=numpy.unique(dy)
        if not numpy.allclose(numpy.abs(a),numpy.abs(b)):
            print("non squareshape rasterpixels or swapped dimensions!")
            print(a,"dx")
            print(b,"dy")
        numbers=numpy.abs([a,b])/2.0
        print(numbers)
        if position=="ct":
            #lon, lat, xul, yul, xur, yur, xlr, ylr, xll, yll, xuu, yuu
            lo,la,ulpts,urpts,lrpts,llpts=VectorTools.geometries(lon,lat,-numbers[0],numbers[1],numbers[0],numbers[1],numbers[0],-numbers[1],-numbers[0],-numbers[1],-numbers[0],numbers[1])
        if position=="ul":
            # lon, lat, xul, yul, xur, yur, xlr, ylr, xll, yll, xuu, yuu
            lo, la,ulpts,urpts,lrpts,llpts= VectorTools.geometries(lon, lat, 0, 0, 2*numbers[0], 0, 2*numbers[0],
                                            -2*numbers[1], 0, -2*numbers[1], 0, 0)
        if position=="ll":
            # lon, lat, xul, yul, xur, yur, xlr, ylr, xll, yll, xuu, yuu
            lo, la,ulpts,urpts,lrpts,llpts= VectorTools.geometries(lon, lat, 0, 2*numbers[1], 2*numbers[0], 2*numbers[1], 2*numbers[0],
                                            0, 0, 0, 0, 2*numbers[1])
        if position=="lr":
            # lon, lat, xul, yul, xur, yur, xlr, ylr, xll, yll, xuu, yuu
            lo, la,ulpts,urpts,lrpts,llpts = VectorTools.geometries(lon, lat, -2*numbers[0], 2*numbers[1], 0, 2*numbers[1], 0,
                                            0, -2*numbers[0], 0, -2*numbers[0], 2*numbers[1])
        if position=="ur":
            # lon, lat, xul, yul, xur, yur, xlr, ylr, xll, yll, xuu, yuu
            lo, la, ulpts,urpts,lrpts,llpts = VectorTools.geometries(lon, lat, -2*numbers[0], 0, 0, 0, 0,
                                            -2*numbers[1], -2*numbers[0], -2*numbers[1], -2*numbers[0], 0)
        pts=[ulpts,urpts,lrpts,llpts,ulpts]
        """else:
            print("only ul,ll,ur,lr and ct allowed as position input")
            print("please see the sketch below for a mxm Subpixel Matrix!")
            print('')
            print("ul---ur")
            print("|----|")
            print("|-ct-|")
            print("|----|")
            print("ll---lr")"""
        return lo,la,pts
    #super slow and clunky due to geopandas as array proc.
    @staticmethod
    def genartorlist(pts):
        ptt=numpy.asarray(pts)
        aps=ptt.shape
        gh=[]
        for j in numpy.arange(0,aps[2],1):
            print(j)
            a=Polygon(list(zip(ptt[:,0,j],ptt[:,1,j])))
            gh.append(a)
        dframe=gpd.GeoDataFrame(index=numpy.arange(0,len(gh),1),crs='epsg:4326',geometry=gh)
        return gh,dframe

    @staticmethod
    def crea_em_inv_data_tno(datafile):
        objj=dhdl.FileIoHDF(datafile)
        objj.load_key_attribs()
        lat=numpy.flip(numpy.asarray(objj.lat))
        lon=numpy.asarray(objj.lon)
        long,latt=numpy.meshgrid(lon,lat)
        lo,la,pts=VectorTools.generate_polygon_from_lolat(long,latt,position="ct")
        gh,dframe=VectorTools.genartorlist(pts)
        return dframe,objj

    @staticmethod
    #for TNO Inventory data #3ddata
    def turnintaroud(object):
        return numpy.flip(numpy.rot90(object,2,[0,2]))

    @staticmethod
    #wrting one CAMS Dataset takes 15min here
    def appenddata(data,gpdframe):
        shp=data.shape
        gdf=copy.deepcopy(gpdframe)
        if shp[0]==12:
            for j in numpy.arange(1,13,1):
                gdf[str(j)]=data[j-1,:,:].flatten()
        else:
            print('Sth is wrong!')
        return gdf
