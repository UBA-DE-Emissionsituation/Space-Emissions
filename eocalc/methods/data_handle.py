import copy
import os
import affine
import numpy
from osgeo import gdal
import netCDF4 as Ncf
import h5py
import glob
import rasterio as rio
from rasterio.mask import mask
from rasterio import Affine
from pyproj import CRS
import geopandas as gpd
from scipy.interpolate import griddata
import pykrige


class MultiTemp:
    def __init__(self, folder: str, pattern1="NO2",pattern2=".zip"):
        """
        Grab all hdf files in one folder and return the full paths
        :param folder: Path to the Data
        :param pattern1: First part of the filename pattern to search for
        :param pattern2: Last part of the filename pattern to search for
        """
        os.chdir(folder)
        srchstr=folder+'/**/*'+pattern1+"*"+pattern2
        self.filelist = glob.glob(srchstr, recursive=True)

    def s5pdatanames(self):
        """
        Method for creating a dictionary of open S5P files
        :return: Dictionary of Open S5P L2 Files Keys are the filenames Values are the open files
        :example: files['NRTI_L2__NO2____20210527T004058_20210527T004558_18750_01_010400'].hdf
        """
        self.files={}
        for j in enumerate(self.filelist):
            self.files.update({os.path.basename(j[1])[4:67]:tropomi_l2_reader(j[1])})

    def s5qafilt(self,qavalmin:float,attr_list=None):
        for j in enumerate(self.files.keys()):
            self.files[j[1]].get_qa_qc_select(qavalmin,attr_list)

    def closeall(self):
        """
        Close all open datafiles of a MultiTemp. files Object for data safety
        :return:
        """
        for j in enumerate(self.files.keys()):
            self.files[j[1]].hdf.hdf5.close()






class d2interpolations:
    @staticmethod
    def krig_pykrige(x, y, z, xi, yi, method="ordinary"):
        if method == "ordinary":
            pass
        if method == "universal":
            pass
        return None

    @staticmethod
    def scp_gdd_interp(x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray, xi: numpy.ndarray, yi: numpy.ndarray,
                       method="cubic") -> numpy.ndarray:
        """

        :param x:1d Vector of x values of the data points
        :param y:1d Vector of y values of the data points
        :param z: Values of the points
        :param xi:1d Vector of X values for interpolation
        :param yi: 1d Vector of Y values for interpolation
        :param method:
        :return: zint
        """
        grid_x, grid_y = numpy.mgrid[xi, yi]  # interpolate at these points
        points = numpy.array([x, y])
        if method == "cubic":
            zint = griddata(points, z, (grid_x, grid_y), method='cubic')
        if method == "nearest":
            zint = griddata(points, z, (grid_x, grid_y), method='nearest')
        if method == "linear":
            zint = griddata(points, z, (grid_x, grid_y), method='linear')
        return zint


class raster_clip:
    """
    Class for Raster clipping operations using shapefile geometries.
    """

    @staticmethod
    def clipping(infile: str, shape: str, key: str, value: str, fill=numpy.nan):
        """
        Method for clipping a georeferenced input raster with a shapefile
        :param infile: georeferenced input raster
        :param shape: shapefile of your choice, for countries e.g. use FOSS "Natural Earth" shapefiles
        :param key: key that identifies your selection property of choice
        :param value: value which you would like to use to select your clipping shapefile
        :param fill: fill value for the clipped raster dataset; defaults to numpy.nan
        :return: saves the clipped datafile to the same location as infile with "clip_value" extension
        """
        data = rio.open(infile)
        shapedata = gpd.read_file(shape)
        gpframe = shapedata[shapedata[key] == value]
        gpframe = gpframe.to_crs(crs=data.crs.data)
        geom = gpframe['geometry']
        clip_img, clip_trafo = mask(dataset=data, shapes=geom, nodata=fill, crop=True)
        clip_meta = data.meta.copy()
        clip_meta.update({"height": clip_img.shape[1], "width": clip_img.shape[2], "transform": clip_trafo})
        with rio.open(infile + "_clip_" + str(value), 'w', **clip_meta) as destination:
            destination.write(clip_img)


class raster_save:
    """
    Class for writing rasters based on the Rasterio Library
    #TODO need to set RPC model for data from LAT/LON COORDS
    """

    @staticmethod
    def getcrs_and_trafo_from_file(infile: str):
        """
        Grab Coordinate data from infile
        :param infile: path to georeferenced inputfile
        :return: Epsg:Code and Affine Transformation Matrix
        """
        fil = gdal.Open(infile)
        try:
            geotrafo = fil.GetGeoTransform()
            matrix = Affine.from_gdal(*geotrafo)
        except AttributeError:
            print("No Transformation Info!")
            matrix = None
        try:
            a = fil.GetProjection()
            ll = CRS.from_wkt(a)
            auth = ll.to_authority()
            epsgcode = auth[0] + ':' + auth[1]
        except AttributeError:
            print("No CRS Info!")
            epsgcode = None
        return epsgcode, matrix

    @staticmethod
    def calculate_affine_North_up(xlonUL: float, ylatUL: float, XSize: float,
                                  YSize: float) -> affine.Affine:  # TODO Watch out for the right Coordinates TOPLEFT OF TOPLEFT-CORNER!
        """
        Calculate the Affine transform Matrix for the dataset
        for non-rotated N-Up data.
        This can be used if you cannot parse the CRS data from the original files,
        or if you need to write the metadata "manually"
        :param xlonUL:Upper Left Longitude Corner Coordinate
        :param ylatUL:Uper Left Latitued Corner Coordinate
        :param XSize:XPixelSize
        :param YSize:YPixelSize
        :return: the affine matrix you need for writing georeferenced data with rasterio
        """
        affin = (xlonUL, XSize, 0.0, ylatUL, 0.0, YSize * -1.0)
        # needs factor:-1.0 in YSize otherwise upside Down as Y Direction is downward!
        matrix = Affine.from_gdal(*affin)
        return matrix

    @staticmethod
    def write_data(numpyarray: numpy.ndarray, outputfile: str, driverr="ENVI", epsg=None, affinemat=None):
        """
        Write your numpy 2d or 3d array either unreferenced or georeferenced

        :param numpyarray: Numpy Array with Geodata, can either be 2d or 3d Numpy-ndarray
        :param outputfile: Path to the outputfile you would like to write to
        :param driverr: write the output file with the given driver, default: "ENVI"
        :param epsg: epsg string of your Coordinate System of choice, if None then write unreferenced image
        :param affinemat: affine transformation matrix for forward data transformation, if None then write unreferenced image
        :return: Nothing just write the data
        """
        shp = numpyarray.shape
        if (epsg == None) and (affinemat == None):
            if len(shp) == 2:
                print('2d without coords')
                deneo = rio.open(outputfile, 'w', driver=driverr, height=shp[0], width=shp[1], count=1,
                                 dtype=numpyarray.dtype)
                deneo.write(numpyarray, 1)
                deneo.close()
            if len(shp) == 3:
                print('3d without coords')
                deneo = rio.open(outputfile, 'w', driver=driverr, height=shp[1], width=shp[2], count=shp[0],
                                 dtype=numpyarray.dtype)
                for j in numpy.arange(1, shp[0] + 1, 1):
                    deneo.write(numpyarray[j - 1, :, :], j)
                    deneo.close()
        else:
            if len(shp) == 2:
                print('2d with coords')
                deneo = rio.open(outputfile, 'w', driver=driverr, height=shp[0], width=shp[1], count=1,
                                 dtype=numpyarray.dtype, crs=epsg, transform=affinemat)
                deneo.write(numpyarray, 1)
                deneo.close()
            if len(shp) == 3:
                print('3d with coords')
                deneo = rio.open(outputfile, 'w', driver=driverr, height=shp[1], width=shp[2], count=shp[0],
                                 dtype=numpyarray.dtype, crs=epsg, transform=affinemat)
                for j in numpy.arange(1, shp[0] + 1, 1):
                    deneo.write(numpyarray[j - 1, :, :], j)
                deneo.close()


# For Tropomi Module
class tropomi_l2_reader:
    """
    A class for reading singlefile Tropomi_L2_Data
    """

    def __init__(self, filename: str):
        """
        Entry point of the tropomi_l2_reader class
        :param filename: Path to the L2 data product file
        """
        self.hdf = FileIoHDF(filename)
        self.hdf.load_key_attribs()
        self.qafilt = None

    def read_2darraysfull(self):#TODO Fix routine for full raw read
        d2arrays = {}
        d2arraysshape = {}
        for j in enumerate(self.hdf.kessy):
            try:
                data = numpy.squeeze(self.hdf.hdf5[j[1]])
                shp = data.shape
                if len(data.shape) == 2:
                    d2arrays.update({j[1]: data})
                    d2arraysshape.update({j[1]: shp})
            except(AttributeError):
                print(numpy.shape(self.hdf.hdf5[j[1]]))
        self.d2arrays = d2arrays
        self.d2arraysshape = d2arraysshape

    def calculate_long_distort(self, discut=50):
        """
        Calculate "Longitudal Distortion"
        :param discut: Cutoff over which Distortion will be cut defaults to Median (50%)
        :return: sets the idx_distort Attribute containing the valid indices below Median-Distortion
        """
        long = numpy.squeeze(self.hdf.PRODUCT_longitude)
        ll = numpy.abs(long - numpy.roll(long, 1, axis=1))
        colsum = numpy.sum(ll, axis=1)
        longdiffcutoff = numpy.percentile(colsum, discut)
        self.idx_distort = numpy.where(colsum <= longdiffcutoff)

    def get_qa_qc_select(self, cutoff=50,paramlist=[]):
        """
        Method for
        :param cutoff: Cutoff value for the L2A data quality
        :return: sets the qaidx attribute to contain those indices (ge) to the qa threshold plus sets a masked array
        """
        qa = numpy.squeeze(self.hdf.PRODUCT_qa_value)
        self.qaidx = numpy.where(qa >= cutoff)
        self.maskarr = numpy.ma.masked_less(qa, cutoff)
        if len(paramlist) != 0:
            for a in enumerate(paramlist):
                try:
                    self.__setattr__(a[1],numpy.squeeze(self.hdf.__getattribute__(a[1]))[self.qaidx])
                except AttributeError:
                    print("Attribute error ", a[1], " not found or oddly shaped!")

    @staticmethod
    def grab_2d_data_arrays(d2arrays: dict, d2arrayshape: dict) -> (list, list):
        """
        grab all of the 2d arrays from the hdf file and return their names
        :param d2arrays: dictionary of the data arrays
        :param d2arrayshape: dictionary of the data array shapes
        :return: bandnames list and list of the corresponding 2d arrays
        """
        val = numpy.asarray(list(d2arrayshape.values()))
        arr, counts = numpy.unique(val[:, 0], return_counts=True)
        dim0 = arr[numpy.argmax(counts)]
        arr, counts = numpy.unique(val[:, 1], return_counts=True)
        dim1 = arr[numpy.argmax(counts)]
        dimm = [dim0, dim1]
        bandnames = []
        dataarr = []
        for j in enumerate(d2arrays.keys()):
            t = d2arrays[j[1]].shape
            if (t[0] == dimm[0]) and (t[1] == dimm[1]):
                dataarr.append(d2arrays[j[1]])
                bandnames.append(j[1])
        return bandnames, dataarr

    @staticmethod
    def generic_ds(dataset: str):
        """
        Generate a BSQ file from a TROPOMI L2 Product containing only the 2d matrix vales of the data
        :param dataset: Path to the Tropomi L2 file
        :return: writes a dataset with the data in original geometry
        """
        dset = tropomi_l2_reader(dataset)
        dname = dataset.split('/')[-1]  # TODO fix winproblem
        dname = dname.split('.')[0]
        outdir = dataset.split('/')[:-1]  # TODO get rid of HACK with os module
        outdir = '/'.join(outdir)
        bandnames, dataarr = tropomi_l2_reader.grab_2d_data_arrays(dset.d2arrays, dset.d2arraysshape)
        dataarr = numpy.asarray(dataarr)
        print(outdir + '/' + dname + "_dstack")
        GenOutput.write(dataarr, 'ENVI', outdir + '/' + dname + "_dstack")
        header_manipulation.append_to_header(outdir + '/' + dname + "_dstack" + '.hdr', ["band names"], [bandnames],
                                             [1, 2, 3, 4])
        return


# For Base Module
class header_manipulation:
    """
    Class for the manipulation of .hdr files that are associated with BSQ files
    """

    @staticmethod
    def replace_char(string: str, keyword: str) -> str:
        """
        Patch together a metadata descriptor to be in correct format to be put in .hdr file
        :param string: String containing the data values
        :param keyword: String containing the metadata keyword
        :return: full string that can be written to a .hdr file
        """
        string = string.replace(']', '}\n')
        string = string.replace('[', '{')
        keyword = keyword.replace('/', '_')
        keyword = keyword.replace('_', ' ')
        keyword = keyword.strip()
        string = keyword + '=' + string
        return string

    @staticmethod
    def to_str_params(vector, keyword: str) -> str:
        """
        Convert lists or numpy.ndarrays to a formated string that can be written to an .hdr file.
        :param vector: data vector (either list or 1d numpy.ndarray)
        :param keyword: metadata keyword e.g. 'wavelength'
        :return: Formatted string that can be written to a .hdr file
        """
        if type(vector) == list:
            vector = str(vector)
        elif type(vector) == numpy.ndarray:
            if len(numpy.shape) > 1:
                print("Array Dim >1 not supported!")
                raise ValueError
            vector = str(list(vector))
        vecstring = header_manipulation.replace_char(vector, keyword)
        return vecstring

    @staticmethod
    def crea_hdr_params(keys: list, values: list):
        """
        Create header parameters (metadata)
        :param keys: list of metadata keywords
        :param values: list of array alike entries
        :return: list of metadata strings
        """
        lll = []
        for j in enumerate(keys):
            keyy = j[1]
            val = values[j[0]]
            vecstring = header_manipulation.to_str_params(val, keyy)
            lll.append(vecstring)
        return lll

    @staticmethod
    def construct_crs_string(xres: float, yres: float, ellipsoid: str, ulx: float, uly: float, projection: str, ulpx=1,
                             ulpy=1, projectionzone='33',
                             projectionhemisph='N'):
        """
        Construct a CRS String for BSQ associated .hdr files
        :param xres: X Resolution of the pixel
        :param yres: Y Resolution of the Pixel
        :param ellipsoid: Ellipsoid (in most cases "WGS84" anyway)
        :param ulx: X of UL UL Corner
        :param uly: Y of UL UL Corner
        :param projection: UTM Zone, if aplicable
        :param ulpx: X of UL UL Corner in Pixel Coordinates (mostly 0 if not offseted)
        :param ulpy: Y of UL UL Corner in Pixel Coordinates  (mostly 0 if not offseted)
        :param projectionzone: UTM Zone Number
        :param projectionhemisph: UTM Hemisphere (N or S)
        :return: constructed CRS string for your .hdr file
        """
        if projection == "Geographic Lat/Lon":
            return "map info={" + projection + ',' + str(ulpx) + ',' + str(ulpy) + ',' + str(ulx) + ',' + str(
                uly) + ',' + str(xres) + ',' + str(yres) + ',' + ellipsoid + '}\n'
        elif projection == "UTM":
            return "map info={" + projection + ',' + str(ulpx) + ',' + str(ulpy) + ',' + str(ulx) + ',' + str(
                uly) + ',' + str(xres) + ',' + str(yres) + ',' + str(projectionzone) + ',' + str(
                projectionhemisph) + '}\n'
        else:
            print("No Standard Projection Information available.")
            print("Try to choose either UTM or Geographic Lat/Lon.")
            print("Or you don't have any CRS with this file, then all is well.")
            return None

    @staticmethod
    def append_header_meta(headerfile: str, metadatalist: list):
        """

        :param headerfile: Name of the headerfile to append the metadata to
        :param metadatalist: Metadata list consisting of strings
        :return:
        """
        if '.hdr' not in headerfile:
            print('This is no headerfile we are unable to open it.')
        try:
            with open(headerfile, 'a') as file:
                file.writelines(metadatalist)
        except OSError:
            print('file IO Error!')

    @staticmethod  # call this function here!!!
    def append_to_header(headerfile, keys: list, values: list, geoinfolist):
        if len(geoinfolist) != 10:
            print("geoinfolist smaller than 10 entries assuming no relevant coordinate info")
            liste = header_manipulation.crea_hdr_params(keys, values)
            header_manipulation.append_header_meta(headerfile, liste)
        elif len(geoinfolist) == 10:
            liste = header_manipulation.crea_hdr_params(keys, values)
            geoinfo = header_manipulation.construct_crs_string(*geoinfolist)
            liste.append(geoinfo)
            header_manipulation.append_header_meta(headerfile, liste)
        else:
            print("something went wrong no data to append")


# Generic Output Functions For Base Module
class GenOutput:
    """File Output directly via gdal"""

    @staticmethod
    def write(array, drivername, out):
        """
        write dataset directly via python gdal bindings
        :param array:
        :param drivername: "ENVI", "GTIFF", etc.
        :param out: Name of the output file
        :return: Nothing
        """
        driver = gdal.GetDriverByName(drivername)
        h = numpy.shape(array)
        print(h)
        if len(h) == 3:
            dst_ds = driver.Create(out, h[2], h[1], h[0], gdal.GDT_Float32)
            ka = 1
            while ka <= h[0]:
                hhh = ka - 1
                dst_ds.GetRasterBand(ka).WriteArray(array[hhh, :, :])
                ka += 1
            dst_ds = None  # TODO Add Geoinfo and Metadata
        elif len(h) == 2:
            h = numpy.shape(array)
            dst_ds = driver.Create(out, h[1], h[0], 1, gdal.GDT_Float32)
            dst_ds.GetRasterBand(1).WriteArray(array)
            dst_ds = None  # TODO Add Geoinfo and Metadata
        else:
            print('Strange Datashape')
            print('Does your numpy 2d/3d array sit at the correct argument position?')
            raise ValueError


# Read HDF5 Everything (with all type of data), For Base Module
class FileIoHDF:
    """
    Class for reading HDF data via h5py
    """

    def __init__(self, datafile: str):
        """
        Open the Datafile
        :param datafile: Path to the Datafile
        """
        self.datafile = datafile
        self.hdf5 = None
        self.kessy = None
        try:
            self.hdf5 = h5py.File(self.datafile, 'r')
        except OSError:
            print('Not a valid HDF datafile!')

    @staticmethod
    def feil(hdfobject: h5py._hl.dataset.Dataset) -> (list, list, list):
        """
        Read everything from the Dataset and its Groups and Subgroups
        :param hdfobject: The open hdf5 file for reading
        :return: list of dataset, groups and datatypes
        """
        dset = []
        gp = []
        dt = []
        hdfobject.visit(lambda item: dset.append(item) if (type(hdfobject[item]) == h5py._hl.dataset.Dataset) else (
            gp.append(item) if (type(hdfobject[item]) == h5py._hl.group.Group) else (
                dt.append(item) if (type(hdfobject[item]) == h5py._hl.datatype.Datatype) else print(item))))
        return dset, gp, dt

    @staticmethod
    def methode(a, b):
        """
        See below
        :param a:
        :param b:
        :return:
        """
        if isinstance(b, h5py.Dataset):
            return a  # dataset

    @staticmethod
    def extracto(h5objekt):
        """
        Lambdafunction
        :param h5objekt:
        :return:
        """
        h5objekt.visititems(FileIoHDF.methode)

    @staticmethod
    def aff(name):
        if isinstance(name, h5py.Dataset):
            print(name)

    def load_key_attribs(self):
        """
        Attribute Constructor for appending all subdatasets, and groups to the open hdf5 object
        Constructor uses groups subgroups and dataset names as attributes and joins them with underscore
        :return:
        """
        self.kessy, b, c = FileIoHDF.feil(self.hdf5)
        for i in enumerate(self.kessy):
            try:
                ii = copy.deepcopy(i[1])
                ii = ii.replace('/', '_')
                ii = ii.replace(' ', '_')
                ii = ii.strip()
                setattr(self, ii, self.hdf5.__getitem__(i[1]))
            except AttributeError:
                print('No Attribute found setting setattr to None for now')
            except ValueError:
                print('No Value found setting setattr to None for now')


# Read netcdf with everything, for Base Module
class FileIoNC4:
    """Open Netcdf data"""

    def __init__(self, datafile: str):
        """
        Open netcdf file for reading
        :param datafile:
        """
        self.datafile = datafile
        self.net4 = None
        self.kessy = None
        try:
            self.net4 = Ncf.Dataset(self.datafile)
            self.kessy = self.net4.variables.keys()
        except OSError:
            print('Not a valid NCDF datafile!')

    def load_key_attribs(self):
        """
        Method for appending all variables of the netcdf to the file object
        :return:
        """
        for i in self.kessy:
            try:
                setattr(self, i.replace(' ', '_'), self.net4.variables[i])
            except AttributeError:
                print('No Attribute found setting setattr to None for now')  # TODO still missing None
            except ValueError:
                print('No Value found setting setattr to None for now')  # TODO still missing None
