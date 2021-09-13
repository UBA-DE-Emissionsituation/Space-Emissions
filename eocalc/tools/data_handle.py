import copy
import numpy
from osgeo import gdal
import netCDF4 as Ncf
import h5py

#For Tropomi Module
class tropomi_l2_reader:
    @staticmethod
    def open(filename):
        hdf = FileIoHDF(filename)
        hdf.load_key_attribs()
        d2arrays = {}
        d2arraysshape = {}
        for j in enumerate(hdf.kessy):
            try:
                data = numpy.squeeze(hdf.hdf5[j[1]])
                shp = data.shape
                if len(data.shape) == 2:
                    d2arrays.update({j[1]: data})
                    d2arraysshape.update({j[1]: shp})
            except(AttributeError):
                print(numpy.shape(hdf.hdf5[j[1]]))
        return d2arrays, d2arraysshape

    @staticmethod
    def grab_2d_data_arrays(d2arrays, d2arrayshape):
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
    def generic_ds(dataset):
        arr, shpp = tropomi_l2_reader.open(dataset)
        dname = dataset.split('/')[-1]  # TODO fix winproblem
        dname = dname.split('.')[0]
        outdir = dataset.split('/')[:-1]  # TODO get rid of HACK with os module
        outdir = '/'.join(outdir)
        bandnames, dataarr = tropomi_l2_reader.grab_2d_data_arrays(arr, shpp)
        dataarr = numpy.asarray(dataarr)
        print(outdir + '/' + dname + "_dstack")
        GenOutput.write(dataarr, 'ENVI', outdir + '/' + dname + "_dstack")
        header_manipulation.append_to_header(outdir + '/' + dname + "_dstack" + '.hdr', ["band names"], [bandnames],
                                             [1, 2, 3, 4])
        return

#For Base Module
class header_manipulation:
    @staticmethod
    def replace_char(string, keyword):
        string = string.replace(']', '}\n')
        string = string.replace('[', '{')
        keyword = keyword.replace('/', '_')
        keyword = keyword.replace('_', ' ')
        keyword = keyword.strip()
        string = keyword + '=' + string
        return string

    @staticmethod
    def to_str_params(vector, keyword):
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
    def crea_hdr_params(keys, values):
        lll = []
        for j in enumerate(keys):
            keyy = j[1]
            val = values[j[0]]
            vecstring = header_manipulation.to_str_params(val, keyy)
            lll.append(vecstring)
        return lll

    @staticmethod
    def construct_crs_string(xres, yres, ellipsoid, ulx, uly, projection, ulpx=1, ulpy=1, projectionzone='33',
                             projectionhemisph='N'):
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
    def append_header_meta(headerfile, metadatalist):
        if '.hdr' not in headerfile:
            print('This is no headerfile we are unable to open it.')
        try:
            with open(headerfile, 'a') as file:
                file.writelines(metadatalist)
        except OSError:
            print('file IO Error!')

    @staticmethod  # call this function here!!!
    def append_to_header(headerfile, keys, values, geoinfolist):
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
    @staticmethod
    def write(array, drivername, out):
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
            dst_ds = None#TODO Add Geoinfo and Metadata
        elif len(h) == 2:
            h = numpy.shape(array)
            dst_ds = driver.Create(out, h[1], h[0], 1, gdal.GDT_Float32)
            dst_ds.GetRasterBand(1).WriteArray(array)
            dst_ds = None#TODO Add Geoinfo and Metadata
        else:
            print('Strange Datashape')
            print('Does your numpy 2d/3d array sit at the correct argument position?')
            raise ValueError


# Read HDF5 Everything (with all type of data), For Base Module
class FileIoHDF:
    def __init__(self, datafile):
        self.datafile = datafile
        self.hdf5 = None
        self.kessy = None
        try:
            self.hdf5 = h5py.File(self.datafile, 'r')
        except OSError:
            print('Not a valid HDF datafile!')

    @staticmethod
    def feil(hdfobject):
        dset = []
        gp = []
        dt = []
        hdfobject.visit(lambda item: dset.append(item) if (type(hdfobject[item]) == h5py._hl.dataset.Dataset) else (
            gp.append(item) if (type(hdfobject[item]) == h5py._hl.group.Group) else (
                dt.append(item) if (type(hdfobject[item]) == h5py._hl.datatype.Datatype) else print(item))))
        return dset, gp, dt

    @staticmethod
    def methode(a,b):
        if isinstance(b,h5py.Dataset):
            return a#dataset

    @staticmethod
    def extracto(h5objekt):
        h5objekt.visititems(FileIoHDF.methode)

    @staticmethod
    def aff(name):
        if isinstance(name, h5py.Dataset):
            print(name)

    def load_key_attribs(self):
        self.kessy,b,c = FileIoHDF.feil(self.hdf5)
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
    def __init__(self, datafile):
        self.datafile = datafile
        self.net4 = None
        self.kessy = None
        try:
            self.net4 = Ncf.Dataset(self.datafile)
            self.kessy = self.net4.variables.keys()
        except OSError:
            print('Not a valid NCDF datafile!')

    def load_key_attribs(self):
        for i in self.kessy:
            try:
                setattr(self, i.replace(' ', '_'), self.net4.variables[i])
            except AttributeError:
                print('No Attribute found setting setattr to None for now')# TODO still missing None
            except ValueError:
                print('No Value found setting setattr to None for now')# TODO still missing None
