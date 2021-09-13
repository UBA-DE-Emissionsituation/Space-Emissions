from methods import data_handle as dhdl
import numpy
import os
import urllib.request
import datetime
import calendar
from methods import equal_earth_raster_area as eera

URLL = 'https://d1qb6yzwaaq4he.cloudfront.net/airpollution/no2col/GOME_SCIAMACHY_GOME2ab_TroposNO2_v2.3_041996-092017_temis.nc'
LOCAL_DATA = 'data/testdata/themis_sat_series'
LOCAL_FILE = LOCAL_DATA+'/GOME_SCIAMACHY_GOME2ab_TroposNO2_v2.3_041996-092017_temis.nc'

#For Base Module
class temporal:
    @staticmethod
    def split_isostring_temis(timeobject) -> numpy.ndarray:
        L = []
        for j in enumerate(timeobject):
            datestring = str(j[1])[:4] + "-" + str(j[1])[4:6] + "-" + "15"
            L.append(datetime.date.fromisoformat(datestring))
        timearr=numpy.asarray(L)
        return timearr

    @staticmethod
    def month_days(timearr)->numpy.ndarray:
        L=[]
        for j in enumerate(timearr):
            L.append(calendar.monthrange(j[1].year,j[1].month))
        daysarr=numpy.asarray(L)
        return daysarr

    @staticmethod
    def time_delta(timearr,datetimeobj) -> int:
        timepoint = numpy.repeat(datetimeobj, len(timearr))
        timindex=numpy.argmin(numpy.abs(timearr-timepoint))
        return timindex


class TemisLongdata:

    @staticmethod
    def download_data(url: str, local_file: str):
        lf = os.path.abspath(local_file)
        if os.path.isfile(lf):
            print(local_file + ' Already exists no download necessary')
        else:
            print(local_file + ' Downloading please be patient')
            try:
                urllib.request.urlretrieve(url, lf)
            except(FileNotFoundError):
                print('invalid filename')
            return "file does not exist, downloading ..."

    @staticmethod
    def handle_time(startdate:str,stopdate:str,timevectemis:numpy.ndarray) -> (numpy.ndarray,numpy.ndarray,int,int):#TODO Handle THEMIS Time correctly
        try:
            startd=datetime.date.fromisoformat(startdate)
            stopd=datetime.date.fromisoformat(stopdate)
        except ValueError:
            print('a string has to be in str format: YYY-MM-DD')
            raise ValueError
        if (stopd-startd).days<0:
            start=stopd
            stop=startd
            print(start)
            print(stop)
        else:
            start=startd
            stop=stopd
        timevec=temporal.split_isostring_temis(timevectemis)
        dayvec=temporal.month_days(timevec)
        startidx=temporal.time_delta(timevec,start)
        stopidx=temporal.time_delta(timevec,stop)
        return timevec[startidx:stopidx],dayvec[startidx:stopidx],startidx,stopidx

    @staticmethod
    def open_nc(filename:str):
        try:
            themis=dhdl.FileIoNC4(filename)
            themis.load_key_attribs()
            return themis
        except(FileNotFoundError):
            print ('file not found!')
        except(OSError):
            print ('something went wrong with the file!')

    @staticmethod
    # per squaremeter
    def molecule():
        multiplicator = 10 ** 15  # gain for data to get molecules_per cm
        cminm = 10 ** 4
        varno2 = ((multiplicator * cminm) / (6.02214 * 10 ** 23)) * 46.001
        return varno2

    @staticmethod
    def grab_data(nc_file_object):
        try:
            data=numpy.asarray(nc_file_object.TroposNO2)
            lat=numpy.asarray(nc_file_object.lat)
            lon=numpy.asarray(nc_file_object.lon)
            time=numpy.asarray(nc_file_object.time)
        except(AttributeError):
            raise AttributeError
        rotated=numpy.rot90(data,k=2,axes=(1,0))#right direction
        rotated[rotated < 0]=numpy.nan# TODO Pixel based spline interpolation of NoData Values, or median of neighboring months without nan, or simple kriging of data
        lat=numpy.flip(lat)
        return rotated,lon,lat,time

    @classmethod
    def _run(cls,urlin:str,pathout:str,startdate:str,stopdate:str):
        cls._factor = TemisLongdata.molecule()
        #sqg = (squaremeter * gramss) / 1000000  # tonnen
        #output = data * sqg * 30
        TemisLongdata.download_data(urlin,pathout)
        nc_object=TemisLongdata.open_nc(pathout)
        cls._rotated,cls._lon,cls._lat,cls._time=TemisLongdata.grab_data(nc_object)
        cls._timevec,cls._dayvec,cls._startidx,cls._stopidx=TemisLongdata.handle_time(startdate,stopdate,cls._time)#TODO Fix the No of Days with the start and the stoptime
        cls._lonmat,cls._latmat=numpy.meshgrid(cls._lon,cls._lat)
        cls._area=eera.TransformEqEeasy.easytrafo(cls._lonmat,cls._latmat)#TODO in meters and, fix edges please!!!
        print(cls._startidx,cls._stopidx)
        print(cls._rotated.shape)
        cls._rotated=cls._rotated[cls._startidx:cls._stopidx,:,:]
        print(cls._rotated.shape,type(cls._rotated))
        cls._rotated=cls._rotated*(cls._factor)*cls._dayvec[:,1][:,numpy.newaxis,numpy.newaxis]#TODO potential error for the units (m/vs km vs g/t)
        print(os.path.basename(pathout))
        dhdl.GenOutput.write(cls._rotated,'ENVI',os.path.dirname(pathout)+"/emissions_ds_ts")
        output=numpy.nansum(cls._rotated,axis=0)#TODO Do clipping with Shapefile save as GIS Raster
        dhdl.GenOutput.write(output, 'ENVI', os.path.dirname(pathout) + "/emissions_ds_flat")
        nox_em=numpy.nansum(output)





