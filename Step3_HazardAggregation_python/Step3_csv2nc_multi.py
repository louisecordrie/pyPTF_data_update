"""
=================================
KOS_BODRUM_MULTI_HAZARD_CSV2NC.PY
=================================

THIS SCRIPT TRANSLATES THE INPUT CSV WITH KOS-BODRUM DATA INTO A
NETCDF FILE WHICH IS COMPLIANT WITH CF-1.8 CONVENTIONS.

DATA IN THE CSV IS ORDERED BY:
    1, 2) LON, LAT: LONGITUDE AND LATITUDE COORDINATES OF POIs (WGS84);
    3) PROB: THE UNCONDITIONAL EXCEEDANCE PROBABILITY THRESHOLD OF THE HAZARD CURVES FOR
             AN EXPOSURE TIME OF 5 YEARS;

FOR EACH TRIPLE (LON, LAT, PROB) THE CSV CONTAINS THE CORRESPONDING:
    4) WATER HIGHT (WH) VALUE.

WH VALUES IN THE CSV ARE IN METERS AND NEED TO BE DIVIDED BY 1000.

-999 VALUES REPRESENT MISSING VALUES.
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc4
import time


def csv2nc(PATH, EVENT_ID):
	#print('PATH:   ', PATH)
	print('Converting csv to nc')
	# INPUT CSV FILE NAME
	FILE_DATA_CSV = 'Step3_HazardCurves_multi.csv'
	# INPUT CSV FILE COLUMNS
	INPUT_CSV_COLS = ['lon', 'lat', 'prob', 'wh']

	# OUTPUT NETCDF FILE NAME
	NETCDF_FILE = 'Step3_HazardCurves_multi.nc'

	# =====================
	# READ KOS-BODRUM CSV DATA
	# =====================

	# DATA FRAME WITH CSV DATA
	df = pd.read_csv(PATH+'/'+FILE_DATA_CSV, comment='#', usecols=INPUT_CSV_COLS)

	# LIST WITH UNIQUE PAIRS (LONGITUDE, LATITUDE), i.e. THE POIs
	lonlat = df.loc[:,['lon','lat']].drop_duplicates().values

	# SEPARATE LISTS WITH LONGITUDES AND LATITUDES
	lon = lonlat[:,0]
	lat = lonlat[:,1]

	# LIST WITH UNIQUE PROBABILITIES
	prob = df.loc[:,['prob']].drop_duplicates().values
	prob = prob[:,0]

	# WH VALUES
	wh = df.loc[:,['wh']].values
	wh.shape = (len(lonlat), len(prob))

	# =========================
	# CREATE THE NETCDF DATASET
	# =========================

	ds = nc4.Dataset(PATH+'/'+NETCDF_FILE, 'w', format='NETCDF4')

	# DEFINE NETCDF DIMENSIONS
	ds.createDimension('exceedance_probability_50yr', len(prob))
	ds.createDimension('target_point', len(lon))
	ds.createDimension('time', None)

	# DEFINE AND POPULATE NETCDF VARIABLES WITH CSV DATA
	target_point = ds.createVariable('target_point', 'i4', 'target_point')
	longitude = ds.createVariable('lon', 'f4', 'target_point')
	latitude = ds.createVariable('lat', 'f4', 'target_point')
	probability = ds.createVariable('exceedance_probability_50yr', 'f4', 'exceedance_probability_50yr')
	waveh = ds.createVariable('wave_height_1m', 'f4', ('target_point', 'exceedance_probability_50yr'))

	# AVERAGE RETURN PERIOD
	import math
	average_return_period = ds.createVariable('average_return_period', 'f4', ('exceedance_probability_50yr'))
	average_return_period[:] = np.array([round(-50/math.log(1-p+1e-100)) for p in prob])
	average_return_period.standard_name = "average_return_period"
	average_return_period.long_name = "Average return period"
	average_return_period.comment = "Defined as a function of exceedance_probability_50yr"
	average_return_period.units = "year"

	target_point[:] = np.array([i for i in range(len(lonlat))])
	longitude[:] = lon
	latitude[:] = lat
	probability[:] = prob
	waveh[:,:] = wh

	# =====================
	# SET NETCDF ATTRIBUTES
	# =====================

	# GLOBAL ATTRIBUTES
	ds.title = "Probabilistic Tsunami Forecast related to the Earthquake of " + EVENT_ID
	ds.institution = "Istituto Nazionale di Geofisica e Vulcanologia (INGV)"
	ds.references = "http://www.ingv.it"
	ds.history = "TO BE DEFINED"
	ds.authors = "TO BE DEFINED"
	ds.contact_person = "Enrico Baglione <enrico.baglione@ingv.it>; Jacopo Selva <jacopo.selva@ingv.it>"
	ds.Conventions = "CF-1.8"

	# VARIABLE ATTRIBUTES
	target_point.long_name = "Point of Interest (PoI)"
	target_point.standard_name = "target_point"
	target_point.axis = "T"

	longitude.long_name = "Longitude coordinate of the PoI (WGS84)"
	longitude.standard_name = "longitude"
	longitude.units = "degrees_east"

	latitude.long_name = "Latitude coordinate of the PoI (WGS84)"
	latitude.standard_name = "latitude"
	latitude.units = "degrees_north"

	probability.long_name = "Unconditional probability of exceedence for an exposure time of 50 years"
	probability.standard_name = "exceedance_probability_50yr"
	probability.axis = "Y"

	waveh.long_name = "Near-coast wave height at 1m depth"
	waveh.standard_name = "wave_height_1m"
	waveh.units = "meter"
	waveh.scale_factor = 1.e-3
	waveh.add_offset = 0
	waveh.missing_value = -999
	waveh.axis = "X"

	# CLOSE DATASET
	ds.close()


if __name__ == '__main__':
	start_time = time.time()
	#program_path = os.getcwd()+'/'
	#code_path = program_path
	#risale di una directory e ridefinisce il percorso 
	#os.chdir('..')
	#program_path = os.getcwd()+'/'
	#print('directory di lavoro: ', program_path)
	# IF THE NUMBER OF ARGUMENTS IS CORRECT
	if len(sys.argv) == 3:
		# GET FIRST ARGUMENT WITH INPUT CSV NAME
		INPUT_PATH =  sys.argv[1]
		# GET SECOND ARGUMENT WITH OUTPUT CSV NAME
		EVENT_NAME = sys.argv[2]
		# START CONVERSION
		csv2nc(INPUT_PATH, EVENT_NAME)
		end_time = time.time()
		print('il codice ha impiegato ', end_time-start_time, 'seconds')
		# EXIT WITH SUCCESS CODE
		sys.exit(0)
	else:
		# PRINT SCRIPT DESCRIPTION
		print(__file__.upper(), "-- converts input CSV with HAZARD CURVES results into the CSV INPUT NetCDF format.")
		# PRINT USAGE
		print("Usage:", __file__, "[INPUT_PATH]" "[EVENT_ID]")
		# EXIT WITH ERROR CODE
		sys.exit(1)

