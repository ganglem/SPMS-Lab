import sys
import pandas as pd
import numpy as np
from numpy.random import random_sample
from scipy.special import lambertw
import math
from haversine import haversine
from datetime import datetime

assert (sys.version_info[0] ==
        3), "This is a Python3.X code and you probably are using Python2.X"


# constants      
RADIANT_TO_KM_CONSTANT = 6371.0088
EPSILON = 1.6/0.05
#EPSILON = 16.0/0.05


def add_vector_to_pos(original_lat, original_lon, distance, angle):
    """
    Add the generated noise directly on the gps coordinates.
    """
    ang_distance = distance / RADIANT_TO_KM_CONSTANT
    lat1 = rad_of_deg(original_lat)
    lon1 = rad_of_deg(original_lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(ang_distance) +
                     math.cos(lat1) * math.sin(ang_distance) * math.cos(angle))
    lon2 = lon1 + math.atan2(
        math.sin(angle) * math.sin(ang_distance) * math.cos(lat1),
        math.cos(ang_distance) - math.sin(lat1) * math.sin(lat2))
    lon2 = (lon2 + 3 * math.pi) % (2 * math.pi) - \
        math.pi  # normalise to -180..+180
    return deg_of_rad(lat2), deg_of_rad(lon2)


def rad_of_deg(ang): 
    """
    Useful for the add_vector_to_pos function.
    """
    return ang * math.pi / 180 

def deg_of_rad(ang): 
    """
    Useful for the add_vector_to_pos function.
    """
    return ang * 180 / math.pi 

def compute_noise(param):
    """
    Useful for the add_vector_to_pos function.
    """    
    epsilon = param
    theta = random_sample() * 2 * math.pi
    r = -1. / epsilon * \
        (np.real(lambertw((random_sample() - 1) / math.e, k=-1)) + 1)
    return r, theta


def load_data():
    """
    Loads the trip dataset and the POI data set from csv
    files and returns them as panda dataframes.
    """
    
    # load trip data with user locations
    dtypes = {"user_id": np.int32, "time": str,
              "lat": np.float64, "lon": np.float64, "poi": np.int32}
    tripdata = pd.read_csv('tripdata.csv', names=[
        "user_id", "time", "lat", "lon", "poi"], header=0, dtype=dtypes)
    tripdata["time"] = pd.to_datetime(tripdata["time"], errors='coerce')
    
    # only use 100 entries of the dataset
    tripdata = tripdata.head(100)

    # load POI location data
    dtypes = {"poi_id": np.int32, "category": str,
              "lat": np.float64, "lon": np.float64}
    poidata = pd.read_csv('poidata.csv', names=[
        "poi_id", "category", "lat", "lon"], header=0, dtype=dtypes)

    return tripdata, poidata



def add_noise(tripdata, epsilon):
    """
    Copies the panda dataframe tripdata and applies a
    Location Privacy Protection Mechanism. Epsilon
    determines the level of noise. Returns the noisy trip
    data.
    """
    noisy_tripdata = tripdata.copy()
    
    # the nearest_poi information in the dataset will not
    # be correct after adding the noise anymore, so drop
    # it:
    del noisy_tripdata["poi"]

    # apply Geo-Indistinguishability
    for i, row in tripdata.iterrows():
        r, theta = compute_noise(epsilon)

        lat = row["lat"]
        lon = row["lon"]
        lat_noise, lon_noise = add_vector_to_pos(lat, lon, r, theta)
        # write output (with same precision as in original data)
        noisy_lat = round(lat_noise, 5)
        noisy_lon = round(lon_noise, 5)

        noisy_tripdata.at[i, "lat"] = noisy_lat
        noisy_tripdata.at[i, "lon"] = noisy_lon

    return noisy_tripdata



def get_distance_in_meters(lat1, lon1, lat2, lon2):
    """
    Returns the distance between (lat1, lon1) and
    (lat2, lon2) in meters.
    """
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    distance = haversine(loc1, loc2) * 1000
    return distance



#########################################################
#                                                       #
#   Fill in your code here!                             #
#                                                       #
#########################################################

# run code:

tdata, pdata = load_data()
print("--- loaded data ---")

noisy_tdata = add_noise(tdata, EPSILON)
print("--- added noise ---")

#########################################################
#                                                       #
#   Task 1: Match noisy data                            #
#   Match each row of the noisy trip data with the      #
#   closest POI from the POI dataset. You can use the   #
#   get_distance_in_meters function.                    #
# Task 1: Match noisy data
# Add a new column "closest_poi" to the noisy_tdata dataframe
noisy_tdata["closest_poi"] = ""

# Iterate over each row in the noisy_tdata dataframe
for i, row in noisy_tdata.iterrows():
    # Initialize the minimum distance to infinity
    min_distance = float('inf')
    # Initialize the closest_poi variable to None
    closest_poi = None
    
    # Iterate over each row in the pdata dataframe
    for j, poi_row in pdata.iterrows():
        # Calculate the distance between the current row in noisy_tdata and the current row in pdata
        distance = get_distance_in_meters(row["lat"], row["lon"], poi_row["lat"], poi_row["lon"])
        
        # Check if the calculated distance is smaller than the current minimum distance
        if distance < min_distance:
            # Update the minimum distance and closest_poi variables
            min_distance = distance
            closest_poi = poi_row["poi_id"]
    
    # Assign the closest_poi value to the "closest_poi" column in the current row of noisy_tdata
    noisy_tdata.at[i, "closest_poi"] = closest_poi

print("Noisy data:", noisy_tdata)
noisy_tdata.to_csv('noisy_tripdata.csv', index=False)
#########################################################


#########################################################
#   Task 2: Measuring Privacy Gain                      #
#########################################################
# Measure privacy gain as represented by the ratio of the number of incorrectly matched POIs, 
# to the total number of POIs associated with the users.
# Task 2: Measuring Privacy Gain
# Measure privacy gain as represented by the ratio of the number of incorrectly matched POIs,
# to the total number of POIs associated with the users.

incorrect_matches = 0
for index, row in noisy_tdata.iterrows():
    if tdata.iloc[index]['poi'] != row['closest_poi']:
        incorrect_matches += 1

privacy_gain = incorrect_matches / len(tdata)
print("Privacy Gain:", privacy_gain)
# Result: Privacy Gain: 0.42


#########################################################
#   Task 3: Measuring Utility Loss                      #
#########################################################
# Calculate the utility loss in terms of the additional walking distance users have to walk 
# to the new locations. Measure the distance of the noisy trip data to POI positions and 
# compare the data of non-noisy trip data to POI positions. Compute the average distance that 
# has to be walked.
# Calculate the total difference in distance between the original trip data and the noisy trip data
total_distance_difference = 0

# Iterate over each row in the noisy_tdata dataframe
for index, row in noisy_tdata.iterrows():
    # Calculate the distance between the original location and the original POI location
    original_distance = get_distance_in_meters(
        tdata.iloc[index]['lat'], tdata.iloc[index]['lon'],
        pdata.loc[pdata['poi_id'] == tdata.iloc[index]['poi'], 'lat'].values[0],
        pdata.loc[pdata['poi_id'] == tdata.iloc[index]['poi'], 'lon'].values[0])
    
    # Calculate the distance between the noisy location and the closest POI location
    noisy_distance = get_distance_in_meters(
        row['lat'], row['lon'],
        pdata.loc[pdata['poi_id'] == row['closest_poi'], 'lat'].values[0],
        pdata.loc[pdata['poi_id'] == row['closest_poi'], 'lon'].values[0])
    
    # Add the difference in distance to the total distance difference
    total_distance_difference += (noisy_distance - original_distance)

# Calculate the average additional distance that needs to be walked
average_additional_distance = total_distance_difference / len(tdata)

# Print the average utility loss in meters
print("Average Utility Loss (meters):", average_additional_distance)
# Result: Average Utility Loss (meters): 8.294935822683216