import os
import argparse
from dotenv import load_dotenv

# Time Manipulation
from datetime import datetime, timedelta
from dateutil import parser
from time import time

# Data Processing
import pandas as pd
from pandas import Timestamp, DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.coords import CoordinateSequence
from shapely.geometry import Point

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Serialization
import pickle

# Map Visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


def get_time_in_seconds_from_datetime_str(datetime_str: str) -> int:
    datetime_obj = parser.parse(datetime_str)
    time = datetime_obj.time()
    seconds = timedelta(hours=time.hour, minutes=time.minute, seconds=time.second).total_seconds()
    return seconds


def get_time_in_seconds_from_timestamp(timestamp: Timestamp) -> int:
    datetime_str = str(timestamp)
    return get_time_in_seconds_from_datetime_str(datetime_str)
    

"""
Columns here are 

['VendorID' 'tpep_pickup_datetime' 'tpep_dropoff_datetime'
 'passenger_count' 'trip_distance' 'RatecodeID' 'store_and_fwd_flag'
 'PULocationID' 'DOLocationID' 'payment_type' 'fare_amount' 'extra'
 'mta_tax' 'tip_amount' 'tolls_amount' 'improvement_surcharge'
 'total_amount' 'congestion_surcharge' 'Airport_fee']

The problem is to predict the fare for each pickup location ID and datetime
Input: longitude latitude, tpep_pickup_datetime 
Output: fare_amount
"""
def process_taxi_data(
    taxi_trip_record_filename: str, 
    time_column: str ='tpep_pickup_datetime', 
    location_column: str = 'PULocationID', 
    fare_column: str = 'fare_amount'
) -> tuple[DataFrame, DataFrame]:
    df = pd.read_parquet(taxi_trip_record_filename, engine='fastparquet')
    df = df[[time_column, location_column, fare_column]]   
    
    # process the time
    df[time_column] = df[time_column].apply(get_time_in_seconds_from_timestamp)

    # get x and y
    X = df[[time_column, location_column]]
    y = df[[fare_column]]

    # rename columns
    X = X.rename(columns={time_column: 'Time', location_column: 'Location ID'})
    y = y.rename(columns={fare_column: 'Fare'})

    return X, y


def process_location_id_data(taxi_zones_filename: str) -> GeoDataFrame:
    gdf = gpd.read_file(taxi_zones_filename)
    gdf = gdf.to_crs(epsg=4326)
    return gdf
    

def get_location_id_from_coords(
    lat: int, 
    lon: int, 
    gdf: GeoDataFrame, 
    location_id_column: str = 'OBJECTID'
) -> int:
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(index=[0], geometry=[point], crs=gdf.crs)
    result = gpd.sjoin(point_gdf, gdf, how='left', predicate='within')
    return result[location_id_column][0]


def generate_random_forest_regressor(
        model_filename: str = 'models/taxi_fare_model.pkl',
        X_test_filename: str = 'test/X_test.parquet',
        y_test_filename: str = 'test/y_test.parquet'
    ) -> None:
    assert model_filename.endswith('.pkl'), "Model file must be a .pkl file."
    assert X_test_filename.endswith('.parquet'), "X test file must be a .parquet file"
    assert y_test_filename.endswith('.parquet'), "y test file must be a .parquet file"

    taxi_zones_filename = os.getenv("TAXI_ZONES_FILENAME")
    taxi_trip_record_filename = os.getenv("TAXI_TRIP_RECORD_FILENAME")
    
    gdf = process_location_id_data(taxi_zones_filename)
    X, y = process_taxi_data(taxi_trip_record_filename)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    with open(model_filename, 'wb') as file:
        pickle.dump(rf, file)

    X_test.to_parquet(X_test_filename)
    y_test.to_parquet(y_test_filename)


def load_model(model_filename: str = 'models/taxi_fare_model.pkl') -> RandomForestRegressor:
    assert model_filename.endswith('.pkl'), "File must be a .pkl file"

    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
        return loaded_model
    

def load_test_data(
    X_test_filename: str = 'test/X_test.parquet', 
    y_test_filename: str = 'test/y_test.parquet'
) -> tuple[DataFrame]:
    assert X_test_filename.endswith('.parquet'), "X test file must be a .parquet file"
    assert y_test_filename.endswith('.parquet'), "y test file must be a .parquet file"
    
    X_test = pd.read_parquet(X_test_filename, engine='fastparquet')
    y_test = pd.read_parquet(y_test_filename, engine='fastparquet')

    return X_test, y_test

if __name__ == "__main__":
    load_dotenv()

    argparser = argparse.ArgumentParser(
        prog = 'TaxiFarePredictor',
        description = 'A program that predicts NYC taxi fares based on location and time of the day.'
    )
    subparsers = argparser.add_subparsers(
        dest='command',
        help='Specify which subcommand to run.'
    )

    generate_model_parser = subparsers.add_parser(
        'generate_model',
        help='Run the generate_model command to generate a new model.'
    )
    generate_model_parser.add_argument(
        '--model_filename',
        type=str,
        help="An optional argument for the model's destination filename. The filename must use the .pkl extension. \
            Default value is models/taxi_fare_model.pkl."
    )
    generate_model_parser.add_argument(
        '--X_test_filename',
        type=str,
        help="An optional argument for the X_test set destination filename. \
            The filename must use the .parquet extension. Default value is test/X_test.parquet."
    )
    generate_model_parser.add_argument(
        '--y_test_filename',
        type=str,
        help="An optional argument for the y_test set destination filename. \
            The filename must use the .parquet extension. Default value is test/y_test.parquet."
    )

    use_model_parser = subparsers.add_parser(
        'use_model',
        help='Run the use_model command to use an existing model.'
    )
    use_model_parser.add_argument(
        '--model_filename',
        type=str,
        help='An optional argument for the filename of the model that will be used. \
            The filename must use the .pkl extension. Default value is models/taxi_fare_model.pkl'
    )

    evaluate_model_parser = subparsers.add_parser(
        'evaluate_model',
        help='Run the evaluate_model command to evaluate an existing model.'
    )
    evaluate_model_parser.add_argument(
        '--model_filename',
        type=str,
        help='An optional argument for the filename of the model that will be used. \
            The filename must use the .pkl extension. Default value is models/taxi_fare_model.pkl'
    )
    evaluate_model_parser.add_argument(
        '--X_test_filename',
        type=str,
        help="An optional argument for the X_test set destination filename. \
            The filename must use the .parquet extension. Default value is test/X_test.parquet."
    )
    evaluate_model_parser.add_argument(
        '--y_test_filename',
        type=str,
        help="An optional argument for the y_test set destination filename. \
            The filename must use the .parquet extension. Default value is test/y_test.parquet."
    )

    args = argparser.parse_args()
    kwargs = {key: value for key, value in vars(args).items() if key != 'command' and value is not None}

    if args.command == 'generate_model':
        generate_random_forest_regressor(**kwargs)
        
    elif args.command == 'use_model':
        start_time = time()
        rf = load_model(**kwargs)
        gdf = process_location_id_data(os.getenv("TAXI_ZONES_FILENAME"))
        go_on = True
        
        while go_on:
            inputs = input('Please supply the latitude, longitude, and time of day as numeric values in comma separated format.\n> ')
            lat, lon, time_of_day= None, None, None
            
            try:
                # Gather latitude, longitude, and time
                lat, lon, time_of_day = inputs.split(',')
                lat, lon = float(lat), float(lon)
            except ValueError as e:
                print("An error occurred: {e}")
                continue
            
            seconds = None
            if time_of_day.isnumeric():
                seconds = int(float(time_of_day))
                if not (0 <= seconds <= 86400):
                    print("Supplied numeric value for time_of_day cannot be outside of [0, 86400].")
                    continue
            try:
                seconds = get_time_in_seconds_from_datetime_str(time_of_day)
            except Exception as e:
                print("An error occurred: {e}")
                continue
            
            location_id = get_location_id_from_coords(lat=lat, lon=lon, gdf=gdf)
            X = [seconds, location_id]
            predicted_fare = rf.predict(X)
            print('Predicted fare is:', predicted_fare)

            while True:
                yes_or_no = input('Would you like to supply more inputs (Y/N) ?\n> ').lower()
                if yes_or_no.lower() in ['n', 'no']:
                    go_on = False
                    break
                elif yes_or_no.lower() in ['y', 'yes', 'yep', 'yeah', 'yup']:
                    go_on = True
                    break
                else:
                    print('Please specify yes or no (Y/N). Other inputs are not accepted and will cause me to ask again.')

    elif args.command == 'evaluate_model':
        rf = load_model(**kwargs)
        X_test, y_test = load_test_data(**kwargs)
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        print(f'MAE: {mae}')
        print(f'MSE: {mse}')
        print(f'RMSE: {rmse}')
        print(f'R²: {r2}')