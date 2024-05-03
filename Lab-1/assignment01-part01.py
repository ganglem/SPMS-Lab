import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# constants
DATAFILE = "maltespitz.csv"
MAPFILE = "map-berlin.png"
PLOTFILENAME = "plot-berlin.pdf"

MINLON = 13.3477
MAXLON = 13.4089
MINLAT = 52.4981
MAXLAT = 52.5240

PLOTTITLE = "Gruppe 1"


def load_data():
    """
    Function imports location protocol from Malte Spitz. Returns data in a pandas dataframe.
    """
    dtypes = {"longitude": np.float64, "latitude": np.float64}
    df = pd.read_csv(DATAFILE, names=["start", "end", "service", "incoming-outgoing", "longitude", "latitude", "direction","cell-id-a","cell-id-b"], header=0, dtype=dtypes)
    return df

def filter_data(df):
    """
    Function takes a pandas dataframe, selects all rows with longitude an latitude in a given area and returns the filtered data frame.
    """
    df = df[(df.latitude >= MINLAT) & (df.latitude <= MAXLAT) & (df.longitude >= MINLON) & (df.longitude <= MAXLON)]
    return df

def prepare_plot():
    """
    Function prepares a pyplot, loads a background image and sets the correct dimensions. String "name" is used as title. Returns an matplotlib.axes object for further usage.
    """
    map = plt.imread(MAPFILE)
    fig, ax = plt.subplots()
    
    ax.set_title(PLOTTITLE)
    ax.set_xlim(MINLON, MAXLON)
    ax.set_ylim(MINLAT, MAXLAT)

    ax.imshow(map, zorder=0, extent= (MINLON, MAXLON, MINLAT, MAXLAT), aspect="auto")
    return ax


#########################################################
#                                                       #
#   Fill in your code here!                             #
#                                                       #
#########################################################

# get data
dataset = load_data()
dataset = filter_data(dataset)

# prepare plot creation
plot_handle = prepare_plot()

#########################################################
#                                                       #
#   Task:                                               #
#   Plot the coordinates to the prepared canvas. Use a  #
#   scatter plot. Mark each point with a red x.         #
plot_handle.scatter(dataset.longitude, dataset.latitude, marker='x', color='red')
#                                                       #
#########################################################


#########################################################
#                                                       #
#   Task:                                               #
# Round the coordinates to 3 positions after decimal point. Plot again. Use yellow markers this time.
dataset["longitude"] = dataset["longitude"].round(3)
dataset["latitude"] = dataset["latitude"].round(3)
plot_handle.scatter(dataset.longitude, dataset.latitude, marker='o', color='yellow')
#                                                       #
#########################################################


#########################################################
#                                                       #
#   Task:                                               #
#   Round the coordinates to 2 positions after decimal  #
#   point. Plot again. Use blue markers this time.      #
dataset["longitude"] = dataset["longitude"].round(2)
dataset["latitude"] = dataset["latitude"].round(2)
plot_handle.scatter(dataset.longitude, dataset.latitude, marker='o', color='blue')
#                                                       #
#########################################################


# The implication on privacy and the utility LPPMs have in this case
# The more the data is rounded the less information can be derived from the map
# The red data indicates preicse positions. The yellow data points are less excact and may only indicate the street rather than the exact address
# The blue data only indicates a rough position.
# -> The data is changed so that less information can be derived from it




# create plot
#plt.show() # use this instead of savefig to directly open the plot
plt.savefig(PLOTFILENAME, dpi=300)