import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord,Angle
import astropy.units as u
import numpy as np


def plot_cord_in_galactic(new_df):
    """
    This function plots the RA and Dec values in galactic coordinates (l and b)
    :param new_df: A pandas dataframe with RA and Dec values
    :return: A plot of the RA and Dec values in galactic coordinates (l and b)
    """
    # Get the RA and DEC values (in degrees)
    #Condition if df['RA'] and df['Dec'] are not present in the dataframe ask user to enter the column names
    if 'RA' not in new_df.columns and 'Dec' not in new_df.columns:
        print("Please enter the column names for RA and Dec in the dataframe")
        ra = input("Enter the column name for RA: ")
        dec = input("Enter the column name for Dec: ")
        df = new_df
        ra_values = df[ra]
        dec_values = df[dec]
    else:
        df = new_df
        ra_values = df['RA']
        dec_values = df['Dec']

    # Convert RA and DEC to galactic coordinates (l and b)
    coords = SkyCoord(ra=Angle(ra_values, unit=u.degree), dec=Angle(dec_values, unit=u.degree), frame='icrs')
    gal = coords.galactic

    # Create a aitoff projection plot
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection='aitoff')
    ax.grid(True)


    x_rad = gal.l.wrap_at('180d')
    x = np.radians(x_rad)
    y = np.radians(gal.b)

    ax.scatter(- 1*x, y, s=2, alpha=0.5)

    # Set the plot title and labels
    plt.title('FRBs in Galactic Coordinates')
    plt.xlabel('Galactic Longitude (l)')
    plt.ylabel('Galactic Latitude (b)')

    # Show the plot
    plt.show()