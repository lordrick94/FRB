"""
This code generates FRBs from using the CHIME FRB catalog values.
The purpose is to generate a large number of FRBs to be used in the prediction of the fraction of
FRB hosts with spectra in the DESI survey.
The FRB host redshift are inferred from the FRB dispersion measure (DM) and the Macquart relation.
"""
import numpy as np
import pandas as pd
import os
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from frb.dm import igm
from frb import mw

# DESI Calculations directory
DESI_DIR = os.environ.get('desi_calc')

def create_new_frbs_df(filename=None):
    """
    This function creates a new data frame with random values for the DM, RA, and Dec columns
    :param filename: .csv (optional), The name of the file containing the CHIME FRB catalog
    :return: A new data frame with random values for the DM, RA, and Dec columns
    """
    if filename is None:
        filename = os.path.join(DESI_DIR,'chime_frbs.csv')
    
    df = pd.read_csv(filename)

    target_df = df[['event_id','dm','RA','Dec']]
    # Create a new data frame with the same columns as target_df
    new_df = pd.DataFrame(columns=target_df.columns)

    # Define the number of rows for the new data frame
    num_rows = 10000

    # Generate random values for each column based on the range and distribution of the original data
    for column in target_df.columns:
        if column != 'event_id':
            original_array = target_df[column].values
            new_dm = np.random.choice(original_array, size=num_rows, replace=True)
            new_df[column] = new_dm

    # Set the event_id column to a sequence of numbers
    new_df['event_id'] = range(1, num_rows + 1)

    #Convert the DM column to units of pc/cm^3
    new_df['dm'] = [k * (u.pc / u.cm**3) for k in new_df['dm']]
    return new_df

def calculate_redshift(df, dm_halo=False, dm_ism=False):
    """
    This function calculates the redshift of the FRB host galaxy based on the Macquart relation
    :param dm: The dispersion measure of the FRB
    :param dm_halo: Boolean, True if the DM halo should be calculated, False if the DM halo should be set to 40 pc/cm^3
    :param dm_ism: Boolean, True if the DM ISM should be calculated, False if the DM ISM should be set to 75 pc/cm^3
    :return: The redshift of the FRB host galaxy
    """
    # Calculate DM halo and DM ISM
    ra_values = df['RA']
    dec_values = df['Dec']
    coords = SkyCoord(ra=Angle(ra_values, unit=u.degree), dec=Angle(dec_values, unit=u.degree), frame='icrs')
    
    # Conditional statements to calculate DM halo and DM ISM
    # If dm_halo is True, calculate DM halo
    if dm_halo:
        df['DMhalo'] = [mw.haloDM(c) for c in coords]
    # If dm_halo is False, set DM halo to 40 pc/cm^3
    else:
        df['DMhalo'] = 40
        df['DMhalo'] =  [k * (u.pc / u.cm**3) for k in df['DMhalo']]
    # If dm_ism is True, calculate DM ISM
    if dm_ism:
        df['DMISM'] = [mw.ismDM(c) for c in coords]
    # if dm_ism is False, set DM ISM to 0 pc/cm^3
    else:
        df['DMISM'] = 75
        df['DMISM'] =  [k * (u.pc / u.cm**3) for k in df['DMISM']]

    
    # Calculate DM_EG by subtracting DM halo and DM ISM from DM
    df['DM_Cosm'] = df['dm'] - df['DMhalo'] - df['DMISM']

    # If DM_Cosm is negative, set it to 0
    df['DM_Cosm'] = [0 if k < 0 else k for k in df['DM_Cosm']]
    
    # Calculate redshift using the Macquart relation
    z_values = []
    for dm in df['DM_Cosm']:
        try:
            z = igm.z_from_DM(dm)
        except ValueError:
            z = np.nan
        z_values.append(z)
    df['z'] = z_values

    df.to_csv('frb_redshifts.csv',index=False)

    return df

if __name__ == '__main__':
    import time
    start = time.time()
    df = create_new_frbs_df()
    df = calculate_redshift(df)
    print(df)
    end = time.time()
    print('Time elapsed: ', end - start)
