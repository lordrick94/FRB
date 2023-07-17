"""
This script is used to estimate the absolute magnitude of FRBs from their redshifts.
"""
# Regular modules
import os
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Astropy modules
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo

# FRB modules
from frb.galaxies import hosts

def estimate_Mr_from_z(filename=None):

    # Host Galaxy Mr Distribution
    xvals, prob1 = hosts.load_Mr_pdf(pdf_file=None)

    # Read in the FRB catalog
    if filename is None:
        filename = os.path.join(os.environ.
                                get('desi_calc'),'final_curve_1.csv')
    df = pd.read_csv(filename)

    # Select only the FRBs with BGS redshifts 0.1 < z < 0.7
    df = df[(df['z'] > 0.01) & (df['z'] < 0.7)]

    ra  = [x*u.degree for x in df['RA']]
    dec = [x*u.degree for x in df['Dec']]
    z   = df['z'].values

    # Hiding this here to avoid a dependency
    from dustmaps.sfd import SFDQuery
    # Deal with extinction
    coords = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs')
    sfd = SFDQuery()
    Ar = sfd(coords)*2.285 # SDSS r-band
    index_Ar = np.arange(len(Ar))
    Index_Ar = np.random.shuffle(index_Ar)
    mw_extinction = np.squeeze(Ar[Index_Ar]) 

    dist_mod = cosmo.distmod(z).value

    n_samples = len(z)
    print('Number of FRBs = {:d}'.format(n_samples))

    mr_value = 19.5
    r_mag_1 = mr_value
    r_mag_2 = mr_value + 1
    r_mag_3 = mr_value + 2
    r_mag_4 = mr_value + 3

    samples = 10000
    frac = []
    m_rs= []
    Dist_s= []
    for i in range(samples):
        M_r = np.random.choice(xvals, n_samples, p=prob1)
        Index_ = np.random.shuffle(np.arange(len(Ar)))
        mw_extinction = np.squeeze(Ar[Index_])
        Dist_mod = dist_mod[Index_]
        host_m_r = Dist_mod + M_r + mw_extinction
        frac1 = len(np.where(host_m_r <= r_mag_1)[0])/n_samples
        frac2 = len(np.where((host_m_r > r_mag_1) & (host_m_r <= r_mag_2))[0])/n_samples
        frac3 = len(np.where((host_m_r > r_mag_2) & (host_m_r <= r_mag_3))[0])/n_samples
        frac4 = len(np.where((host_m_r > r_mag_3) & (host_m_r <= r_mag_4))[0])/n_samples
        frac5 = len(np.where(host_m_r > r_mag_4)[0])/n_samples
        val = np.squeeze(np.array([frac1,frac2,frac3,frac4,frac5]))
        # Save
        frac.append(val)
        m_rs.append(host_m_r.flatten())
        Dist_s.append(Dist_mod.flatten())

    frac_ = np.round(np.mean(frac,axis=0),2)

    figfile = os.path.join(os.environ.get('desi_calc'),'frb_z_to_Mr.png')

    if figfile:
        plt.style.use('seaborn-poster')
        fig, ax = plt.subplots(figsize=(8,6))

        # Save the chart so we can loop through the bars below.
        bars = ax.bar(
            x= [0,1,2,3,4],
            height= frac_ ,
            tick_label=[f'$<$ {mr_value}',
                        f'{mr_value} $-$ {mr_value+1}',
                        f'{mr_value+1} $-$ {mr_value+2}',
                        f'{mr_value+2} $-$ {mr_value+3}',
                        f'$>$ {mr_value+3}']

        )

        # Axis formatting.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top')

        # Add text annotations to the top of the bars.
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            round(bar.get_height(), 2),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )

        ax.set_xlabel('R-band Apparent magnitude [AB] ', labelpad=15, color='#333333')
        ax.set_ylabel('Fraction of CHIME FRBs Hosts', labelpad=15, color='#333333')
        ax.set_title('R-band Magnitude of CHIME FRBs in BGS', pad=15, color='#333333',
                    weight='bold')

        fig.tight_layout()

        figfile = os.path.join(os.environ.get('desi_calc'),'frb_z_to_Mr.png')

        #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.savefig(figfile, dpi=300)
        plt.close()
        print('Wrote {:s} '.format(figfile))

if __name__ == '__main__':
    import time
    t0 = time.time()
    estimate_Mr_from_z()
    t1 = time.time()
    print('Elapsed time = {:g} seconds'.format(t1-t0))


