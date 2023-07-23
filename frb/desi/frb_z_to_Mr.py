"""
This script is used to estimate the absolute magnitude of FRBs from their redshifts.
"""
# Regular modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Astropy modules
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo

# FRB modules
from frb.galaxies import hosts

def estimate_mr_from_z(filename=None):

    # Host Galaxy Mr Distribution
    xvals, prob1 = hosts.load_Mr_pdf(pdf_file=None)

    # Read in the FRB catalog
    if filename is None:
        filename = os.path.join(os.environ.
                                get('desi_calc'),'frbs_z_table.csv')
    df = pd.read_csv(filename)

    # Select only the FRBs with BGS redshifts 0.1 < z < 0.6
    #df = df[(df['z'] > 0.6) & (df['z'] < 1.6)]

    #Remove the FRBs with no redshifts
    df = df[df['z'] > 0]

    n_frbs = len(df)

    # Select FRBs in the DESI footprint
    df = df[df['is_in_DESI'] == True]

    ra  = [x*u.degree for x in df['RA']]
    dec = [x*u.degree for x in df['Dec']]
    z   = df['z'].values

    n_samples = len(df)

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

    
    print('Number of FRBs = {:d}'.format(n_frbs))
    print('Number of FRBs in desi = {:d}'.format(n_samples))
    print(' "%" of FRBs in desi = {:g}'.format(n_samples/n_frbs))
    samples = 100
    m_rs= []
    frac = []

    for i in range(samples):
        M_r = np.random.choice(xvals, n_samples, p=prob1)
        Index_ = np.random.shuffle(np.arange(len(Ar)))
        mw_extinction = np.squeeze(Ar[Index_])
        Dist_mod = dist_mod[Index_]
        host_m_r = Dist_mod + M_r + mw_extinction
        m_rs.append(host_m_r.flatten())
        df['mr'] = host_m_r.flatten()


        # Calculate the fraction of FRBs with Mr < 19.5 and z<0.6 - BGS Bright
        frac1 = len(df[(df['mr'] <= 19.5) & (df['z'] < 0.6)])/n_frbs

        # Calculate the fraction of FRBs with 19.5 < Mr < 20.175 and z<0.6 - BGS Faint
        frac2 = len(df[(df['mr'] > 19.5) & (df['mr'] <= 20.175) & (df['z'] < 0.6)])/n_frbs

        # Calculate the fraction of FRBs with 20 < Mr < 24 and 0.6 < z < 1.6 - ELGs
        frac3 = len(df[(df['mr'] > 20) & (df['mr'] <= 24) & (df['z'] > 0.6) & (df['z'] <= 1.6)])/n_frbs

        # Calculate the fraction of overlap in the above two samples
        frac_overlap = len(df[(df['mr'] <= 20.175) & (df['mr'] > 20) & (df['z'] > 0.6) & (df['z'] <= 1.6)])/n_frbs

        # Calculate the fraction for the rest of the FRBs which are not in the above three categories
        frac_rest = 1 - (frac1 + frac2 + frac3 - frac_overlap) 

        val = np.squeeze(np.array([frac1,frac2,frac3,frac_rest]))

        # Save
        frac.append(val)

    frac_ = np.round(np.mean(frac,axis=0),2)
    
    df['mr'] = np.mean(m_rs,axis=0)

    print(df)

    figfile = os.path.join(os.environ.get('desi_calc'),'frb_z_to_Mr.png')

    if figfile:
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(8,6))

        # Save the chart so we can loop through the bars below.
        bars = ax.bar(
            x= [0,1,2,3],
            height= frac_ ,
            tick_label=['$<$ 19.5 - BGS Bright',
                        '19.5 $-$ 20.175 - BGS Faint',
                        '20 $-$ 24 - ELGs',
                        'out of range',
                        ]

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
        ax.set_title('R-band Magnitude of CHIME FRBs in DESI', pad=15, color='#333333',
                    weight='bold')

        fig.tight_layout()

        #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        plt.savefig(figfile, dpi=300)
        plt.close()
        print('Wrote {:s} '.format(figfile))

if __name__ == '__main__':
    import time
    t0 = time.time()
    estimate_mr_from_z('final_curve_3.csv')
    t1 = time.time()
    print('Elapsed time = {:g} seconds'.format(t1-t0))


