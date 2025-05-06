import numpy as np
import pandas as pd
import scipy 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import polychrom
from polychrom import polymer_analyses
from polychrom.hdf5_format import list_URIs, load_URI

def Ps_sorter(blocks, bin_edges, chains, cutoff=1.1):

    def process(uri):
        idx = int(uri.split('::')[-1])
        try:
            data = load_URI(uri)['pos']
        except:
            print(uri)

        ser = {}
        chunk = np.searchsorted(blocks, idx, side='right')
        ser['chunk'] = [chunk]

        bins = None
        contacts = np.zeros(len(bin_edges)-1)
        for st, end in zip(chains[0:-1],chains[1:]):
            conf = data[st:end,:]
            x,y = polymer_analyses.contact_scaling(conf, bins0=bin_edges, cutoff=cutoff)
            
            if bins is None:
                bins = x

            contacts = contacts + y

        ser['Ps'] = [(bins, contacts)]
        return pd.DataFrame(ser)

    return process


def pin_normalization(x, y, x_pin):

    min_idx = np.argmin(np.abs(x - x_pin))
    y = y/y[min_idx]

    return y


def log_derivative(x, y, sigma=1):
    logx = np.log10(x)
    logy = np.log10(y)

    slope_y = (logy[1:] - logy[0:-1])/(logx[1:] - logx[0:-1])
    slope_y = gaussian_filter1d(slope_y, sigma=sigma)
    slope_x = 10**((logx[1:] + logx[0:-1])/2)

    return slope_x, slope_y


def interpolate_Ps(bins, Ps, N):
    x = np.log10(bins)
    y = np.log10(Ps)

    f = interp1d(x, y, kind='linear')

    last_bin = np.floor(bins[-1])
    interp_x = np.log10(np.arange(last_bin))
    interp_y = f(interp_x)

    interp_y = np.r_[interp_y, [y[-1]]*int(N-last_bin)]
    interp_x = np.r_[interp_x, np.log10(np.arange(last_bin, N))]

    interp_y = 10**interp_y
    interp_x = 10**interp_x

    assert len(interp_x) == len(interp_y)
    assert len(interp_y) == N

    return interp_x, interp_y


