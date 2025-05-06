from functools import partial
import h5py 
import os
import numpy as np
import numpy.ma as ma
import pickle
import copy
from collections import defaultdict
from collections.abc import Iterable
import multiprocessing as mp

import extrusionlib as el
from extrusionlib.lef_factory import *
import argparse


parser = argparse.ArgumentParser(description='1d extrusion with different densities')
parser.add_argument('-o', '--save_basepath', default="extrusion_density_sims", help='location where simulation save folder is created')
args = parser.parse_args()

def bind_walker(walker, context, lifetime, probs): # executed inside of step function which only feeds walker and context, so other params need to be added thru partial function
    """Birth in a list of positions... 
    If all are occupied, it adds search_direction to all positions and retries 
    
    This is necessary to guarantee loading if all sites are occupied! 
    
    For PolII, for example, search_direction should probably be the direction of the gene 
    
    Parameters
    ----------

    
    """
    if not walker['solution']:
        return
    assert len(probs) == sum(context.L)
    
    if np.random.random() < (1/lifetime):
        
        cumprobs = np.cumsum(probs)
        cumprobs = cumprobs / cumprobs[-1]

        while True:
            a = np.searchsorted(cumprobs, np.random.random())

            if (context[a:a+len(walker)] == 0).all():
                walker['solution'] = False
                walker['pos',:] = np.arange(a,a+len(walker))
                context[a:a+len(walker)] = walker.legs
                return 
def extrusion(d_X, d_AB, lambda_, chain_list, n_steps, mon_types):
    
    save_path = f'/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/extrusion_and_compartments/{args.save_basepath}/lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
    os.makedirs(save_path, exist_ok=True)
    traj_fh = f'LEFpositions_steps-{n_steps}_nChains-{n_chains}_Xboundaries.h5'
    try:
        with h5py.File(f'{save_path}/{traj_fh}', 'r') as traj_file:
            traj_file['positions']
    except FileNotFoundError:
        v_lef = 1 # Drift velocity of LEF leg
        D_lef = 0 # Diffusion coeff of LEF leg (set to 0 for simplicity)

        dna_lifetime = lambda_/(2*v_lef) # Average lifetime that a LEF stays on the DNA. The lifetimes are exponentially distributed   
        f_bound = 0.8333 # Fraction of total LEFs that are on DNA at any given time
        sol_lifetime = (dna_lifetime/f_bound) - dna_lifetime

        # Defining loading probability landscape, so that LEFs load in X and in AB with chosen densities
        # Defining loading probability landscape
        if not d_AB:
            pr_AB = 0
            pr_X = 1/n_X
            n_lefs_AB = 0
        else:
            n_lefs_AB = int((lambda_/d_AB)*n_AB/(2*f_bound*dna_lifetime))
            pr_AB = 1/((d_AB/d_X)*n_X + n_AB)
            pr_X = pr_AB*(d_AB/d_X)


        n_lefs_X  = int((lambda_/d_X)*n_X/(2*f_bound*dna_lifetime))
        n_lefs = int(n_lefs_X+n_lefs_AB) # Total numer of LEFs in the system (DNA + solution)

        probs = np.zeros(N)
        probs[mon_types == 2] = pr_X
        probs[mon_types != 2] = pr_AB

        if np.sum(probs) != 1.0:
            print(np.sum(probs))

        # Defining extrusion boundaries as the starts and ends of X domains
        mon_copy = copy.deepcopy(mon_types)
        mon_copy[mon_types != 2] = 0
        bounds = np.zeros(N)
        bounds = np.where(np.diff(mon_copy) != 0)[0]

        # Boundary strengths defined such that boundaries are impermeable
        capture_prob = 1
        release_prob = 0.0

        capture_dict = {
            -1:{},
            1 :{}
        }
        release_dict = {
            -1:{},
            1 :{}
        }

        for bound in bounds:
            capture_dict[-1][bound] = capture_prob
            release_dict[-1][bound] = release_prob
            capture_dict[1][bound] = capture_prob
            release_dict[1][bound] = release_prob    

        print(
        f'Average number of times a LEF will unbind and rebind over the course of the simulation:\
            \n{np.round(n_steps/(dna_lifetime + sol_lifetime),2)}'
             )

        # Initializing LEFs
        sim_context = el.lef_dynamics.Context(chain_list)

        for lef in range(n_lefs):
            legs = []
            for leg_direc in [-1, 1]: # orientation of stepping for each leg
                leg = Leg(leg_template={
                    'v':v_lef,
                    'D':D_lef,
                    'max_v':v_lef,
                    'max_D':D_lef,
                    'dir':leg_direc,
                    'pos':np.nan,
                    'stalled':False,
                    'halted':False
                    },
                reset_args=['stalled', 'pos', 'halted']
                         )
                capture_fn = partial(CTCF_capture, capture_probs=capture_dict)
                leg.add_cell_check('ctcf_capture', capture_fn) 
                release_fn = partial(CTCF_release, release_probs=release_dict)
                leg.add_cell_check('ctcf_release', release_fn)
                leg.add_encounter_check('stall', stall_any)
                legs.append(leg)
            walker = Walker(legs=legs, walker_attrs={'solution':True, 'name':'cohesin'})

            bind_fn = partial(bind_walker, lifetime=sol_lifetime, probs=probs)
            walker.add_step_check('bind', bind_fn)
            unbind_fn = partial(unbind, lifetime=dna_lifetime)
            walker.add_step_check('unbind', unbind_fn)

            sim_context.add_walker(walker)

        # Saving output
        n_chunks = 500
        bins = np.linspace(0, n_steps, n_chunks, dtype=int)

        with h5py.File(f"{save_path}/{traj_fh}", "w") as f:
            dset = f.create_dataset('positions', 
                             shape=(n_steps, n_lefs, 2),
                             dtype=np.int32, 
                             compression="gzip")
            f.attrs['chain_list'] = chain_list
            f.attrs['n_lefs'] = n_lefs


            # Doing the 1D sim
            for start, end in zip(bins[:-1], bins[1:]):
                pos_list = []
                for i in range(start, end):
                    sim_context.step()
                    pos = np.vstack(tuple([lef['pos',:] for lef in sim_context.walkers]))
                    pos_list.append(pos)

                pos_array = np.array(pos_list) 
                pos_array[np.isnan(pos_array)] = -1
                pos_array = pos_array.astype(int)

                dset[start:end] = pos_array

    print(f"Completed lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}/{traj_fh}")

# Defining chains
with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info_extrusion.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
chain_list = n_chains*[n_per_chain]
N = sum(chain_list)

# Defining compartment landscape
mon_types = np.array(n_chains*list(monInfo['compartment_ID']))
n_X = len(mon_types[mon_types == 2]) # number of X monomers
n_AB = N - n_X # number of A or B monomers

# Defining one lambda for all LEFs and different ds based on compartment type
lambda_ = 100
d_list = [50, 100, 200, 400, 1000]

# Defining params for LEF dynamics
n_steps = 500_000 # Duration of the simulation
extrusion_input_list = [(d, d, lambda_) for d in d_list]

with mp.Pool(len(extrusion_input_list)) as p:
    f = partial(extrusion, 
                chain_list=chain_list, n_steps=n_steps,\
                mon_types=mon_types)
    p.starmap(f, extrusion_input_list)

