import pickle
import os
import sys
import time
import numpy as np
import pandas as pd
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
from polychrom.hdf5_format import HDF5Reporter
import warnings
import h5py
import argparse
import traceback

sys.path.append("../")
import manySphericalSpheres

parser = argparse.ArgumentParser(description='Loop extrusion and Compartment Simulator')
parser.add_argument('-g', '--GPU', help='GPU number to use', type=str)
parser.add_argument('-o', '--out_path', default='SC-1JoinedPol_errTol-0.01_coll-0.01', help='location where simulation save folder is created')
args = parser.parse_args()

def initiate_sim(polymer, r, cell_size, chains, GPU, ixn_mtx, mon_types):
        
    sim = simulation.Simulation(
        platform="CUDA",
        GPU=GPU,
        integrator="variableLangevin",
        error_tol=0.01,
        collision_rate=0.01,
        N=len(polymer),
        save_decimals=2,
        reporters=[reporter],
    )
    sim.set_data(polymer)
    
    sim.add_force(manySphericalSpheres.spherical_confinement_many(sim, r=r, cell_size=cell_size))
        
    
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            
            # Bond forces
            chains=chains, 
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0, # avg dist or dist at rest
                "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
            },

            # Angle forces
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5, # persistence length
            },

            nonbonded_force_func=forces.heteropolymer_SSW,
            nonbonded_force_kwargs={
                'attractionEnergy': 0,
                'attractionRadius': 1.5,
                'interactionMatrix': ixn_mtx,
                'monomerTypes': mon_types,
                'extraHardParticlesIdxs': []
            },

            except_bonds=True,
        )
    )
    
    return(sim)

# Defining chains
with open(f'chr15:0-6p5Mb.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file) # mon info for each copy of the polymer
mon_conds = args.monomer_filename.split(".")[0]
base_path =  f'/net/levsha/scratch2/emily/flagship'
n_per_chain = 3*monInfo['N']
n_chains = int(15/3)
chain_list = n_chains*[n_per_chain]
N = sum(chain_list)

if n_chains == 1: 
    chains = [(0, None, False)]
else:
    chains = [(i*n_per_chain, (i+1)*n_per_chain, False) for i in range(n_chains)]

# Block copol params
mon_types = np.tile(monInfo['mon'], 3*n_chains)
comp_param_sets = pd.read_csv(f"CompartmentSweep-AB_param_sets.csv", sep="\t", index_col=0)
AB = XA = XB = 0.00
XX = 0.00

# Extrusion params
traj_fh = 'LEFpositions_steps-100000_nChains-15.h5'
lambda_list = [55, 110]
dX_list     = [19, 37, 55, 110]
AB_fac  = np.array([5,10,np.inf])
max_lodAB = 0.3

# Other things
lef_wiggle_dist_unscaled = 0.2
lef_bond_dist_unscaled = 0.5
max_data_length = 100
restartSimulationEveryBlocks = 100
saveEveryBlocks = 10
n_steps_per_int = 750
n_lefIters = lef_positions.shape[0]

# Variables for initializing conformations to later position each chain into its own sphere
density = 0.2
n_chains_per_sphere = 1
r = (3 * n_per_chain * n_chains_per_sphere / (4 * 3.141592 * density)) ** (1/3)
cell_size = 5*r
offsets = [[cell_size*i,0,0] for i in range(int(n_chains/n_chains_per_sphere))]
offsets_multi = np.repeat(np.array(offsets), int(n_per_chain*n_chains_per_sphere), axis=0)

interrupt = False
for lambda_ in lambda_list:
    for d_X in dX_list:
        dAB_list = d_X*AB_fac
        dAB_list = dAB_list[np.where(lambda_/dAB_list < max_lodAB)[0]]
        dAB_list = np.round(np.where(dAB_list == np.inf, 0, dAB_list)).astype(int)

        for d_AB in dAB_list:
            
            extrusion_conds = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
            traj_path = f"{base_path}/extrusion_sweep/sims_{mon_conds}/{extrusion_conds}"
            print(f'\n\nlambda: {lambda_}\td_X: {d_X}\td_AB: {d_AB}')
        
            if os.path.exists(f'{traj_path}/_in_progress_flag'): # progress flag for 1d sim
                print('\t1D simulation is not finished\n\n')
                continue
            for idx in comp_param_sets.index:
                vals = comp_param_sets.iloc[idx]
                AA = vals.AA_attr
                BB = vals.BB_attr
                
                save_path = f'{base_path}/compartment_sweep/sims_{mon_conds}/{extrusion_conds}/AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}/{args.out_path}'
                os.makedirs(save_path, exist_ok=True)
                
                print(f'\n\nAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f} \t{extrusion_conds}')
                final_conf_fh = f'blocks_{int((n_lefIters/saveEveryBlocks)-max_data_length)}-{int(n_lefIters/saveEveryBlocks)-1}.h5'

                if os.path.exists(f'{save_path}/{final_conf_fh}'):
                    print("\tSimulation is complete\n\n")
                    continue
                    
                try:
                    with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                        pass       
                except FileExistsError:
                    print(f"\tSimulation is currently running on another GPU")
                    continue

                try:
                    traj_file = h5py.File(f'{traj_path}/{traj_fh}', 'r')
                    lef_positions = traj_file["positions"]
                except FileNotFoundError:
                    print('\tNo 1D simulation could be found\n\n')
                    continue

                try:
                    ixn_mtx = np.array([[AA, AB, XA],
                                       [AB, BB, XB],
                                       [XA, XB, XX],
                                      ])
                    reporter = HDF5Reporter(folder=save_path, max_data_length=max_data_length, 
                                            overwrite=False, check_exists=False, blocks_only=False
                                           )

                    confs_starting = np.concatenate([np.array(starting_conformations.grow_cubic(n_per_chain*n_chains_per_sphere, 
                                                                                                int((n_per_chain*n_chains_per_sphere)**(1/3)+2))) 
                                                     for i in range(int(n_chains/n_chains_per_sphere))])


                    polymer = confs_starting+offsets_multi

                    # Doing the thing
                    milker = extrusionPropagator([lef_positions])
                    for milking_iteration in range(n_lefIters//restartSimulationEveryBlocks):
                        sim = initiate_sim(polymer, r=r, cell_size=cell_size, chains=chains, GPU=args.GPU, ixn_mtx=ixn_mtx, mon_types=mon_types)

                        kbond = sim.kbondScalingFactor / (lef_wiggle_dist_unscaled ** 2)
                        lef_bond_dist = lef_bond_dist_unscaled * sim.length_scale

                        activeParams = {"length":lef_bond_dist, "k":kbond}
                        inactiveParams = {"length":lef_bond_dist, "k":0}
                        milker.setParams(activeParams, inactiveParams)
                        milker.setup(bondForce=sim.force_dict['harmonic_bonds'],
                                     blocks=restartSimulationEveryBlocks
                                    )
                        if milking_iteration == 0:
                            sim.local_energy_minimization()
                        else:
                            sim._apply_forces()

                        for lefIter in range(restartSimulationEveryBlocks):
                            if lefIter % saveEveryBlocks == saveEveryBlocks - 1: 
                                # do block on last set of MD steps for a set of LEF positions
                                sim.do_block(n_steps_per_int)
                            else:
                                sim.integrator.step(n_steps_per_int)
                            if lefIter < restartSimulationEveryBlocks - 1:
                                curBonds, pastBonds = milker.step(sim.context)
                        polymer = sim.get_data()
                        del sim
                        print(f'\n\n AA: {AA}\tBB: {BB}\t XX: {XX}\n {extrusion_conds}\n Simulation is percent complete: {round(100* (milking_iteration+1) / (n_lefIters // restartSimulationEveryBlocks), 2)} \n\n\n\n\n\n')
                        reporter.blocks_only = True
                        time.sleep(0.2)
                    reporter.dump_data()

                except BlockingIOError:
                    continue
                except KeyboardInterrupt:
                    print(traceback.format_exc())
                    interrupt = True
                    break
                except:
                    print(traceback.format_exc())
                finally:
                    traj_file.close()
                    os.remove(f'{save_path}/_in_progress_flag')

                if interrupt:
                    break
            if interrupt:
                break
        if interrupt:
            break
    if interrupt:
        break