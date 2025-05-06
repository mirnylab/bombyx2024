import pickle
import os
import time
import sys
import numpy as np
import polychrom
from polychrom import polymerutils
from polychrom import simulation, starting_conformations, forces, forcekits
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
import openmm
import shutil
import extrusionlib as el
from extrusionlib.bond_propagation import extrusionPropagator
import traceback
import warnings
import h5py 
import glob
import argparse
sys.path.append("/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/extrusion_and_compartments")
                
import manySphericalSpheres

parser = argparse.ArgumentParser(description='Loop extrusion and Compartment Simulator')
parser.add_argument('-g', '--GPU', help='GPU number to use', type=str)
parser.add_argument('-o', '--out_path', default='Xboundaries', help='location where simulation save folder is created')
args = parser.parse_args()

def initiate_sim(polymer, r, cell_size, chains, GPU, ixn_mtx, reporter): #r, cell_size
        
    sim = simulation.Simulation(
        platform="CUDA",
        GPU=GPU,
        integrator="variableLangevin",
        error_tol=0.003,
        collision_rate=0.03,
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
                "k": 1.5,
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
with open('/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/polymer_info_extrusion.pkl','rb') as monInfo_file:
    monInfo = pickle.load(monInfo_file)
n_per_chain = monInfo['L']
n_chains = 30
chain_list = n_chains*[n_per_chain]
N = sum(chain_list)

if n_chains == 1:
    chains = [(0, None, False)]
else:
    chains = [(i*n_per_chain, (i+1)*n_per_chain, False) for i in range(n_chains)]
    
# LEF params
# Defining one lambda for all LEFs and different d's based on compartment type
n_steps = 500_000
lambda_ = 100
dX_list = [50, 100, 200, 400, 1000]
traj_fh = f"LEFpositions_steps-{n_steps}_nChains-{n_chains}_Xboundaries.h5" 
    
# Block copol params
# load compartment labels
comp_ids = monInfo['compartment_ID']
mon_types = np.tile(comp_ids, n_chains)

AB = XA = XB = 0.00
B_self_attr = [0, 0.025, 0.05, 0.1, 0.15, 0.2]
X_self_attr = XX = 0.00 

# Other things
lef_wiggle_dist_unscaled = 0.2
lef_bond_dist_unscaled = 0.5

n_lefIters_per_milking = 100
n_ints_per_lefIter = 10
n_steps_per_int = 750
n_lefIters = 500_000
max_data_length = 100
T_blocks = int(n_lefIters/n_ints_per_lefIter)
    
# Spherical Confinement
# Parameters for initializing conformations and positioning each 5 chains into its own sphere
density = 0.2
n_chains_per_sphere = 10
r = (3 * n_per_chain * n_chains_per_sphere / (4 * 3.141592 * density)) ** (1/3)
cell_size = 5*r
offsets = [[cell_size*i,0,0] for i in range(int(n_chains/n_chains_per_sphere))]
offsets_multi = np.repeat(np.array(offsets), int(n_per_chain*n_chains_per_sphere), axis=0)

base_path = '/net/levsha/share/emily/notebooks/sims/bombyx/toy_models/extrusion_and_compartments/extrusion_density_sims'

interrupt = False
for d_X in dX_list:
    d_AB = d_X
    d_dir = f'lambda-{lambda_}_dX-{d_X}_dAB-{d_AB}'
    traj_path = f'{base_path}/{d_dir}'
    traj_file = h5py.File(f'{traj_path}/{traj_fh}', 'r')
    lef_positions = traj_file["positions"]
    n_lefs = traj_file.attrs["n_lefs"]

    for BB in B_self_attr:
        AA = BB
        ixn_mtx = np.array([[AA, AB, XA],
                           [AB, BB, XB],
                           [XA, XB, XX],
                          ])
        save_path = f'{base_path}/{d_dir}/AA{AA:.2f}_BB{BB:.2f}_XX{XX:.2f}/{args.out_path}'

        os.makedirs(save_path, exist_ok=True)
        print(f'\n\nAA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}\n{d_dir}')
        try:
            with open(f'{save_path}/_in_progress_flag', 'x') as fp:
                pass       
        except FileExistsError:
            print("\tSimulation is currently running on another GPU\n\n")
            continue
        reporter = HDF5Reporter(folder=save_path, max_data_length=100, 
                                overwrite=False, check_exists=False, blocks_only=False
                               )

        try:
            if os.path.exists(f'{save_path}/blocks_{T_blocks-max_data_length}-{T_blocks-1}.h5'):
                print("\tSimulation is complete\n\n")
                continue
            confs_starting = np.concatenate([np.array(starting_conformations.grow_cubic(n_per_chain*n_chains_per_sphere, 
                                                                    int((n_per_chain*n_chains_per_sphere)**(1/3)+2))) 
                         for i in range(int(n_chains/n_chains_per_sphere))])
            polymer = confs_starting+offsets_multi
            milker = extrusionPropagator([lef_positions])
            for milking_iteration in range(n_lefIters//n_lefIters_per_milking):
                sim = initiate_sim(polymer, r=r, cell_size=cell_size, chains=chains, GPU=args.GPU, ixn_mtx=ixn_mtx, reporter=reporter)

                kbond = sim.kbondScalingFactor / (lef_wiggle_dist_unscaled ** 2)
                lef_bond_dist = lef_bond_dist_unscaled * sim.length_scale

                activeParams = {"length":lef_bond_dist, "k":kbond}
                inactiveParams = {"length":lef_bond_dist, "k":0}
                milker.setParams(activeParams, inactiveParams)
                milker.setup(bondForce=sim.force_dict['harmonic_bonds'],
                             blocks=n_lefIters_per_milking
                            )
                if milking_iteration == 0:
                    sim.local_energy_minimization()
                else:
                    sim._apply_forces()

                for lefIter in range(n_lefIters_per_milking):
                    if lefIter % n_ints_per_lefIter == n_ints_per_lefIter - 1:
                        sim.do_block(n_steps_per_int)
                    else:
                        sim.integrator.step(n_steps_per_int)
                    if lefIter < n_lefIters_per_milking - 1:
                        curBonds, pastBonds = milker.step(sim.context)
                polymer = sim.get_data()
                del sim
                print(f'\n\n AA: {AA:.2f}\tBB: {BB:.2f}\tXX: {XX:.2f}\n {traj_path.split("/")[-1]}\n Simulation is percent complete: {round(100* (milking_iteration+1) / (n_lefIters // n_lefIters_per_milking), 2)} \n\n\n\n\n\n')
                reporter.blocks_only = True
                time.sleep(0.2)
            reporter.dump_data()
        except KeyboardInterrupt:
            print(traceback.format_exc())
            interrupt = True
            break
        except:
            print(traceback.format_exc())
        finally:
            os.remove(f'{save_path}/_in_progress_flag')
            reporter.dump_data()

        if interrupt:
            break
    if interrupt:
        break
    traj_file.close()
