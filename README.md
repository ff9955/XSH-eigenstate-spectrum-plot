Set of python modules and scripts for calculating absorption spectrum from the electronic structure of the X-SH non-adiabatic dynamics program.

The exciton_dipole_functions.py module contains functions for reconstructing the electronic Hamiltonian of X-SH for a set
of nuclear configurations, and obtaining a set of observables associated with each eigenstate including: transition dipole 
moment, exciton/charge-transfer character, energy with respect to ground state.

The exciton_transition_dipoles.py script calls functions from exciton_dipole_functions.py to calculate the properties above 
for a system of 10-50 acceptor molecules.

The get_adiabats_pulse.py script uses first-order time-dependent perturbation theory to calculate the excitation probability of 
each eigenstate of ensemble of them given a time-dependent pulse of light. It the randomly samples states from the ensemble based 
on their relative excitation probabilities.
