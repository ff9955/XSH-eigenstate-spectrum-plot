from exciton_dipole_functions import *
import numpy as np

'''
python script that uses functions from exciton_dipole_functions module to sample adiabats from all geometries' Hamiltonians in the directory, and calculate their relative probabilities of excitation, via the photon probability, the transition dipole moment and the intrinsic probability of getting the geometry in the first place. Theese probabilites are then organised into a massive list, and are are randomly selected whilst being weighted by their relative total excitation probabilities.
'''


def gaussian_pulse(w, w_0, sigma_w):
    '''
    Function that produces a gaussian distribution of frequencies, given an energy range, a central frequency and a standard deviation
    '''

    p = np.exp(-(w - w_0)**2/(2*sigma_w**2))
    return p/np.sum(p)

trajectory_list = [t for t in range(0,1000)]

#active_PDI_list = [4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 28, 29, 30, 31, 32, 40, 41, 42, 43, 44, 52, 53, 54, 55, 56, 64, 65, 66, 67, 68, 76, 77, 78, 79, 80, 88, 89, 90, 91, 92, 100, 101, 102, 103, 104, 112, 113, 114, 115, 116]

active_PDI_list = [6, 18, 30, 42, 54, 66, 78, 90, 102, 114]

number_diabats = 60
number_excitons = 10
neutral_geometry_energies = np.loadtxt('total_neutral_energies.txt')
non_covalent_array = np.zeros(len(neutral_geometry_energies)-1)

counter = 0
chosen_adiabat_array = np.zeros((5, number_diabats*len(trajectory_list) ))

for trajectory in trajectory_list:
    print(trajectory)

    active_coord_list = get_active_molecules(f'run-fssh-{trajectory}/pos-init.xyz', (13203, 22802), active_PDI_list, 40, True)
    #getting list of arrays of xyz coords of electronically active PDI molecules

    tresp_array = get_TRESP_charges(f'run-fssh-{trajectory}/PTCDIH_S1_TRESP_CHARGES-crys-camb3lyp_6-31Gdp.include')
    #reading TRESP charges into an array

    dipole_array = get_molecular_dipoles(tresp_array, active_coord_list)
    #passing xyz components of all TRESP dipoles of the active molecules into a 2D array

    neutral_energy = neutral_geometry_energies[trajectory]
    #reading in the (e-phonon coupling) energy of the neutral ground state in the this trajectory's initial geometry, the files' values were obtained via reading the energy values of the position files of the initial trajectory coordinates. The coordinate files and trajectories in this directory are ordered the same way

    H = xyz_parse_first_section(f'run-fssh-{trajectory}/run-pseudo-hamilt-1.xyz', 3)
    H = build_sim_H(H, number_diabats)
    eigenvals, eigenvecs = get_eigen(H)
    #getting the eigenvals of the X-SH electronic Hamiltonian

    active_state_array = sh_log_parser(f'run-fssh-{trajectory}/run-sh-1.log')
    #getting 1D array of active state indices for every timestep in run

    initial_active_state = active_state_array[0]
    #taking initial active state index
    initial_state_energy = eigenvals[initial_active_state - 1]
    #using this to get the ELECTRONIC energy of X-SH

    xsh_energy_lines = xyz_parser(f'run-fssh-{trajectory}/run-1.ener',7)
    xsh_energy = float(xsh_energy_lines[0][4])
    #then getting the TOTAL X-SH energy from the first line of the ener file

    non_covalent_energy = xsh_energy - initial_state_energy
    non_covalent_array[trajectory] = non_covalent_energy
    #taking the intermolecular interaction energy as the difference between the total and electronic X-SH energies

    CT_energy = 0.086424
    #assigning the energy of the S1 excitation (2.75eV) MINUS the XT-CT offset, since this is already accounted for in the XT site energies

    for index in range(len(H)):
        if index <= 400:
            H[index,index] = H[index,index] - (neutral_energy - non_covalent_energy) + CT_energy
        else:
            H[index,index] = H[index,index] - (neutral_energy - non_covalent_energy) + CT_energy

            #shifting site energies such that XT site energy = S1 energy at R_eq of neutral ground state

    eigenvals, eigenvecs = get_eigen(H)
    eigenvals = eigenvals*27211.396641307998
    #re-diagonalising and converting to meV

    eigenstate_transition_elements = eigenstate_transition_dipoles(eigenvecs, dipole_array, number_excitons=10)
    eigenstate_transition_elements = np.sum(eigenstate_transition_elements**2, axis=1)

    photon_probabilities = gaussian_pulse(eigenvals, 3150, 47.52) 
    normalised_transition_elements = eigenstate_transition_elements/np.sum(eigenstate_transition_elements)

    adiabat_probabilities = photon_probabilities*normalised_transition_elements
    #getting the excitation probability of all given adiabats by multiplying the squared dipole moments by the photon probability distribution

    XT_characters = get_XT_characters(eigenvecs,10)
   
    chosen_adiabat_array[0, counter: counter + number_diabats] = XT_characters
    chosen_adiabat_array[1, counter: counter + number_diabats] = adiabat_probabilities
    chosen_adiabat_array[2, counter: counter + number_diabats] = eigenvals
    chosen_adiabat_array[3, counter: counter + number_diabats] = np.arange(1, number_diabats+1)
    chosen_adiabat_array[4, counter: counter + number_diabats] = np.ones(number_diabats)*trajectory
    #passing relevant info into an array with 5 rows; the second-last row is the adiabat index (+1), which we will use to select the active adiabat given the geometry; the last row is the geometry index, which we will use to load in the chosen initial geometry as the pos-init file in the trajectory.

    counter += number_diabats

chosen_adiabat_array[1] = chosen_adiabat_array[1]/np.sum(chosen_adiabat_array[1])
#renormalising the products of the excitation/photon probabilities 
np.savetxt('non_covalent_energies.txt', non_covalent_array)

selection_number = 500
random_selected_indices = np.random.choice(np.arange(len(chosen_adiabat_array[0])), size=selection_number, p=chosen_adiabat_array[1])
#randomly selecting the relevant geometries and active surfaces, and weighting them by the different probabilities

adiabat_file = open('phys_selected_geoms_adiabats.txt', 'a')

for index in random_selected_indices:
    adiabat_file.write(str(int(chosen_adiabat_array[4][index])) + ' ' + str(int(chosen_adiabat_array[3][index])) + '\n')

adiabat_file.close()
