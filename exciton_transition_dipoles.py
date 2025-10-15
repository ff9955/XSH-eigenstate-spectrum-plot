from exciton_dipole_functions import *

#CALCULATION OF EIGENSTATES AND THEIR TRANSITION DIPOLES, FROM INITIAL HAMILTONIANS OF SPECIFIED TRAJECTORIES
###############################################################################################################

#initialising the trajectories we want, the decomp indices of the active PDI molecules, and other specific details about the system we're simulating

#a lot of the the comments about what each function is doing is included in the get_adiabat_pulse.py script, or in the comments in the exciton_dipole_functions module

trajectory_list = [t for t in range(0,1000)]
#active_PDI_list = [6,18,30,42,54,66,78,90,102,114,126,138,150,162,174,186,198,210,222,234]
active_PDI_list = [6,18,30,42,54,66,78,90,102,114]

number_diabats = 61
neutral_geometry_energies = np.loadtxt('total_neutral_energies.txt')

transition_dipole_array = np.zeros((2, number_diabats*len(trajectory_list) ))
eigen_character_array = np.zeros((5, number_diabats*len(trajectory_list)) )

single_observable_array = np.zeros((2, number_diabats))
dominant_dipole_array = np.zeros((2,1))

#different arrays for storing the transition dipole moments of all adiabats of all the selected geometries

counter = 0

for trajectory in trajectory_list:
    print(trajectory)

    active_coord_list = get_active_molecules(f'run-fssh-{trajectory}/pos-init.xyz', (13203, 22802), active_PDI_list, 40, True)

    tresp_array = get_TRESP_charges(f'run-fssh-{trajectory}/PTCDIH_S1_TRESP_CHARGES-crys-camb3lyp_6-31Gdp.include')

    dipole_array = get_molecular_dipoles(tresp_array, active_coord_list)

    neutral_energy = neutral_geometry_energies[trajectory]

    H = xyz_parse_first_section(f'run-fssh-{trajectory}/run-pseudo-hamilt-1.xyz', 3)
    H = build_sim_H(H, number_diabats)
    eigenvals, eigenvecs = get_eigen(H)

    active_state_array = sh_log_parser(f'run-fssh-{trajectory}/run-sh-1.log')

    initial_active_state = active_state_array[0]
    initial_state_energy = eigenvals[initial_active_state - 1]

    xsh_energy_lines = xyz_parser(f'run-fssh-{trajectory}/run-1.ener',7)
    xsh_energy = float(xsh_energy_lines[0][4])

    non_covalent_energy = xsh_energy - initial_state_energy
    #extra total potential energy due to non-covalent interactions between molecules

    CT_energy = 0.086424
    XT_non_eq_energy = 0.00733874
    CT_non_eq_energy = 0.008690

    #shifting all site energies upwards by iCT excitation energy (CT_energy) obtained from TD-DFT and diabatisation of 6T-PDI dimer.
    #subtracting energy of neutral GS in same configuration since we want diabatic XT energies to equal the PDI S1 energy in vacuum.
    #non_covalent_energy subtracted from neutral bc it's not included in site energies but is included in total neutral energies.
    for index in range(len(H)):
        if index <= 400:
            H[index,index] = H[index,index] - (neutral_energy - non_covalent_energy) + CT_energy # - CT_non_eq_energy
        else:
            H[index,index] = H[index,index] - (neutral_energy - non_covalent_energy) + CT_energy # - XT_non_eq_energy

    #The commented out "non_eq" terms were incorrectly included, as I though the different FF-terms of the XT/CT-states are defined wrt different energy references (due to the different equilibrium bond lengths), but it turns out that all e-phonon couplings in X-SH are defined relative to the R_eq in the neutral ground state, per Wei-Tao

    eigenvals, eigenvecs = get_eigen(H)

    XT_characters = get_XT_characters(eigenvecs,number_excitons=10)
    xt_populations, ict_populations, nict_populations, css_populations = divide_states(eigenvecs, 'FRZ_DIAGONALS_COULOMB.include', 'CT_DISTANCES.include')

    eigenstate_transition_elements = eigenstate_transition_dipoles(eigenvecs, dipole_array, number_excitons=10)
    eigenstate_transition_elements = np.sum(eigenstate_transition_elements**2, axis=1)

    #PASSING ALL ADIABATS' TRANSITION DIPOLES INTO AN ARRAY

    transition_dipole_array[0, counter : counter + number_diabats] = eigenstate_transition_elements
    transition_dipole_array[1, counter : counter + number_diabats] = eigenvals

    eigen_character_array[0, counter : counter + number_diabats] = xt_populations
    eigen_character_array[1, counter : counter + number_diabats] = ict_populations
    eigen_character_array[2, counter : counter + number_diabats] = nict_populations
    eigen_character_array[3, counter : counter + number_diabats] = css_populations
    eigen_character_array[4, counter : counter + number_diabats] = eigenvals

    #TAKING TRANSITION DIPOLES THAT ARE THE LARGEST FOR THIS SPECIFIC TRAJECTORY

    single_observable_array[0,:] = eigenstate_transition_elements
    single_observable_array[1,:] = eigenvals
    #passing the eigenvals and transition dipoles of a single trajectory into this "single_observable_array"

    TD_sorted_indices = single_observable_array[0].argsort()
    TD_sorted_dipole_array = single_observable_array[:, TD_sorted_indices]
    #sorting it wrt dipole, then reversing it so that the biggest dipoles come first

    TD_sorted_dipole_array = np.fliplr(TD_sorted_dipole_array)

    normalised_dipoles = TD_sorted_dipole_array[0]/np.sum(TD_sorted_dipole_array[0])
    cumulative_dipoles = np.cumsum(normalised_dipoles)
    #cumulative dipoles were used to take the first n dipole elements that make up ~0.9 of the sum
    
    #majority_index = np.where(normalised_dipoles > 0.2)[0]
    majority_index = np.array([0,1])
    #this index is just used to take the top 2 elements of the single_observable array
    
    dominant_dipole_array = np.hstack((dominant_dipole_array, TD_sorted_dipole_array[:,majority_index]))
    counter += number_diabats

dominant_dipole_array = dominant_dipole_array[:,1:]
#removing the first redundant column from the array containig the highest dipoles
energy_sorted_indices = transition_dipole_array[1].argsort()
#sorting elements by energy now, so we can easily plot the spectrum later

energy_sorted_transition_dipole_array = transition_dipole_array[:,energy_sorted_indices]
#energy_sorted_dominant_dipoles = dominant_dipole_array[:, energy_sorted_indices]
energy_sorted_character_array = eigen_character_array[:, energy_sorted_indices]

np.savetxt('transition_dipoles_rel-to-neutral-e5_2xCT.txt', energy_sorted_transition_dipole_array)
np.savetxt('eigen_characters_rel-to-neutral-e5_2xCT.txt', energy_sorted_character_array)
