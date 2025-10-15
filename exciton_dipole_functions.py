import numpy as np
from xsh_analysis_functions import *
from file_parsers import *

#this is a python script that calculates the transition dipole moment of the eigenstates of an FE or FE-CT Hamiltonian, by taking the linear combination of the diabatic state coefficients of each eigenstate, multiplied by the dipole moment of the molecule to which the diabatic state corresponds. The Hamiltonian must be specified manually either by writing out or loading in an array, or you can build the FE Hamiltonian (but not FE-CT) by using the function inside this file. The molecular dipoles are calculated by the sum of products of atomic positions and TRESP charges, so you need to read in the TrESP charges from an external file by using the get_TRESP_charges function below. You also need to read in a position file of the molecules involved in the Hamiltonian, which should be in the format of a standard xyz file. Instructions on how to use the functions is included in their comments and docstrings, and example is also given below.

def get_TRESP_charges(file_path):
    '''
    function that reads the generic file for tresp charges that is created by gaussian, into a 1D array of tresp charges

    Input: file_path: string, denotes path to tresp charge file

    Output: 1D array of each atom's TRESP charge
    '''

    tresp_array = xyz_parser(file_path, 4)
    #reading the whole TRESP charge file into a single array of strings

    tresp_array = tresp_array[:,-1]
    #then only taking the final column of this array, as it's this column that contains each atom's charge

    vector_float = np.vectorize(float)
    tresp_array = vector_float(tresp_array)
    #converting strings to floats

    tresp_array = tresp_array/np.sqrt(2)
    #re-scaling TRESP charges that is spit out by gaussian software

    return tresp_array

def get_active_molecules(position_file_path, text_lines, active_molecules, atoms_per_molecule, convert_to_bohr=False):
    '''
    Function that only reads the coordinates of specified molecules from an xyz file

    Input: position_file_path: string, path to position xyz file; text_lines: tuple of ints, lines numbers at which you want tostart and stop reading the position file ;active_molecules: iterable, indices of molecules you want read, where index 1 refers to 1st molecule shown in the file, 2 means the 2nd molecule shown, etc. ; atoms_per_molecule = int, number of atoms per molecule (all molecules assumed to have constant number of atoms); convert_to_bohr = bool, whether you want the coordinates in bohr or A

    Output: list of arrays, where each array is a 2D array of a chosen molecule's coordinates in bohr or A
    '''

    coordinate_array = xyz_parse_section(position_file_path, 4, text_lines[0], text_lines[1])
    coordinate_array = coordinate_array[:, 1:]
    #neglecting the 1st column as this is just atom types, which we don't need here

    vector_float = np.vectorize(float)
    coordinate_array = vector_float(coordinate_array)
    #converting coordinates to floats

    active_molecule_coordinates = []

    for molecule in active_molecules:

        molecule_index = molecule - 1
        #take one away from the index, as the first molecule will begin at the 0th row, not the 1st
        
        initial_line = molecule_index*atoms_per_molecule
        final_line = initial_line + atoms_per_molecule
        #specifying the rows of the total coord array we want to slice to get the indexed molecule

        indexed_coordinates = coordinate_array[initial_line:final_line]
        active_molecule_coordinates.append(indexed_coordinates)
        #now that we have an array of just the indexed molecule, we append this array to a list

    ang_bohr_conversion = 1.889725989
    if convert_to_bohr == True: 
        active_molecule_coordinates = [coords*ang_bohr_conversion for coords in active_molecule_coordinates]
        #list comprehension that just re-defines the list as all the arrays multiplied by the conversion factor

    return active_molecule_coordinates

def get_molecular_dipoles(tresp_charge_array, molecule_coordinate_list):
    '''
    Function that uses the tresp charges of atoms and molecules' coordinates, to get molecules' dipole moments

    Input: tresp_charge_array: 1D array of atoms' tresp charges; molecule_coordinate_list: list of 2D arrays, where each array contains the cartiesian coordinates of all atoms of chosen molecules (the order of atoms in along arrays' columns is the same)

    Output: 2D array where each row refers to a single molecule, and the 3 columns refer to the x/y/z-components of the dipole moments of that molecule
    '''

    tresp_charge_array = tresp_charge_array[:, np.newaxis]
    #converting 1D array to a column vector so we can use np.multiply

    number_active_molecules = len(molecule_coordinate_list)
    molecule_dipole_moments = np.zeros((number_active_molecules,3))
    #array of dipole elements, 3 columns for x/y/z-components, and n rows for n molecules in molecule_coordinate_list

    for counter, coords in enumerate(molecule_coordinate_list):

        tresp_coord_product = np.multiply(coords, tresp_charge_array)
        #multiplying the x/y/z-coordinates of each atom, by that atom's corresponding TRESP charge

        single_dipole_moment = np.sum(tresp_coord_product, axis=0)
        #then getting the molecule's overlall dipole moment by summing over the products of atomic charges and coordinates

        single_dipole_moment = single_dipole_moment/0.393456
        #converting dipole moment from a.u to debye
        
        molecule_dipole_moments[counter]  = single_dipole_moment
        #then setting a given row of the dipole moment array equal to this molecule's three components.

        #therefore, the order of dipole moments in the resulting array is the same as the order of molecule indices specified in the active_molecule iterable

    return molecule_dipole_moments

def build_excitonic_H(tresp_array, molecule_coordinate_list, eV=False):
    '''
    Function that builds a Hamiltonian matrix of frenkel excitons, given the nuclear coordinates of the active molecules and the atomic TRESP charges of all atoms in the molecule

    Input: tresp_array: 1D array of atomic TRESP charges; molecule_coordinate_list: list of 2D arrays, where each array contains the cartesian coordinates of a molecule inside the system; eV=boolean, specify whether you want the matrix elements in eV or Hartrees

    Output: 2D Hamiltonian matrix, where diagonal elements refer to the energies of excitons localised on single molecules (assumed to be zero here), and off-diagonals refer to electronic couplings between different basis states
    '''
    
    number_basis_states = len(molecule_coordinate_list)
    H = np.zeros((number_basis_states, number_basis_states))
    #initialising the Hamiltonian matrix

    for molecule_index in range(number_basis_states):
        #grabbing the index of the first active molecule

        molecule_coords = molecule_coordinate_list[molecule_index]

        for other_index in range(molecule_index+1,number_basis_states):
            #then grabbing the indices of all the molcules coming after it in the index list

            other_coords = molecule_coordinate_list[other_index]
            tresp_coupling = 0

            for index, coord in enumerate(molecule_coords):
                for index2, coord2 in enumerate(other_coords):
                #itrating over all possible combinations of atomic coordinates of the two molecules
                   
                    nuclear_distance = coord - coord2
                    #calculating euclidean distance between two chosen atoms
                    nuclear_distance = np.sqrt(np.sum(nuclear_distance**2))
                    
                    charge_product = tresp_array[index]*tresp_array[index2]
                    #calculating the product of tresp charges of the two chosen atoms

                    tresp_coupling += charge_product/nuclear_distance
                    #then calculating the coulombic interaction between the charges of the atoms, and adding this to the total tresp coupling between the two chosen molecules

            H[molecule_index,other_index] = tresp_coupling
            #assigning the tresp coupling between the two molecules to the appropariate off-diagonal element

    H = H + H.T
    #the Hamiltonian matrix is symmetric, so I have only calculated the upper triangle, and I am assigining the (identical but reflected) lower triangle by adding the transposes together

    if eV == True: H = H*27.2   #possible conversion to eV from atomic units

    return H

def get_eigen(matrix):
    '''
    function computes energies and eigenvectors of eigenstates in increasing order of energy

    input: square Hamiltonian matrix

    output: row vector (length D) of eigen energies, matrix (DxD) of orthonormal eigenvectors
    where each column is a single eigenvector
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    idx = eigenvalues.argsort()

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def eigenstate_transition_dipoles(eigenvector_array, dipole_moment_array, number_excitons):
    '''
    function that calculates the transition dipole moments of all eigenvectors resulting from a given Hamiltonian, using the dipole moments of individual molecules, and the expansion coefficients of the eigenvector components

    Input: eigenvector_array: 2D array of eigenvectors of the Hamiltonian, where the eigenvectors are given by the columns of the array; dipole_moment_array: 2D array, where each row contains the 3 x/y/z-components of the dipole moment of a given molecule; number_excitons: int, number of exciton diabats in your system
    '''

    number_eigenstates = len(eigenvector_array)

    eigenstate_dipole_elements = np.zeros((number_eigenstates,3))
    #preparing the array that will store the components of each eigenstate's transition dipole moment

    eigenvector_transpose = eigenvector_array.T

    for index, vector  in enumerate(eigenvector_transpose):

        exciton_vector = vector[-number_excitons:]
       #ignoring all diabats apart from the excitons

        column_vector = exciton_vector[:, np.newaxis]
        #re-defining the exciton coefficient array as a column vector

        dipole_coefficient_product = np.multiply(dipole_moment_array, column_vector)
        #multiplying the (signed) expansion coefficient of an exciton diabat, by the three components of the transition dipole moment of the molecule corresponding to that diabat

        total_eigenstate_dipole = np.sum(dipole_coefficient_product, axis=0)
        #now that we have the products of each diabat's coefficient with its associated transition dipole moment, we can now compute the sum of all these products to the get the eigenstate's total transition dipole moment        

        eigenstate_dipole_elements[index] = total_eigenstate_dipole
        #then we insert the three components of this eigenstate's transition dipole moment into the array intialised at the beginning of the function

    return eigenstate_dipole_elements

def get_XT_characters(eigenvector_array, number_excitons):
    '''
    Function that calculates the relative populations of excitonic basis states for each eigenvector in the array of eigenvectors you must pass in

    Input: eigenvector_array: 2D array where each column corresponds to an eigenvector of a matrix; number_excitons: int, the number of excitonic basis states that make up your electronic wavefunction

    Output: 1D array of floats, where each element corresponds to the relative excitonic population of an eigenstate; the order of populations in this row vector is the same as the order of eigenvector columns in the array you pass in
    '''

    eigenvec_transpose = eigenvector_array.T
    number_eigenstates = len(eigenvector_array)

    eigenstate_XT_characters = np.zeros(number_eigenstates)
    #initialising output array, whose length has to be equal to the number of eigenvectors in the input array
    
    for index, vector in enumerate(eigenvec_transpose):

        exciton_vector = vector[-number_excitons:]
        #the last N elements of the eigenvector column correspond to the N excitonic basis states, due to my assumption of the structure of the Hamiltonian that is passed in

        XT_character = np.sum(exciton_vector**2)
        #square each expansion coefficient to get the population
        eigenstate_XT_characters[index] = XT_character

    return eigenstate_XT_characters

def divide_states(eigenvectors, coulomb_barrier_path, ct_distance_path):
    '''
    Function that divides the diabatic populations of each eigenvector in an array into iCT, niCT, CSS and XT components and returns of the sum of these diabatic subsets' populations for all eigenvectors in the array
    '''

    diabatic_populations = eigenvectors**2

    vector_float = np.vectorize(float)
    #reading coulomb barrier elements and associated e-h distances into 1D np arrays
    coulomb_array = xyz_parser(coulomb_barrier_path, 3)
    coulomb_array = vector_float(coulomb_array[:,-1])

    number_excitons = len(eigenvectors) - len(coulomb_array)

    distance_array = xyz_parser(ct_distance_path, 3)
    distance_array = vector_float(distance_array[:,-1])

    #getting XT population of each vector: M_XT states after CT-block in each column
    exciton_population = np.sum(diabatic_populations[-number_excitons,:], axis=0)

    #iCT states will have lowest possible values of coulomb barrier
    sorted_coulomb = np.sort(coulomb_array)
    ict_indices = np.where(coulomb_array == sorted_coulomb[0])[0]
    #get these indices from barrier minima, apply to the eigenvector array columns
    ict_population = np.sum(diabatic_populations[ict_indices,:], axis=0)

    #CSS = all diabats to the right of the barrier peak
    barrier_max = np.where(coulomb_array == max(coulomb_array))[0][0]
    max_distance = distance_array[barrier_max]
    css_indices = np.where(distance_array > max_distance)[0]

    css_population = np.sum(diabatic_populations[css_indices,:], axis=0)
    #therefore niCT is just the remainder of the 3 aforementioned state types
    
    nict_population = 1 - exciton_population - ict_population - css_population

    return exciton_population, ict_population, nict_population, css_population
