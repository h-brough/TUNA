import numpy as np
from tuna_util import *
from tuna_calc import Calculation
from tuna_molecule import Molecule
from tuna_dft import calculate_restricted_exchange_correlation_kernel_matrix, calculate_unrestricted_exchange_correlation_kernel_matrix



"""

This is the TUNA module for configuration interaction, written first for version 0.6.0 of TUNA and largely rewritten for version 0.11.0.

Methods needed to transform the atomic orbital basis molecular integrals to either the spatial orbital or spin orbital basis are stored here,
as well as functions to calculate excited states with configuration interaction singles and time-dependent Hartree-Fock.

The module contains:

1. Some utility functions to transform molecular integrals (spin_block_core_Hamiltonian, build_spin_orbital_Fock_matrix, etc.)
2. Functions to transform molecular integrals (begin_spatial_orbital_calculation, begin_spin_orbital_calculation, etc.)
3. Functions to build the CIS Hamiltonian, diagonalise it and print out excited state information (calculate_weights_matrix, print_CIS_absorption_spectrum, etc.)

"""




def spin_block_core_Hamiltonian(H_core: ndarray) -> ndarray:

    """
    
    Spin blocks core Hamiltonian.

    Args:  
        H_core (array): Core Hamiltonian in AO basis
    
    Returns:
        H_core_spin_block (array): Spin blocked core Hamiltonian in AO basis
    
    """

    H_core_spin_block = np.kron(np.eye(2), H_core)

    return H_core_spin_block










def build_spin_orbital_Fock_matrix(H_core_SO: ndarray, g: ndarray, o: slice) -> ndarray:

    """
    
    Builds Fock matrix in SO basis.

    Args:  
        H_core_SO (array): Core Hamiltonian in SO basis
        g (array): Antisymmetrised ERI in SO basis
        o (slice): Occupied spin orbitals slice
    
    Returns:
        F_SO (array): Fock matrix in SO basis
    
    """


    F_SO = H_core_SO + np.einsum("piqi->pq", g[:, o, :, o], optimize = True)

    return F_SO










def antisymmetrise_integrals(ERI: ndarray) -> ndarray:

    """

    Antisymmetrises two-electron integrals.

    Args:   
        ERI (array): Electron repulsion integrals in physicists' notation.

    Returns:
        ERI_ansym (array): Antisymmetrised electron repulsion integrals

    """

    ERI_ansym = ERI - ERI.transpose(0, 1, 3, 2)

    return ERI_ansym










def spin_block_molecular_orbitals(molecular_orbitals_alpha: ndarray, molecular_orbitals_beta: ndarray, epsilons: ndarray) -> ndarray:

    """

    Spin blocks alpha and beta molecular orbitals.

    Args:   
        molecular_orbitals_alpha (array): Alpha molecular orbitals in AO basis
        molecular_orbitals_beta (array): Beta molecular orbitals in AO basis
        epsilons (array): Orbital eigenvalues

    Returns:
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis

    """

    C_spin_block = np.block([[molecular_orbitals_alpha, np.zeros_like(molecular_orbitals_beta)], 
                             [np.zeros_like(molecular_orbitals_alpha), molecular_orbitals_beta]])
    
    C_spin_block = C_spin_block[:, epsilons.argsort()] 

    return C_spin_block










def transform_ERI_AO_to_SO(ERI_AO: ndarray, C_1: ndarray, C_2: ndarray, calculation: Calculation, silent: bool) -> ndarray:

    """

    Transforms electron repulsion integrals from the AO basis to the SO basis.

    Args:   
        ERI_AO (array): Electron repulsion integrals in AO basis
        C_1 (array): Molecular orbitals in AO basis
        C_2 (array): Molecular orbitals in AO basis
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        ERI_SO (array): Electron repulsion integrals in SO basis

    """
    
    timer("Molecular orbital transformation", 0)
    
    log("\n Transforming integrals step 1 of 4...       ", calculation, 1, end = "", silent = silent)
    
    # The stepwise transformation is faster, since NumPy doesn't have to look for the best contraction order

    temp_mnks = np.einsum("mknl,ls->mnks", ERI_AO, C_1, optimize = True)

    log("[Done]", calculation, 1, silent = silent)

    log(" Transforming integrals step 2 of 4...       ", calculation, 1, end = "", silent = silent)

    temp_mnrs = np.einsum("mnks,kr->mnrs", temp_mnks, C_2, optimize = True)

    log("[Done]", calculation, 1, silent = silent)

    log(" Transforming integrals step 3 of 4...       ", calculation, 1, end = "", silent = silent)

    temp_mqrs = np.einsum("mnrs,nq->mqrs", temp_mnrs, C_1, optimize = True)
    
    log("[Done]", calculation, 1, silent = silent)
    
    # The spin orbital two-electron integrals are in physicists' notation <pq|rs>

    log(" Transforming integrals step 4 of 4...       ", calculation, 1, end = "", silent = silent)

    ERI_SO = np.einsum("mqrs,mp->pqrs", temp_mqrs, C_2, optimize = True)
    
    log("[Done]\n", calculation, 1, silent = silent)

    timer("Molecular orbital transformation", 1)

    return ERI_SO










def transform_ERI_AO_to_MO(ERI_AO: ndarray, C: ndarray, calculation: Calculation, silent: bool) -> ndarray:

    """

    Transforms electron repulsion integrals from the AO basis to the SO basis.

    Args:   
        ERI_AO (array): Electron repulsion integrals in AO basis
        C (array): Molecular orbitals in AO basis
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        ERI_MO (array): Electron repulsion integrals in MO basis

    """

    timer("Molecular orbital transformation", 0)

    # The stepwise transformation is faster, since NumPy doesn't have to look for the best contraction order
    
    log("\n Transforming integrals step 1 of 4...       ", calculation, 1, end = "", silent = silent)

    # The atomic orbital two-electron integrals are in interleaved chemists' notation (mn|kl)

    temp_mnks = np.einsum("mknl,ls->mnks", ERI_AO, C, optimize = True)

    log("[Done]", calculation, 1, silent = silent)

    log(" Transforming integrals step 2 of 4...       ", calculation, 1, end = "", silent = silent)

    temp_mnrs = np.einsum("mnks,kr->mnrs", temp_mnks, C, optimize = True)

    log("[Done]", calculation, 1, silent = silent)

    log(" Transforming integrals step 3 of 4...       ", calculation, 1, end = "", silent = silent)

    temp_mqrs = np.einsum("mnrs,nq->mqrs", temp_mnrs, C, optimize = True)
    
    log("[Done]", calculation, 1, silent = silent)
    
    log(" Transforming integrals step 4 of 4...       ", calculation, 1, end = "", silent = silent)

    # The spatial orbital two-electron integrals are in interleaved chemists' notation (pr|qs)

    ERI_MO = np.einsum("mqrs,mp->prqs", temp_mqrs, C, optimize = True)
    
    log("[Done]", calculation, 1, silent = silent)

    timer("Molecular orbital transformation", 1)

    return ERI_MO










def build_singles_epsilons_tensor(epsilons: ndarray, o: slice, v: slice, level_shift: float = 0) -> ndarray:

    """

    Builds inverse epsilon tensor with shape ia.

    Args:   
        epsilons (array): Orbital eigenvalues
        o (slice): Occupied slice
        v (slice): Virtual slice
        level_shift (float, optional): Level shift

    Returns:
        e_ia (array): Inverse epsilons tensor for single excitations with shape ia

    """

    n = np.newaxis

    try:

        e_ia = 1 / (epsilons[o, n] - epsilons[n, v] - level_shift)
    
    except MemoryError:

        error("Not enough memory to build singles denominator!")

    return e_ia










def build_doubles_epsilons_tensor(epsilons_1: ndarray, epsilons_2: ndarray, o_1: slice, o_2: slice, v_1: slice, v_2: slice, level_shift: float = 0) -> ndarray:

    """

    Builds inverse epsilon tensor with shape ijab.

    Args:   
        epsilons_1 (array): Orbital eigenvalues
        epsilons_2 (array): Orbital eigenvalues
        o_1 (slice): Occupied slice
        o_2 (slice): Occupied slice
        v_1 (slice): Virtual slice
        v_2 (slice): Virtual slice
        level_shift (float, optional): Level shift

    Returns:
        e_ijab (array): Inverse epsilons tensor with shape ijab

    """

    n = np.newaxis

    try:

        e_ijab = 1 / (epsilons_1[o_1, n, n, n] + epsilons_2[n, o_2, n, n] - epsilons_1[n, n, v_1, n] - epsilons_2[n, n, n, v_2] - 2 * level_shift)
    
    except MemoryError:

        error("Not enough memory to build doubles denominator!")

    return e_ijab










def build_triples_epsilons_tensor(epsilons: ndarray, o: slice, v: slice, level_shift: float = 0) -> ndarray:

    """

    Builds inverse epsilon tensor with shape ijkabc.

    Args:   
        epsilons (array): Orbital eigenvalues
        o (slice): Occupied slice
        v (slice): Virtual slice
        level_shift (float, optional): Level shift

    Returns:
        e_ijkabc (array): Inverse epsilons tensor with shape ijkabc

    """

    n = np.newaxis

    try:

        e_ijkabc = 1 / (epsilons[o, n, n, n, n, n] + epsilons[n, o, n, n, n, n] + epsilons[n, n, o, n, n, n] - epsilons[n, n, n, v, n, n] - epsilons[n, n, n, n, v, n] - epsilons[n, n, n, n, n, v] - 3 * level_shift)
    
    except MemoryError:

        error("Not enough memory to build triples denominator!")

    return e_ijkabc










def build_quadruples_epsilons_tensor(epsilons: ndarray, o: slice, v: slice, level_shift: float = 0) -> ndarray:

    """

    Builds inverse epsilon tensor with shape ijklabcd.

    Args:   
        epsilons (array): Orbital eigenvalues
        o (slice): Occupied slice
        v (slice): Virtual slice
        level_shift (float, optional): Level shift

    Returns:
        e_ijklabcd (array): Inverse epsilons tensor with shape ijklabcd

    """

    n = np.newaxis

    try:

        e_ijklabcd = 1 / (epsilons[o, n, n, n, n, n, n, n] + epsilons[n, o, n, n, n, n, n, n] + epsilons[n, n, o, n, n, n, n, n] + epsilons[n, n, n, o, n, n, n, n] - epsilons[n, n, n, n, v, n, n, n] - epsilons[n, n, n, n, n, v, n, n] - epsilons[n, n, n, n, n, n, v, n] - epsilons[n, n, n, n, n, n, n, v] - 4 * level_shift)

    except MemoryError:

        error("Not enough memory to build quadruples denominator!")

    return e_ijklabcd










def build_MP2_t_amplitudes(g_oovv: ndarray, e_ijab: ndarray) -> ndarray:

    """

    Build MP2 t-amplitudes with shape ijab.

    Args:   
        g_oovv (array): Electron-repulsion integrals in SO basis or spatial orbital basis OOVV sliced
        e_ijab (array): Inverse epsilons tensor shape ijab

    Returns:
        t_ijab (array): Doubles t-amplitudes shape ijab

    """

    t_ijab = g_oovv * e_ijab

    return t_ijab










def transform_matrix_AO_to_SO(M: ndarray, molecular_orbitals: ndarray) -> ndarray:

    """

    Transforms two-index tensor from AO basis to SO basis.

    Args:   
        M (array): Matrix in AO basis
        molecular_orbitals (array): Molecular orbitals in AO basis

    Returns:
        M_SO (array): Matrix in SO basis

    """

    M_SO = np.einsum("mi,mn,na->ia", molecular_orbitals, M, molecular_orbitals, optimize = True)

    return M_SO










def transform_P_SO_to_AO(P_SO: ndarray, C_spin_block: ndarray, n_SO: int) -> tuple:

    """

    Transforms density matrix from SO basis to AO basis.

    Args:   
        P_SO (array): Density matrix in SO basis
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis
        n_SO (int): Number of spin orbitals

    Returns:
        P_AO (array): Density matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis

    """

    # Cuts spin-blocked molecular orbitals in two

    C_alpha = C_spin_block[:n_SO // 2, :]  
    C_beta = C_spin_block[n_SO // 2:, :] 

    # Transforms the alpha and beta parts of the total density matrix in the SO basis

    P_alpha = C_alpha @ P_SO @ C_alpha.T  
    P_beta = C_beta @ P_SO @ C_beta.T  

    P = P_alpha + P_beta

    return P, P_alpha, P_beta










def begin_spin_orbital_calculation(molecule: Molecule, ERI_AO: ndarray, SCF_output: Output, calculation: Calculation, silent: bool = False) -> tuple:

    """

    Calculates key factors for spin orbital calculations.

    Args:   
        ERI_AO (array): Two electron integrals in AO basis
        SCF_output (Output): Output object
        n_occ (int): Number of occupied spin orbitals

    Returns:
        g (array): Electron repulsion integrals in SO basis
        C_spin_block (array): Spin blocked molecular orbitals in AO basis
        epsilons_sorted (array): Sorted array of Fock matrix eigenvalues
        ERI_spin_block (array): Spin-blocked electron repulsion integrals in AO basis
        o (slice): Occupied spin orbitals
        v (slice): Virtual spin orbitals

    """
    
    # Defines occupied and virtual slices

    minimum_orbital = molecule.n_core_spin_orbitals if calculation.freeze_core else 0

    if molecule.n_core_spin_orbitals > molecule.n_electrons:

        error("Not enough spin orbitals to freeze!")

    if molecule.n_core_orbitals < 0:

        error("Cannot freeze a negative number of orbitals!")

    o = slice(minimum_orbital, molecule.n_occ)
    v = slice(molecule.n_occ, None)

    epsilons_combined = SCF_output.epsilons_combined

    log("\n Preparing transformation to spin orbital basis...", calculation, 1, silent)

    # Spin-blocks electron repulsion integrals

    ERI_spin_block = np.kron(np.eye(2), np.kron(np.eye(2), ERI_AO).T)

    # Spin-blocks molecular orbitals and transforms electron repulsion integrals

    C_spin_block = spin_block_molecular_orbitals(SCF_output.molecular_orbitals_alpha, SCF_output.molecular_orbitals_beta, epsilons_combined)
    
    ERI_SO = transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block, C_spin_block, calculation, silent)

    log(" Antisymmetrising two-electron integrals...  ", calculation, 1, silent, end = "")

    # Antisymmetrise electron repulsion integrals

    g = antisymmetrise_integrals(ERI_SO)

    log("[Done]", calculation, 1, silent)

    # Sorts epsilons

    epsilons_sorted = np.sort(epsilons_combined)

    # Tracks the spin state of each epsilon

    spin_labels = ["a"] * len(SCF_output.molecular_orbitals_alpha) + ["b"] * len(SCF_output.molecular_orbitals_beta)
    spin_labels_sorted = [spin_labels[i] for i in np.argsort(epsilons_combined)]

    def prefix_counts(seq):

        counts = {}
        result = []

        for x in seq:

            c = counts.get(x, 0)
            result.append(f"{c + 1}{x}")
            counts[x] = c + 1

        return result
    

    spin_orbital_labels_sorted = prefix_counts(spin_labels_sorted)


    if calculation.freeze_core and molecule.n_core_spin_orbitals != 0: 

        log(f"\n The {molecule.n_core_spin_orbitals} lowest energy spin orbitals will be frozen.", calculation, 1, silent)

    else:
        
        log(f"\n All electrons will be correlated.", calculation, 1, silent)
    


    return g, C_spin_block, epsilons_sorted, ERI_spin_block, o, v, spin_labels_sorted, spin_orbital_labels_sorted, ERI_SO










def begin_spatial_orbital_calculation(molecule: Molecule, ERI_AO: ndarray, SCF_output: Output, calculation: Calculation, silent: bool = False) -> tuple:

    """
    
    Sets up useful quantities for a spatial orbital calculation.

    Args:
        molecule (Molecule): Molecule object
        ERI_AO (array): Electron-repulsion integrals in AO basis
        SCF_output (Output): SCF Output object
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed
    
    Returns:
        g (array): Spatial orbital integrals in physicists' notation
        molecular_orbitals (array): Molecular orbitals in AO basis
        epsilons (array): Fock eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice

    """

    # Checks if orbitals need freezing

    minimum_orbital = molecule.n_core_orbitals if calculation.freeze_core else 0

    if molecule.n_core_orbitals * 2 > molecule.n_electrons:

        error("Not enough spatial orbitals to freeze!")

    if molecule.n_core_orbitals < 0:

        error("Cannot freeze a negative number of orbitals!")

    # Builds slices

    o = slice(minimum_orbital, molecule.n_doubly_occ)
    v = slice(molecule.n_doubly_occ, None)

    log("\n Preparing transformation to spatial orbital basis...", calculation, 1, silent)

    g = transform_ERI_AO_to_MO(ERI_AO, SCF_output.molecular_orbitals, calculation, silent)

    # Logs information about freezing orbitals

    if calculation.freeze_core and molecule.n_core_orbitals != 0: 

        log(f"\n The {molecule.n_core_orbitals} lowest energy orbitals will be frozen.", calculation, 1, silent)

    else:
        
        log(f"\n All electrons will be correlated.", calculation, 1, silent)


    return g, SCF_output.molecular_orbitals, SCF_output.epsilons, o, v










def calculate_oscillator_strengths(transition_dipoles: ndarray, excitation_energies: ndarray) -> ndarray:

    """

    Calculates the oscillator strengths of all states at once.

    Args:   
        transition_dipoles (array): Transition dipoles for all states
        excitation_energies (array): Excitation energies for all states

    Returns:
        oscillator_strengths (array): Oscillator strengths for states

    """
    
    oscillator_strengths = (2 / 3) * excitation_energies * transition_dipoles ** 2

    return oscillator_strengths










def calculate_restricted_singlet_A_matrix(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the singlet A matrix for a spin-restricted reference.

    Args:
        g (array): Two-electron integrals in physicists' spatial orbital notation
        epsilons (array): Molecular orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        A_ia_jb (array): Restricted singlet orbital Hessian matrix

    """

    # Builds the contributions to the A matrix

    A = 2 * g[o, o, v, v].transpose(0, 2, 1, 3) - g[o, v, o, v] * calculation.HFX_prop 

    if K_XC is not None:

        A = A + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of excitations

    A_ia_jb = A.reshape(A.shape[0] * A.shape[1], -1)

    # The diagonal elements also need the orbital energy differences, which are added on here

    A_ia_jb[np.diag_indices_from(A_ia_jb)] += (epsilons[v][None, :] - epsilons[o][:, None]).ravel()

    # Ensures matrix is Hermitian

    A_ia_jb = symmetrise(A_ia_jb)

    return A_ia_jb










def calculate_restricted_triplet_A_matrix(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the triplet A matrix for a spin-restricted reference.

    Args:
        g (array): Two-electron integrals in physicists' spatial orbital notation
        epsilons (array): Molecular orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        A_ia_jb (array): Restricted triplet orbital Hessian matrix

    """

    # Builds the contributions to the A matrix

    A = - g[o, v, o, v] * calculation.HFX_prop

    if K_XC is not None:

        A = A + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of excitations

    A_ia_jb = A.reshape(A.shape[0] * A.shape[1], -1)

    # The diagonal elements also need the orbital energy differences, which are added on here

    A_ia_jb[np.diag_indices_from(A_ia_jb)] += (epsilons[v][None, :] - epsilons[o][:, None]).ravel()

    # Ensures matrix is Hermitian

    A_ia_jb = symmetrise(A_ia_jb)

    return A_ia_jb










def calculate_unrestricted_A_matrix(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the orbital Hessian for a spin-unrestricted reference.

    Args:
        g (array): Antisymmetrised spin orbital two-electron integrals in physicists' notation
        epsilons (array): spin orbital molecular orbital eigenvalues
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        A_ia_jb (array): Unrestricted spin orbital Hessian matrix

    """

    # Builds the spin orbital contribution to the orbital Hessian

    A = g[v, o, o, v].transpose(2, 0, 1, 3) 

    if K_XC is not None:

        A = A + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of spin orbital excitations

    A_ia_jb = A.reshape(A.shape[0] * A.shape[1], -1)

    # The diagonal elements also need the spin orbital energy differences, which are added on here

    A_ia_jb[np.diag_indices_from(A_ia_jb)] += (epsilons[v][None, :] - epsilons[o][:, None]).ravel()

    # Ensures matrix is Hermitian

    A_ia_jb = symmetrise(A_ia_jb)

    return A_ia_jb










def calculate_restricted_singlet_B_matrix(g: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the singlet B matrix for a spin-restricted reference.

    Args:
        g (array): Two-electron integrals in physicists' spatial orbital notation
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        B_ia_jb (array): Restricted singlet B matrix

    """

    # Builds the contributions to the B matrix

    B = 2 * g[o, o, v, v].transpose(0, 2, 1, 3) - g[o, o, v, v].transpose(0, 3, 1, 2) * calculation.HFX_prop
    
    if K_XC is not None:

        B = B + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of excitations

    B_ia_jb = B.reshape(B.shape[0] * B.shape[1], -1)

    # Ensures matrix is Hermitian

    B_ia_jb = symmetrise(B_ia_jb)
    
    return B_ia_jb










def calculate_restricted_triplet_B_matrix(g: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the triplet B matrix for a spin-restricted reference.

    Args:
        g (array): Two-electron integrals in physicists' spatial orbital notation
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        B_ia_jb (array): Restricted triplet B matrix

    """

    # Builds the contributions to the B matrix

    B = - g[o, o, v, v].transpose(0, 3, 1, 2) * calculation.HFX_prop

    if K_XC is not None:

        B = B + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of excitations

    B_ia_jb = B.reshape(B.shape[0] * B.shape[1], -1)

    # Ensures matrix is Hermitian

    B_ia_jb = symmetrise(B_ia_jb)
    
    return B_ia_jb










def calculate_unrestricted_B_matrix(g: ndarray, o: slice, v: slice, calculation: Calculation, K_XC: ndarray = None) -> ndarray:

    """
    
    Calculates the B matrix for a spin-unrestricted reference.

    Args:
        g (array): Antisymmetrised spin orbital two-electron integrals in physicists' notation
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        calculation (Calculation): Calculation object
        K_XC (array, optional): Exchange-correlation kernel matrix 

    Returns:
        B_ia_jb (array): Unrestricted spin orbital B matrix

    """

    # Builds the spin orbital contribution to the B matrix

    B = g[v, v, o, o].transpose(2, 0, 3, 1)

    if K_XC is not None:

        B = B + K_XC * calculation.DFX_prop 

    # Reshapes into a matrix of spin orbital excitations

    B_ia_jb = B.reshape(B.shape[0] * B.shape[1], -1)

    # Ensures matrix is Hermitian

    B_ia_jb = symmetrise(B_ia_jb)
    
    return B_ia_jb










def build_orbital_hessian(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, hessian_type: str = "singlet", K_XC: ndarray = None, spin_labels: list = None) -> ndarray:
    
    """
    
    Constructs the orbital Hessian, for either restricted or unrestricted references.

    Args:
        g (array): Spatial orbital two-electron integrals in physicists' notation or antisymmetrised spin orbital integrals
        epsilons (array): Spatial or spin orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        hessian_type (str, optional): Type of Hessian to construct ("singlet" or "triplet") for restricted references
        K_XC (array, optional): Exchange-correlation kernel matrix
        spin_labels (list, optional): List of spin orbital labels
    
    Returns:
        H (array): Orbital Hessian matrix
    
    """
    
    timer("Orbital Hessian construction", 0)

    if calculation.reference == "UHF":   # Handles the spin-unrestricted case

        A = calculate_unrestricted_A_matrix(g, epsilons, o, v, calculation, K_XC)

        B = calculate_unrestricted_B_matrix(g, o, v, calculation, K_XC)

        # Screens out non-spin-conserving excitations

        spin_occupied = np.array(spin_labels)[o]
        spin_virtual  = np.array(spin_labels)[v]
    
        spin_conserving = (spin_occupied[:, None] == spin_virtual[None, :]).ravel()
    
        A = A[spin_conserving][:, spin_conserving]
        B = B[spin_conserving][:, spin_conserving]

    else:   # Handles the spin-restricted case
        
        if hessian_type == "triplet":   # Calculates the restricted triplet Hessian

            A = calculate_restricted_triplet_A_matrix(g, epsilons, o, v, calculation, K_XC)

            B = calculate_restricted_triplet_B_matrix(g, o, v, calculation, K_XC)

        else:   # Calculates the restricted singlet Hessian

            A = calculate_restricted_singlet_A_matrix(g, epsilons, o, v, calculation, K_XC)

            B = calculate_restricted_singlet_B_matrix(g, o, v, calculation, K_XC)

    # The Hessian is a block matrix with A and B submatrices

    H = np.block([[A, B], [B, A]])

    # Symmetrises orbital Hessian

    H = symmetrise(H)

    timer("Orbital Hessian construction", 1)

    return H










def perform_restricted_stability_analysis(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, silent: bool = False, K_XC_singlet: ndarray = None, K_XC_triplet: ndarray = None) -> None:
    
    """
    
    Performs a stability analysis for a spin-restricted reference.

    Args:
        g (array): Spatial orbital two-electron integrals in physicists' notation
        epsilons (array): Molecular orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging
        K_XC_singlet (array, optional): Singlet exchange-correlation kernel matrix
        K_XC_triplet (array, optional): Triplet exchange-correlation kernel matrix
    
    """

    # Builds the singlet and triplet orbital Hessians for RHF/RHF and RHF/UHF stability checks, respectively

    log("  Building singlet orbital Hessian...        ", calculation, 1, silent, end = "")

    H_singlet = build_orbital_hessian(g, epsilons, o, v, calculation, "singlet", K_XC_singlet)

    log("[Done]", calculation, 1, silent)

    log("  Building triplet orbital Hessian...        ", calculation, 1, silent, end = "")

    H_triplet = build_orbital_hessian(g, epsilons, o, v, calculation, "triplet", K_XC_triplet)
    
    log("[Done]", calculation, 1, silent)
    
    log("\n  Diagonalising orbital Hessians...          ", calculation, 1, silent, end = "")

    # Finds the lowest eigenvalues and corresponding eigenvectors of the singlet and triplet Hessians

    singlet_hessian_eigenvalues, _ = np.linalg.eigh(H_singlet)
    
    triplet_hessian_eigenvalues, _ = np.linalg.eigh(H_triplet)
    
    log("[Done]", calculation, 1, silent)

    log(f"\n  Lowest singlet eigenvalue:             {singlet_hessian_eigenvalues[0]:10.5f}", calculation, 1, silent)
    log(f"  Lowest triplet eigenvalue:             {triplet_hessian_eigenvalues[0]:10.5f}", calculation, 1, silent)

    # Checks for negative eigenvalues, indicating instabilities

    if singlet_hessian_eigenvalues[0] <= constants.ORB_HESS_EIG_THRESH:

        log("\n  The SCF is unstable wrt. restricted rotations.", calculation, 1, silent)

    if triplet_hessian_eigenvalues[0] <= constants.ORB_HESS_EIG_THRESH:

        log("\n  The SCF is unstable wrt. unrestricted rotations.", calculation, 1, silent)

    if singlet_hessian_eigenvalues[0] > constants.ORB_HESS_EIG_THRESH and triplet_hessian_eigenvalues[0] > constants.ORB_HESS_EIG_THRESH:

        log("\n  The self-consistent field solution is stable!", calculation, 1, silent)
    
    return










def perform_unrestricted_stability_analysis(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, silent: bool = False, K_XC: ndarray = None, spin_labels: list = None) -> None:
    
    """
    
    Performs a stability analysis for a spin-unrestricted reference.

    Args:
        g (array): Antisymmetrised spin orbital two-electron integrals 
        epsilons (array): spin orbital eigenvalues
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging
        K_XC (array, optional): Exchange-correlation kernel matrix
        spin_labels (list, optional): List of spin orbital labels
    
    """

    # Builds the orbital Hessian for UHF/UHF stability checks

    log("  Building unrestricted orbital Hessian...   ", calculation, 1, silent, end = "")

    H = build_orbital_hessian(g, epsilons, o, v, calculation, hessian_type = None, K_XC = K_XC, spin_labels = spin_labels)
    
    log("[Done]", calculation, 1, silent)
    
    log("\n  Diagonalising orbital Hessian...           ", calculation, 1, silent, end = "")

    # Finds the lowest eigenvalues and corresponding eigenvectors of the orbital Hessian

    hessian_eigenvalues, _ = np.linalg.eigh(H)
    
    log("[Done]", calculation, 1, silent)

    log(f"\n  Lowest Hessian eigenvalue:             {hessian_eigenvalues[0]:10.5f}", calculation, 1, silent)

    if hessian_eigenvalues[0] <= constants.ORB_HESS_EIG_THRESH:

        log("\n  The SCF is unstable wrt. unrestricted rotations.", calculation, 1, silent)

    else:

        log("\n  The self-consistent field solution is stable!", calculation, 1, silent)
    
    return










def determine_self_consistent_field_stability(molecule: Molecule, calculation: Calculation, ERI_AO: ndarray, SCF_output: Output, bfs_on_grid: ndarray = None, weights: ndarray = None, silent: bool = False) -> None:

    """
    
    Performs the stability analysis for an SCF solution.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        ERI_AO (array): Electron repulsion integrals in AO basis
        SCF_output (Output): Output from SCF calculation
        bfs_on_grid (array, optional): Basis functions on grid
        weights (array, optional): Integration weights for DFT
        silent (bool, optional): Cancel logging

    """
    
    K_XC_singlet, K_XC_triplet, K_XC = None, None, None

    # Stability analysis will work for the same functionals as TD-DFT

    if calculation.method.density_functional_method and not calculation.functional.time_dependent_available:

        error("Stability analysis is not yet available for this exchange-correlation functional!")

    if calculation.reference == "RHF":   # Handles the spin-restricted case
        
        g, _, epsilons, o, v = begin_spatial_orbital_calculation(molecule, ERI_AO, SCF_output, calculation, silent)
        
        if calculation.method.density_functional_method:   # Handles the DFT case
            
            K_XC_singlet, K_XC_triplet = calculate_restricted_exchange_correlation_kernel_matrix(o, v, SCF_output.density, bfs_on_grid, SCF_output.molecular_orbitals, calculation, weights, silent)
            
    else:   # Handles the spin-unrestricted case
        
        g, C_spin_block, epsilons, _, o, v, spin_labels, _, ERI_SO = begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, calculation, silent)
        
        g = ERI_SO - calculation.HFX_prop * ERI_SO.transpose(0, 1, 3, 2)

        if calculation.method.density_functional_method:   # Handles the DFT case
            
            K_XC = calculate_unrestricted_exchange_correlation_kernel_matrix(o, v, SCF_output.P_alpha, SCF_output.P_beta, bfs_on_grid, C_spin_block, spin_labels, calculation, weights, silent)

    log_spacer(calculation, 1, silent, start = "\n")
    log("                  Stability Analysis", calculation, 1, silent, colour = "white")
    log_spacer(calculation, 1, silent)
    
    # Performs the stability analysis for the restricted or unrestricted case

    if calculation.reference == "RHF":

        perform_restricted_stability_analysis(g, epsilons, o, v, calculation, silent, K_XC_singlet, K_XC_triplet)

    else:
        
        perform_unrestricted_stability_analysis(g, epsilons, o, v, calculation, silent, K_XC, spin_labels)
    
    log_spacer(calculation, 1, silent)

    return










def split_state_vector(column: ndarray, n_occ: int, n_virt: int) -> tuple:

    """

    Splits a state vector into its X (excitation) and Y (de-excitation) blocks.

    Args:
        column (array): State vector for one root
        n_occ (int): Number of occupied orbitals
        n_virt (int): Number of virtual orbitals

    Returns:
        X (array): Excitation block, shape (n_occ, n_virt)
        Y (array): De-excitation block, shape (n_occ, n_virt)

    """

    # Number of possible excitations

    n_ia = n_occ * n_virt

    if column.shape[0] == 2 * n_ia:   # Handles the full TDHF case

        X = column[:n_ia].reshape(n_occ, n_virt)
        Y = column[n_ia:].reshape(n_occ, n_virt)

    else:   # Handles the TDA case

        X = column.reshape(n_occ, n_virt)
        Y = np.zeros_like(X)

    return X, Y










def calculate_time_dependent_hartree_fock_states(A: ndarray, B: ndarray) -> tuple:

    """
    
    Calculates the state energies and vectors for time-dependent Hartree-Fock.

    Args:
        A (array): Excitation matrix
        B (array): Deexcitation matrix

    Returns:
        state_energies (array): Energies of excited states
        state_vectors (array): Vectors of excited states
    
    """

    if A is None or B is None:

        return None, None

    # Number of excitations

    n_ia = A.shape[0]

    # Forms non-Hermitian full Casida Hamiltonian

    H_TDHF = np.block([[A, B], [-B, -A]])

    state_energies, state_vectors = np.linalg.eig(H_TDHF)
    
    if np.max(np.abs(state_energies.imag)) > constants.COMPLEX_EIG_THRESH:

        warning("Diagonalisation gave complex excitation energies - the reference may be unstable!", space = 2)
    
    # Avoids complex energies

    state_energies = state_energies.real
    state_vectors = state_vectors.real 
    
    X, Y = state_vectors[:n_ia], state_vectors[n_ia:]

    metric_norm = np.einsum("in,in->n", X, X, optimize = True) - np.einsum("in,in->n", Y, Y, optimize = True)

    state_vectors = state_vectors / np.sqrt(np.abs(metric_norm))

    # Gets rid of negative eigenvalues and sorts the remaining states

    state_energies, state_vectors = state_energies[state_energies > 0], state_vectors[:, state_energies > 0]
    state_energies, state_vectors = state_energies[state_energies.argsort()], state_vectors[:, state_energies.argsort()]

    return state_energies, state_vectors










def print_initial_excited_state_information(calculation: Calculation, silent: bool) -> None:

    """
    
    Prints the initial information for excited state calculations.

    Args:
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
    
    """

    # Initial logging for excited state calculation

    log_spacer(calculation, 1, silent, start = "\n")

    if calculation.method.density_functional_method:

        log("      Time-dependent Density Functional Theory", calculation, 1, silent, colour = "white")

    elif calculation.tamm_dancoff_approximation:

        log("          Configuration Interaction Singles", calculation, 1, silent, colour = "white")
    else:

        log("            Time-dependent Hartree-Fock", calculation, 1, silent, colour = "white")
    
    log_spacer(calculation, 1, silent)
    
    if calculation.tamm_dancoff_approximation:

        log("  Using the Tamm-Dancoff approximation...", calculation, 1, silent, end = "\n\n")  
        
    else:
        
        log("  Not using the Tamm-Dancoff approximation...", calculation, 1, silent, end = "\n\n")
    
    if calculation.reference == "RHF":   # There is no multiplicity limitation on unrestricted references

        if not calculation.calculate_no_triplets and not calculation.calculate_no_singlets:

            log("  Singlet and triplet states will be calculated.", calculation, 1, silent)

        elif not calculation.calculate_no_triplets:

            log("  Only triplet states will be calculated.", calculation, 1, silent)

        else:
            
            log("  Only singlet states will be calculated.", calculation, 1, silent)

    return










def calculate_restricted_single_reference_excited_states(g: ndarray, epsilons: ndarray, o: slice, v: slice, calculation: Calculation, silent: bool = False, K_XC_singlet: ndarray = None, K_XC_triplet: ndarray = None) -> tuple:

    """
    
    Calculates the CIS or TDHF excited states for a spin-restricted reference.
    
    Args:
        g (array): Spatial orbital integrals in physicists' notation
        epsilons (array): Molecular orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging
        K_XC_singlet (array, optional): Singlet exchange-correlation kernel matrix
        K_XC_triplet (array, optional): Triplet exchange-correlation kernel matrix

    Returns:
        singlet_energies (array): Singlet state energies
        triplet_energies (array): Triplet state energies
        singlet_vectors (array): Singlet state weights
        triplet_vectors (array): Triplet state weights

    """

    singlet_energies, triplet_energies = None, None
    singlet_vectors, triplet_vectors = None, None
    A_singlet, A_triplet, B_singlet, B_triplet = None, None, None, None   

    # Determine TDA locally without mutating the shared calculation object state

    calculation.tamm_dancoff_approximation = True if "CIS" in calculation.method.name else calculation.tamm_dancoff_approximation

    print_initial_excited_state_information(calculation, silent)

    timer("Excited state calculation", 0)

    # Convert two-electron integrals to physicists' notation

    g = g.transpose(0, 2, 1, 3)
    
    log("\n  Building excited state Hamiltonian...      ", calculation, 1, silent, end = "")

    if not calculation.calculate_no_singlets:   # Calculates singlet matrices

        A_singlet = calculate_restricted_singlet_A_matrix(g, epsilons, o, v, calculation, K_XC_singlet)

        if not calculation.tamm_dancoff_approximation:  

            B_singlet = calculate_restricted_singlet_B_matrix(g, o, v, calculation, K_XC_singlet)

    if not calculation.calculate_no_triplets:   # Calculates triplet matrices

        A_triplet = calculate_restricted_triplet_A_matrix(g, epsilons, o, v, calculation, K_XC_triplet)

        if not calculation.tamm_dancoff_approximation:  

            B_triplet = calculate_restricted_triplet_B_matrix(g, o, v, calculation, K_XC_triplet) 
    
    log("[Done]", calculation, 1, silent)

    log("  Diagonalising Hamiltonian...               ", calculation, 1, silent, end = "")

    if calculation.tamm_dancoff_approximation:   # Uses the TDA

        if A_singlet is not None:
            
            singlet_energies, singlet_vectors = np.linalg.eigh(A_singlet)

        if A_triplet is not None:

            triplet_energies, triplet_vectors = np.linalg.eigh(A_triplet)
            
    else:   # Full TDHF eigenvalue problem

        singlet_energies, singlet_vectors = calculate_time_dependent_hartree_fock_states(A_singlet, B_singlet)

        triplet_energies, triplet_vectors = calculate_time_dependent_hartree_fock_states(A_triplet, B_triplet)

    log("[Done]", calculation, 1, silent)

    timer("Excited state calculation", 1)

    return singlet_energies, triplet_energies, singlet_vectors, triplet_vectors










def calculate_unrestricted_single_reference_excited_states(g: ndarray, epsilons: ndarray, o: slice, v: slice, n_occ: int, n_virt: int, spin_labels: list, calculation: Calculation, silent: bool = False, K_XC: ndarray = None) -> tuple:

    """

    Calculates the CIS or TDHF excited states for a spin-unrestricted reference.

    Args:
        g (array): Antisymmetrised spin orbital two-electron integrals in physicists' notation
        epsilons (array): spin orbital eigenvalues, sorted in ascending order
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        n_occ (int): Number of occupied spin orbitals
        n_virt (int): Number of virtual spin orbitals
        spin_labels (list): Spin ("a" or "b") of each spin orbital, in ascending energy order
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        excitation_energies (array): Excitation energies
        excitation_vectors (array): Weight vectors over the full occupied-virtual spin orbital space

    """

    calculation.tamm_dancoff_approximation = True if "CIS" in calculation.method.name else calculation.tamm_dancoff_approximation
    
    print_initial_excited_state_information(calculation, silent)

    timer("Excited state calculation", 0)

    # Selects only spin-conserving excitations, where the occupied and virtual spin orbitals share a spin

    spin_occupied = np.array(spin_labels)[o]
    spin_virtual = np.array(spin_labels)[v]

    spin_conserving = (spin_occupied[:, None] == spin_virtual[None, :]).ravel()

    n_spin_conserving = int(np.sum(spin_conserving))

    log("  Building excited state Hamiltonian...      ", calculation, 1, silent, end = "")

    # The unrestricted A matrix is built in the full spin orbital basis, then restricted to spin-conserving excitations

    A = calculate_unrestricted_A_matrix(g, epsilons, o, v, calculation, K_XC)[np.ix_(spin_conserving, spin_conserving)]

    log("[Done]", calculation, 1, silent)

    log("  Diagonalising Hamiltonian...               ", calculation, 1, silent, end = "")

    if calculation.tamm_dancoff_approximation:

        # Just run CIS, not TDHF, if the "TDA" keyword (or a CIS method) is used

        excitation_energies, vectors = np.linalg.eigh(A)

        # Scatters the spin-conserving eigenvectors back into the full occupied-virtual space (spin-flip weights are zero)

        excitation_vectors = np.zeros((n_occ * n_virt, len(excitation_energies)))
        excitation_vectors[spin_conserving, :] = vectors

    else:

        # The B matrix is also restricted to spin-conserving excitations for full TDHF

        B = calculate_unrestricted_B_matrix(g, o, v, calculation, K_XC)[np.ix_(spin_conserving, spin_conserving)]

        excitation_energies, vectors = calculate_time_dependent_hartree_fock_states(A, B)

        # Scatters the X and Y blocks back into the full occupied-virtual space (spin-flip weights are zero)

        excitation_vectors = np.zeros((2 * n_occ * n_virt, len(excitation_energies)))
        excitation_vectors[:n_occ * n_virt][spin_conserving, :] = vectors[:n_spin_conserving]
        excitation_vectors[n_occ * n_virt:][spin_conserving, :] = vectors[n_spin_conserving:]

    log("[Done]", calculation, 1, silent)

    timer("Excited state calculation", 1)

    return excitation_energies, excitation_vectors










def calculate_restricted_transition_dipoles(SCF_output: Output, singlet_vectors: ndarray, triplet_vectors: ndarray, n_occ: int, n_virt: int, o: slice, v: slice) -> ndarray:

    """
    
    Calculates the transition dipole from the ground state to each excited state.

    Args:
        SCF_output (Output): Output from SCF calculation
        singlet_vectors (array): Eigenvectors of singlet Hessian
        triplet_vectors (array): Eigenvectors of triplet Hessian
        n_occ (int): Number of doubly occupied orbitals
        n_virt (int): Number of doubly virtual orbitals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice

    Returns:
        transition_dipole (array): Transition dipoles
    
    """

    transition_dipoles = []

    # All three Cartesian dipole matrices in the MO basis

    D_MO = [transform_matrix_AO_to_SO(M, SCF_output.molecular_orbitals) for M in SCF_output.D] 

    if singlet_vectors is not None:

        for state in range(singlet_vectors.shape[1]):

            column = singlet_vectors[:, state]

            # The transition density is X + Y

            X, Y = split_state_vector(column, n_occ, n_virt)

            transitions_matrix = X + Y

            # Appends the magnitude of the transition dipole moment vector

            transition_dipoles.append(np.linalg.norm([np.sum(M[o, v] * transitions_matrix) for M in D_MO]))
        
    # Singlet to triplet transitions are always zero

    if triplet_vectors is not None:

        transition_dipoles += [0] * triplet_vectors.shape[1]

    # Normalises the transition dipoles array

    transition_dipoles = np.array(transition_dipoles) * np.sqrt(2)

    return transition_dipoles










def calculate_unrestricted_transition_dipoles(SCF_output: Output, excitation_vectors: ndarray, n_occ: int, n_virt: int, o: slice, v: slice, C_spin_block: ndarray) -> ndarray:

    """

    Calculates the transition dipole from the ground state to each excited state, for a spin-unrestricted reference.

    Args:
        SCF_output (Output): Output from SCF calculation
        excitation_vectors (array): Excitation weight vectors over the full occupied-virtual spin orbital space
        n_occ (int): Number of occupied spin orbitals
        n_virt (int): Number of virtual spin orbitals
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis

    Returns:
        transition_dipoles (array): Transition dipoles

    """

    transition_dipoles = []

    # All three Cartesian dipole matrices in the spin orbital MO basis (the AO dipole is spin-blocked first)

    D_SO = [transform_matrix_AO_to_SO(np.kron(np.eye(2), M), C_spin_block) for M in SCF_output.D]

    for state in range(excitation_vectors.shape[1]):

        column = excitation_vectors[:, state]

        # The transition density is X + Y 

        X, Y = split_state_vector(column, n_occ, n_virt)

        transitions_matrix = X + Y

        # Appends the magnitude of the transition dipole moment vector (no sqrt(2) factor, as both spins are summed explicitly)

        transition_dipoles.append(np.linalg.norm([np.sum(M[o, v] * transitions_matrix) for M in D_SO]))

    transition_dipoles = np.array(transition_dipoles)

    return transition_dipoles










def determine_restricted_excited_state_energy_and_density(excitation_energies: ndarray, excitation_vectors: ndarray, state: int, n_occ: int, n_virt: int, SCF_output: Output, o: slice, v: slice, molecular_orbitals: ndarray) -> tuple:

    """
    
    Picks out the excited state energy, and forms the state's density matrices.

    Args:
        excitation_energies (array): Excitation energies
        excitation_vectors (array): Weights of each determinant in each state
        state (int): Chosen root
        n_occ (int): Number of occupied orbitals
        n_virt (int): Number of virtual orbitals
        SCF_output (Output): Output object from SCF
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        molecular_orbitals (array): Molecular orbitals

    Returns:
        E_state (float): Energy of chosen state
        E_transition (float): Transition energy to chosen state
        P_state (array): Density matrix of chosen state
        P_state_alpha (array): Alpha spin density matrix of chosen state
        P_state_beta (array): Beta spin density matrix of chosen state
        P_diff (array): Difference density matrix of chosen state
        P_diff_alpha (array): Alpha spin difference density matrix of chosen state
        P_diff_beta (array): Beta spin difference density matrix of chosen state   

    """

    # This is the transition energy to the chosen state

    try:

        E_transition = excitation_energies[state]
    
    except IndexError: 
        
        error(f"Specified root ({state + 1}) does not exist!")

    # Picks out the weights for the selected state, for TDHF the unrelaxed density is X^2 + Y^2

    X, Y = split_state_vector(excitation_vectors[:, state], n_occ, n_virt)

    # Builds up difference density matrix for chosen state in spatial MO basis

    P_diff_MO = np.zeros_like(SCF_output.P)

    P_diff_MO[v, v] = np.einsum("ia,ib->ab", X, X, optimize = True) + np.einsum("ia,ib->ab", Y, Y, optimize = True)
    P_diff_MO[o, o] = -1 * (np.einsum("ia,ja->ij", X, X, optimize = True) + np.einsum("ia,ja->ij", Y, Y, optimize = True))

    # Transforms the density matrix to the AO basis

    P_diff = molecular_orbitals @ P_diff_MO @ molecular_orbitals.T

    # The difference density is symmetric in alpha and beta spins for restricted references

    P_diff_alpha = P_diff_beta = P_diff / 2

    # Forms the energy and densities of the state, rather than the transition

    E_state = SCF_output.energy + E_transition

    P_state = SCF_output.P + P_diff
    P_state_alpha = SCF_output.P_alpha + P_diff_alpha
    P_state_beta = SCF_output.P_beta + P_diff_beta

    return E_state, E_transition, P_state, P_state_alpha, P_state_beta, P_diff, P_diff_alpha, P_diff_beta










def determine_unrestricted_excited_state_energy_and_density(excitation_energies: ndarray, excitation_vectors: ndarray, state: int, n_occ: int, n_virt: int, SCF_output: Output, o: slice, v: slice, C_spin_block: ndarray) -> tuple:

    """

    Picks out the excited state energy, and forms the state's density matrices, for a spin-unrestricted reference.

    Args:
        excitation_energies (array): Excitation energies
        excitation_vectors (array): Weights of each determinant in each state, over the spin orbital space
        state (int): Chosen root
        n_occ (int): Number of occupied spin orbitals
        n_virt (int): Number of virtual spin orbitals
        SCF_output (Output): Output object from SCF
        o (slice): Occupied spin orbital slice
        v (slice): Virtual spin orbital slice
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis

    Returns:
        E_state (float): Energy of chosen state
        E_transition (float): Transition energy to chosen state
        P_state (array): Density matrix of chosen state
        P_state_alpha (array): Alpha spin density matrix of chosen state
        P_state_beta (array): Beta spin density matrix of chosen state
        P_diff (array): Difference density matrix of chosen state
        P_diff_alpha (array): Alpha spin difference density matrix of chosen state
        P_diff_beta (array): Beta spin difference density matrix of chosen state

    """

    # This is the transition energy to the chosen state

    try:

        E_transition = excitation_energies[state]

    except IndexError:

        error(f"Specified root ({state + 1}) does not exist!")

    # Picks out the weights for the selected state, for TDHF the unrelaxed density is X^2 + Y^2

    X, Y = split_state_vector(excitation_vectors[:, state], n_occ, n_virt)

    # Number of spin orbitals

    n_SO = C_spin_block.shape[1]

    # Builds up the difference density matrix for the chosen state in the spin orbital MO basis

    P_diff_MO = np.zeros((n_SO, n_SO))

    P_diff_MO[v, v] = np.einsum("ia,ib->ab", X, X, optimize = True) + np.einsum("ia,ib->ab", Y, Y, optimize = True)
    P_diff_MO[o, o] = -1 * (np.einsum("ia,ja->ij", X, X, optimize = True) + np.einsum("ia,ja->ij", Y, Y, optimize = True))

    # Transforms the spin orbital density matrix to the AO basis, splitting into alpha and beta spin densities

    P_diff, P_diff_alpha, P_diff_beta = transform_P_SO_to_AO(P_diff_MO, C_spin_block, n_SO)

    # Forms the energy and densities of the state, rather than the transition

    E_state = SCF_output.energy + E_transition

    P_state = SCF_output.P + P_diff
    P_state_alpha = SCF_output.P_alpha + P_diff_alpha
    P_state_beta = SCF_output.P_beta + P_diff_beta

    return E_state, E_transition, P_state, P_state_alpha, P_state_beta, P_diff, P_diff_alpha, P_diff_beta










def print_excited_state_absorption_spectrum(molecule: Molecule, excitation_energies: ndarray, calculation: Calculation, transition_dipoles: ndarray, oscillator_strengths: ndarray, state_types: ndarray, silent: bool = False) -> None:
    
    """

    Prints excited state absorption spectrum information.

    Args:   
        molecule (Molecule): Molecule object
        excitation_energies (array): Excitation energies
        calculation (Calculation): Calculation object
        transition_dipoles (array): Transition dipoles
        oscillator_strengths (array): Oscillator strengths
        silent (bool, optional): Should output be silenced

    """

    # Converts sorted excitation energies to different units

    wavelengths_nm = 1e7 / (excitation_energies * constants.per_cm_in_hartree)
    excitation_energies_eV = constants.eV_in_hartree * excitation_energies

    log_spacer(calculation, 1, silent, start = "\n")

    log(f"\n Transition dipole moment origin is the centre of mass, {bohr_to_angstrom(molecule.centre_of_mass):.4f} angstroms from the first atom.", calculation, 1, silent = silent)
    
    log_big_spacer(calculation, silent = silent, start = "\n")

    log("                                     Excited State Absorption Spectrum", calculation, 1, silent = silent, colour = "white")
    
    log_big_spacer(calculation, silent = silent)

    log("   State         Energy          Energy (eV)     Wavelength (nm)    Osc. Strength     Transition Dipole", calculation, 1, silent = silent)
    
    log_big_spacer(calculation, silent = silent)

    # Prints absorption frequency and intensity for each state

    for state in range(len(excitation_energies)):

        if state < calculation.n_states:
            
            # Appends either "S" or "T" for singlet and triplet restricted reference states

            state_type = " - " + state_types[state][0] if calculation.reference == "RHF" else "  "

            gap = "" if calculation.reference == "RHF" else "  "

            log(f"  {gap}{(state + 1):2}{state_type.upper()}  {excitation_energies[state]:16.10f}  {excitation_energies_eV[state]:14.5f}   {wavelengths_nm[state]:16.5f}       {oscillator_strengths[state]:10.5f}          {transition_dipoles[state]:10.5f}", calculation, 1, silent = silent)

    log_big_spacer(calculation, silent = silent)

    return










def print_excited_state_contributions(calculation: Calculation, silent: bool, excitation_energies: ndarray, excitation_vectors: ndarray, state_types: ndarray, n_occ: int, n_virt: int, o: slice, orbital_labels: ndarray = None) -> None:

    """
    
    Prints the orbital transition contributions to each excited state.

    Args:
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        excitation_energies (array): Excitation energies
        excitation_vectors (array): Weights of excitations
        state_types (array): Either "triplet" or "singlet" for restricted references
        n_occ (int): Number of occupied orbitals
        n_virt (int): Number of virtual orbitals
    
    """

    # Results without TDA will not perfectly match ORCA weights just due to random degeneracies of eigenvectors

    log("\n  Printing excited state information...", calculation, 2, silent = silent)

    log(f"  Only printing contributions larger than {calculation.excited_state_contribution_threshold:.1f} %.", calculation, 2, silent = silent)

    # Prints energies and transition weights of each excited state

    for state in range(min(len(excitation_energies), calculation.n_states)):

        log(f"\n  ~~~~~ State {state + 1} ~~~~~  {state_types[state].capitalize()}", calculation, 2, silent = silent)

        log(f"\n  Excitation energy: {excitation_energies[state]:16.10f}\n", calculation, 2, silent = silent)

        # The contribution of each orbital transition is the square of its weight, as a percentage

        X, Y = split_state_vector(excitation_vectors[:, state], n_occ, n_virt)

        # This normalisation is necessary to match THDF

        contributions = 100 * (X ** 2 - Y ** 2)

        # Loops over the transitions from largest to smallest contribution

        for index in np.argsort(contributions, axis = None)[::-1]:

            i, a = divmod(index, n_virt)

            # The list is sorted, so once one transition falls below the threshold the rest do too

            if contributions[i, a] <= calculation.excited_state_contribution_threshold: break

            # Prints occupied orbital i and virtual orbital a in one-indexed molecular orbital numbering

            if orbital_labels is not None:

                # Prints "a" or "b" for orbital transitions for UHF

                occ_label, virt_label = orbital_labels[o.start + i], orbital_labels[o.stop + a]

            else:                          

                # Prints just the orbital number for RHF

                occ_label, virt_label = f"{o.start + i + 1}", f"{o.stop + a + 1}"

            log(f"    {occ_label:>4}  ->  {virt_label:<4}  {contributions[i, a]:7.2f} %", calculation, 2, silent = silent)
            
    return










def calculate_restricted_doubles_correction(excitation_energy: ndarray, epsilons: ndarray, root: int, g: ndarray, o: slice, v: slice, b_ia: ndarray, state_type: str, calculation: Calculation, silent: bool = False) -> float:
 
    """
 
    Calculates the doubles correction to the CIS excitation energy of one state, for a spin-restricted reference.
 
    Args:
        excitation_energy (float): CIS excitation energy of the state of interest
        epsilons (array): Spatial molecular orbital eigenvalues
        root (int): State of interest, in zero-indexed counting
        g (array): Two-electron integrals in physicists' spatial orbital notation
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        b_ia (array): Normalised spatial CIS amplitudes for the state of interest
        state_type (str): Spin of the excited state, "singlet" or "triplet"
        calculation (Calculation): Calculation object
        silent (bool, optional): Should output be silenced
 
    Returns:
        E_D (float): (D) correction to the TDA excitation energy
 
    """
 
    # Spin-adaptation of the spin orbital equations of Head-Gordon, Rico, Oumi and Lee, Chem. Phys. Lett. 219, 21 (1994).
  
    log_spacer(calculation, silent = silent, start = "\n")
    log("          Perturbative Doubles Correction", calculation, 1, silent = silent, colour = "white")
    log_spacer(calculation, silent = silent)
 
    log(f"  Applying doubles correction to state {root + 1} only.", calculation, 1, silent = silent)
 
    log(f"\n  Building doubles amplitudes...           ", calculation, 1, silent = silent, end = "")
  
    e_ijab = build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
 
    shifted_denominator = 1 / (1 / e_ijab + excitation_energy)
 
    # First-order (opposite-spin) MP2 doubles amplitudes of the ground state
 
    t_ijab = build_MP2_t_amplitudes(g[o, o, v, v], e_ijab)
 
    log(f"  [Done]", calculation, 1, silent = silent)
 
    log(f"\n  Calculating direct contribution...  ", calculation, 1, silent = silent, end = "")
 
    p_1 = np.einsum("abcj,ic->ijab", g[v, v, v, o], b_ia, optimize = True)
    p_2 = np.einsum("abic,jc->ijab", g[v, v, o, v], b_ia, optimize = True)
    h_1 = np.einsum("kaji,kb->ijab", g[o, v, o, o], b_ia, optimize = True)
    h_2 = np.einsum("kbij,ka->ijab", g[o, v, o, o], b_ia, optimize = True)
 
    u_S = p_1 + p_2 - h_1 - h_2
    u_T = p_1 - p_2 + h_1 - h_2
 
    u_S_exchange = u_S.transpose(1, 0, 2, 3)
 
    if state_type == "singlet":
 
        E_direct = np.einsum("ijab,ijab,ijab->", shifted_denominator, u_S, u_S, optimize = True) - (1 / 2) * np.einsum("ijab,ijab,ijab->", shifted_denominator, u_S, u_S_exchange, optimize = True)
 
    else:
 
        E_direct = (1 / 2) * np.einsum("ijab,ijab,ijab->", shifted_denominator, u_S, u_S, optimize = True) - (1 / 2) * np.einsum("ijab,ijab,ijab->", shifted_denominator, u_S, u_S_exchange, optimize = True) + (1 / 2) * np.einsum("ijab,ijab,ijab->", shifted_denominator, u_T, u_T, optimize = True)
 
    log(f"       [Done]", calculation, 1, silent = silent)
 
    log(f"  Calculating indirect contribution...  ", calculation, 1, silent = silent, end = "")
 
    # Indirect contribution

    J = g[o, o, v, v]
    K = g[o, o, v, v].swapaxes(2, 3)
 
    if state_type == "singlet":
 
        v_ia = np.einsum("jkbc,jb,ikac->ia", 2 * J - K, b_ia, 2 * t_ijab - t_ijab.transpose(0, 1, 3, 2), optimize = True)
 
    else:
 
        v_ia = np.einsum("jkbc,jb,ikac->ia", K, b_ia, t_ijab.transpose(0, 1, 3, 2), optimize = True)
 
    v_ia += (1 / 2) * np.einsum("jkbc,ja,ikcb->ia", J, b_ia, t_ijab, optimize = True) - np.einsum("jkbc,ja,ikbc->ia", J, b_ia, t_ijab, optimize = True) - np.einsum("jkbc,ja,ikcb->ia", K, b_ia, t_ijab, optimize = True) + (1 / 2) * np.einsum("jkbc,ja,ikbc->ia", K, b_ia, t_ijab, optimize = True)
 
    v_ia += (1 / 2) * np.einsum("jkbc,ib,jkca->ia", J, b_ia, t_ijab, optimize = True) - np.einsum("jkbc,ib,jkac->ia", J, b_ia, t_ijab, optimize = True) - np.einsum("jkbc,ib,jkca->ia", K, b_ia, t_ijab, optimize = True) + (1 / 2) * np.einsum("jkbc,ib,jkac->ia", K, b_ia, t_ijab, optimize = True)
 
    log(f"     [Done]", calculation, 1, silent = silent)
 
    log(f"\n  Calculating doubles correction...         ", calculation, 1, silent = silent, end = "")
 
    E_D = E_direct + np.einsum("ia,ia->", b_ia, v_ia, optimize = True)
 
    log(f" [Done]", calculation, 1, silent = silent)
 
    # Summary of the correction for the state of interest
 
    log(f"\n  Original excitation energy:       {excitation_energy:15.10f}", calculation, 1, silent = silent)
    log(f"  Correction energy from (D):       {E_D:15.10f}", calculation, 1, silent = silent)
    log(f"  Correction energy (eV):           {(E_D * constants.eV_in_hartree):15.10f}", calculation, 3, silent = silent)
    log(f"\n  Corrected excitation energy:      {(E_D + excitation_energy):15.10f}", calculation, 1, silent = silent)

    log_spacer(calculation, silent = silent)
 
    return E_D










def calculate_unrestricted_doubles_correction(excitation_energy: ndarray, epsilons: ndarray, root: int, g: ndarray, o: slice, v: slice, b_ia: ndarray, calculation: Calculation, silent: bool = False) -> float:

    """

    Calculate sdoubles correction to TDA excitation energy of one state.

    Args:   
        excitation_energy (float): Excitation energy of state of interest
        epsilons (array): Fock matrix orbital eigenvalues
        root (int): State of interest
        g (array): Electron repulsion integrals in SO basis, antisymmetrised
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        b_ia (array): Two-index weights matrix for state of interest
        calculation (Calculation): Calculation object
        silent (bool, optional): Should output be silenced

    Returns:
        E_D (float): Doubles correction to TDA excitation energy

    """

    # Equations taken from Head-Gordon paper

    log_spacer(calculation, 1, silent, start = "\n")
    log("          Perturbative Doubles Correction", calculation, 1, silent, colour = "white")
    log_spacer(calculation, 1, silent)

    log(f"  Applying doubles correction to state {root + 1} only.", calculation, 1, silent)
   
    log(f"\n  Building doubles amplitudes...             ", calculation, 1, silent, end = "")
    
    # Builds and inverts inverse epsilons tensor, to upright e_ijab_inv

    e_ijab_inv = 1 / build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
    e_ijab_inv_minus_w = 1 / (e_ijab_inv + excitation_energy)

    t_ijab = build_MP2_t_amplitudes(g[o, o, v, v], 1 / e_ijab_inv)

    log(f"[Done]", calculation, 1, silent)

    log(f"\n  Calculating direct contribution...         ", calculation, 1, silent, end = "")
    
    u_1 = np.einsum("abcj,ic->ijab", g[v, v, v, o], b_ia, optimize = True)
    u_2 = np.einsum("abci,jc->ijab", g[v, v, v, o], b_ia, optimize = True)
    u_3 = np.einsum("kaij,kb->ijab", g[o, v, o, o], b_ia, optimize = True)
    u_4 = np.einsum("kbij,ka->ijab", g[o, v, o, o], b_ia, optimize = True)

    u_ijab = u_1 - u_2 + u_3 - u_4

    log(f"[Done]", calculation, 1, silent)

    log(f"  Calculating indirect contribution...       ", calculation, 1, silent, end = "")
    
    v_1 = (1 / 2) * np.einsum("jkbc,ib,jkca->ia", g[o, o, v, v], b_ia, t_ijab, optimize = True)
    v_2 = (1 / 2) * np.einsum("jkbc,ja,ikcb->ia", g[o, o, v, v], b_ia, t_ijab, optimize = True)
    v_3 = np.einsum("jkbc,jb,ikac->ia", g[o, o, v, v], b_ia, t_ijab, optimize = True)

    v_ia = v_1 + v_2 + v_3

    log(f"[Done]", calculation, 1, silent)

    log(f"\n  Calculating doubles correction...          ", calculation, 1, silent, end = "")
    
    E_D = (1 / 4) * np.einsum("ijab,ijab,ijab->", u_ijab, u_ijab, e_ijab_inv_minus_w, optimize = True) + np.einsum("ia,ia->", b_ia, v_ia, optimize = True)

    log(f"[Done]", calculation, 1, silent)
    
    # Correction energy in eV prints if P used

    log(f"\n  Original excitation energy:       {excitation_energy:15.10f}", calculation, 1, silent)
    log(f"  Correction energy from (D):       {E_D:15.10f}", calculation, 1, silent)
    log(f"  Correction energy (eV):           {(E_D * constants.eV_in_hartree):15.10f}", calculation, 3, silent)
    log(f"\n  Corrected excitation energy:      {(E_D + excitation_energy):15.10f}", calculation, 1, silent)

    log_spacer(calculation, 1, silent)
  
    return E_D










def run_perturbative_doubles(state: int, n_occ: int, n_virt: int, excitation_vectors: ndarray, g: ndarray, epsilons: ndarray, o: slice, v: slice, state_types: ndarray, state_of_interest_energies_and_densities: tuple, calculation: Calculation, silent: bool) -> tuple:

    """
    
    Runs a perturbative doubles correction to a state.

    Args:
        state (int): State of interest
        n_occ (int): Number of occupied orbitals
        n_virt (int): Number of virtual orbitals
        excitation_vectors (array): Excitation weight vectors
        g (array): Two-electron integrals in MO basis
        epsilons (array): Molecular orbital eigenvalues
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        state_types (array): Either "triplet" or "singlet" for restricted references
        state_of_interest_energies_and_densities (tuple): Energy and densities for chosen state
        calculation (Calculation): Calculation object
        silent (bool, optional): Should output be silenced

    Returns:
        state_of_interest_energies_and_densities (tuple): Energy and densities for chosen state
    
    """

    timer("Perturbative doubles", 0)

    X, Y = split_state_vector(excitation_vectors[:, state], n_occ, n_virt)

    b_ia = X + Y

    # Calculates an optional perturbative doubles correction - need to check whether this works as intended with HFX-scaled g

    if calculation.reference == "RHF":

        E_CIS_D = calculate_restricted_doubles_correction(state_of_interest_energies_and_densities[1], epsilons, state, g.transpose(0, 2, 1, 3), o, v, b_ia, state_types[state], calculation, silent)

    else:

        E_CIS_D = calculate_unrestricted_doubles_correction(state_of_interest_energies_and_densities[1], epsilons, state, g, o, v, b_ia, calculation, silent)

    # Updates the state and transition energies

    state = state_of_interest_energies_and_densities

    state_of_interest_energies_and_densities = (state[0] + E_CIS_D, state[1] + E_CIS_D) + state[2:]
    
    timer("Perturbative doubles", 1)

    return state_of_interest_energies_and_densities










def run_excited_state_calculation(molecule: Molecule, calculation: Calculation, SCF_output: Output, bfs_on_grid: ndarray = None, weights: ndarray = None, silent: bool = False) -> tuple:
    
    """
    
    Runs a single reference (TD-HF or TD-DFT) excited state calculation.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        SCF_output (Output): Output object
        bfs_on_grid (array, optional): Basis functions on integration grid
        weights (array, optional): Integration weights
        silent (bool, optional): Supress logging

    Returns:
        state_of_interest_energies_and_densities (tuple): Energy and densities for chosen state
    
    """

    if calculation.calculate_no_singlets and calculation.calculate_no_triplets:

        error("There are no excited states to calculate!")

    if calculation.method.density_functional_method and not calculation.functional.time_dependent_available:

        error("Time-dependent DFT is not yet available for this exchange-correlation functional!")

    if not calculation.method.excited_state_method and not calculation.method.density_functional_method and not calculation.method.name == "HF":

        error(f"Excited states are not available with {calculation.method.name}!")

    K_XC_singlet, K_XC_triplet, K_XC, spin_orbital_labels = None, None, None, None

    # Picks out the desired state, in zero-indexed computer counting

    state = calculation.root - 1

    if calculation.reference == "RHF":
        
        # Transforms integrals to the spatial MO basis

        g, molecular_orbitals, epsilons, o, v = begin_spatial_orbital_calculation(molecule, SCF_output.integrals.ERI_AO, SCF_output, calculation, silent)
        
        # Allows frozen core calculations

        n_occ, n_virt = o.stop - o.start, molecule.n_doubly_virt
        
        # For TD-DFT calculation, determine the exchange-correlation kernel on a grid, then its MO matrix elements

        if calculation.method.density_functional_method:
            
            K_XC_singlet, K_XC_triplet = calculate_restricted_exchange_correlation_kernel_matrix(o, v, SCF_output.density, bfs_on_grid, molecular_orbitals, calculation, weights, silent)
            
        # Calculates the singlet and triplet state energies and weight vectors

        singlet_energies, triplet_energies, singlet_vectors, triplet_vectors = calculate_restricted_single_reference_excited_states(g, epsilons, o, v, calculation, silent, K_XC_singlet, K_XC_triplet)
        
        # Combined array of all excitation energies and weight vectors

        excitation_energies = np.concatenate([e for e in (singlet_energies, triplet_energies) if e is not None])

        excitation_vectors = np.concatenate([v for v in (singlet_vectors, triplet_vectors) if v is not None], axis = 1)

        state_types = np.concatenate([np.full(len(e), label) for e, label in ((singlet_energies, "singlet"), (triplet_energies, "triplet")) if e is not None])

    elif calculation.reference == "UHF":
        
        # Transforms integrals to the spin orbital basis

        g, C_spin_block, epsilons, _, o, v, spin_labels, spin_orbital_labels, ERI_SO = begin_spin_orbital_calculation(molecule, SCF_output.integrals.ERI_AO, SCF_output, calculation, silent)
        
        # Allows frozen core calculations

        n_occ, n_virt = o.stop - o.start, molecule.n_virt

        # For TD-DFT calculation, determine the exchange-correlation kernel on a grid, then its MO matrix elements

        if calculation.method.density_functional_method:
            
            K_XC = calculate_unrestricted_exchange_correlation_kernel_matrix(o, v, SCF_output.P_alpha, SCF_output.P_beta, bfs_on_grid, C_spin_block, spin_labels, calculation, weights, silent)
        
        # Antisymmetrised integrals scaled by HFX

        g = ERI_SO - calculation.HFX_prop * ERI_SO.transpose(0, 1, 3, 2)

        # Calculates the state energies and weight vectors

        excitation_energies, excitation_vectors = calculate_unrestricted_single_reference_excited_states(g, epsilons, o, v, n_occ, n_virt, spin_labels, calculation, silent, K_XC)

        # Unrestricted references do not separate states by spin multiplicity

        state_types = np.array([""] * len(excitation_energies))
    
    log("\n  Calculating oscillator strengths...        ", calculation, 1, silent, end = "")

    # Compute the transition dipoles between states

    if calculation.reference == "RHF":

        transition_dipoles = calculate_restricted_transition_dipoles(SCF_output, singlet_vectors, triplet_vectors, n_occ, n_virt, o, v)

    else:

        transition_dipoles = calculate_unrestricted_transition_dipoles(SCF_output, excitation_vectors, n_occ, n_virt, o, v, C_spin_block)

    # Calculates the oscillator strengths for each transition

    oscillator_strengths = calculate_oscillator_strengths(transition_dipoles, excitation_energies)
    
    log("[Done]", calculation, 1, silent)
    
    # Reorders the state arrays from smallest to largest excitation energy

    order = np.argsort(excitation_energies)

    excitation_vectors = excitation_vectors[:, order]

    excitation_energies, state_types, transition_dipoles, oscillator_strengths = (arr[order] for arr in (excitation_energies, state_types, transition_dipoles, oscillator_strengths))
    
    log("  Constructing density matrix...             ", calculation, 1, silent, end = "")
    
    # Final energy and density matrix of state of interest

    if calculation.reference == "RHF":

        state_of_interest_energies_and_densities = determine_restricted_excited_state_energy_and_density(excitation_energies, excitation_vectors, state, n_occ, n_virt, SCF_output, o, v, molecular_orbitals)

    else:

        state_of_interest_energies_and_densities = determine_unrestricted_excited_state_energy_and_density(excitation_energies, excitation_vectors, state, n_occ, n_virt, SCF_output, o, v, C_spin_block)
    
    log("[Done]", calculation, 1, silent)

    # Print excited state information

    print_excited_state_contributions(calculation, silent, excitation_energies, excitation_vectors, state_types, n_occ, n_virt, o, spin_orbital_labels)

    # Prints excited state absorption spectrum

    print_excited_state_absorption_spectrum(molecule, excitation_energies, calculation, transition_dipoles, oscillator_strengths, state_types, silent)
    
    # Optional (D) correction

    if calculation.do_perturbative_doubles or "[D]" in calculation.method.name or "(D)" in calculation.method.name:
        
        state_of_interest_energies_and_densities = run_perturbative_doubles(state, n_occ, n_virt, excitation_vectors, g, epsilons, o, v, state_types, state_of_interest_energies_and_densities, calculation, silent)


    return state_of_interest_energies_and_densities