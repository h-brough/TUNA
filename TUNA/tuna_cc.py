import numpy as np
import tuna_ci as ci
import tuna_mp as mp
from tuna_util import *
import sys
from tuna_molecule import Molecule
from tuna_calc import Calculation


"""

This is the TUNA module for coupled cluster theory, written first for version 0.7.0 and rewritten for version 0.10.0.

All iterative coupled cluster calculations run through the same function, calculate_coupled_cluster_energy, which updates the amplitudes
using damping and DIIS. For most methods, interleaved physicist's notation is used for the molecular orbital basis two-electron integrals. The
einsum function of NumPy is used throughout with "optimize=True" to approximate the most efficient contraction path, leading to huge speedups and 
reductions in memory requirements.

Updated in version 0.10.1 to add spin-restricted CC2 and CC3.

The module contains:

1. Functions to calculate the coupled cluster energy (calculate_restricted_coupled_cluster_energy, calculate_unrestricted_coupled_cluster_energy)
2. Useful utility functions (is_coupled_cluster_converged, permute)
3. The amplitude update equations for all coupled cluster methods
4. Perturbative triples and quadruples functions (calculate_restricted_CCSD_T_energy, calculate_restricted_CCSDT_Q_energy)
4. The main routine to run coupled cluster energy calculations, and to apply the post-processing (begin_coupled_cluster_calculation)

"""




def calculate_restricted_coupled_cluster_energy(o: slice, v: slice, w: ndarray, t_amplitudes: tuple, method: Method, F: ndarray) -> tuple:
   
    """

    Calculates the spin-restricted coupled cluster energy.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        method (Method): Electronic structure method
        F (array): Spatial-orbital Fock matrix

    Returns:
        E_CC (float): Restricted coupled cluster energy
        E_singles (float): Restricted coupled cluster energy from single excitations
        E_connected_doubles (float): Restricted coupled cluster energy from connected double excitations
        E_disconnected_doubles (float): Restricted coupled cluster energy from disconnected double excitations

    """

    t_ia, t_ijab, _, _ = t_amplitudes

    # Contribution to coupled cluster energy from single excitations (should be zero)

    E_singles = np.einsum("ia,ia->", F[o, v], t_ia, optimize=True) if t_ia is not None and F is not None else 0

    # Contribution to coupled cluster energy from connected double excitations (should be large at normal bond lengths)

    E_connected_doubles = np.einsum("abij,ijab->", w[v, v, o, o], t_ijab, optimize=True)

    # Contribution to coupled cluster energy from disconnected double excitations (should be small at normal bond lengths)
    
    E_disconnected_doubles = np.einsum("abij,ia,jb->", w[v, v, o, o], t_ia, t_ia, optimize=True) if t_ia is not None else 0

    # In linearised and QCI methods, the total energy does not have a disconnected contribution

    if method.name in ["LCCD", "LCCSD", "QCISD", "QCISD[T]"]:

        E_disconnected_doubles = 0

    E_CC = E_singles + E_connected_doubles + E_disconnected_doubles

    return E_CC, E_singles, E_connected_doubles, E_disconnected_doubles










def calculate_unrestricted_coupled_cluster_energy(o: slice, v: slice, g: ndarray, t_amplitudes: tuple, method: Method, F: ndarray) -> tuple:

    """
    
    Calculates the spin-unrestricted coupled cluster energy.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        g (array): Spin-orbital antisymmetrised two-electron integrals
        t_amplitudes (tuple): Amplitudes
        method (Method): Electronic structure method
        F (array): Spin-orbital Fock matrix
    
    Returns:
        E_CC (float): Unrestricted coupled cluster energy
        E_singles (float): Unrestricted coupled cluster energy from single excitations
        E_connected_doubles (float): Unrestricted coupled cluster energy from connected double excitations
        E_disconnected_doubles (float): Unrestricted coupled cluster energy from disconnected double excitations
    
    """

    t_ia, t_ijab, _, _ = t_amplitudes

    # Contribution to coupled cluster energy from single excitations (should be zero)

    E_singles = np.einsum("ia,ia->", F[o, v], t_ia, optimize=True) if t_ia is not None and F is not None else 0

    # Contribution to coupled cluster energy from connected double excitations (should be large at normal bond lengths)

    E_connected_doubles = (1 / 4) * np.einsum("ijab,ijab->", g[o, o, v, v], t_ijab, optimize=True)

    # Contribution to coupled cluster energy from disconnected double excitations (should be small at normal bond lengths)

    E_disconnected_doubles = (1 / 2) * np.einsum("ijab,ia,jb->", g[o, o, v, v], t_ia, t_ia, optimize=True) if t_ia is not None else 0
    
    # In linearised and QCI methods, the total energy does not have a disconnected contribution

    if method.name in ["LCCD", "LCCSD", "QCISD", "QCISD[T]", "ULCCD", "ULCCSD", "UQCISD", "UQCISD[T]"]:

        E_disconnected_doubles = 0

    E_CC = E_singles + E_connected_doubles + E_disconnected_doubles

    return E_CC, E_singles, E_connected_doubles, E_disconnected_doubles










def coupled_cluster_initial_print(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, reference: str, method: Method, calculation: Calculation, silent: bool) -> None:

    """
    
    Prints common and prerequisite information for coupled cluster calculations.

    Args:
        g (array): Spatial orbital two-electron integrals or antisymmetrised spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (array): Amplitudes
        reference (str): Either RHF or UHF
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
    
    """

    log_spacer(calculation, silent=silent, start="\n")
    log(f"              {method.name:>5} Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log(f"  Energy convergence tolerance:        {calculation.correlated_energy_convergence:.10f}", calculation, 1, silent=silent)
    log(f"  Amplitude convergence tolerance:     {calculation.amp_conv:.10f}", calculation, 1, silent=silent)

    # Calculates the initial guess MP2 energy

    _, t_ijab, _, _ = t_amplitudes

    if reference == "RHF":
        
        E_MP2 = mp.calculate_restricted_MP2_energy(t_ijab, g[o, o, v, v])

    else:
            
        E_MP2 = mp.calculate_unrestricted_MP2_energy(t_ijab, g[o, o, v, v])


    log(f"\n  Guess t-amplitude MP2 energy:       {E_MP2:.10f}\n", calculation, 1, silent=silent)

    if calculation.correlated_damping_parameter != 0 : 
        
        log(f"  Using damping parameter of {calculation.correlated_damping_parameter:.2f} for convergence.", calculation, 1, silent=silent)

    if calculation.DIIS: 

        log(f"  Using DIIS, storing {calculation.max_DIIS_matrices} matrices, for convergence.", calculation, 1, silent=silent)


    log(f"\n  Starting {method.name} iterations...\n", calculation, 1, silent=silent)


    log_spacer(calculation, silent=silent)
    log("  Step          Correlation E               DE", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)


    return










def permute(array: ndarray, idx_1: int, idx_2: int) -> ndarray:

    """

    Incorporates antisymmetric permutation into an array. This is the definition of P- from the Stanton paper on CCSD.

    Args:
        array (array): Array to be permuted
        idx_1 (int): First index
        idx_2 (int): Second index

    Returns:
        permuted_array (array): Antisymmetrically permuted array

    """

    permuted_array = array - array.swapaxes(idx_1, idx_2)

    return permuted_array










def is_coupled_cluster_converged(delta_E: float, t_amplitudes: tuple, t_amplitudes_old: tuple, calculation: Calculation) -> bool:

    """
    
    Checks if the coupled cluster iterations have converged. Only checks singles and doubles amplitudes, since these determine 
    the energy and all properties in TUNA are numerical derivatives of the energy.
    
    Args:
        delta_E (float): Change in energy from last iteration
        t_amplitudes (tuple): Amplitudes
        t_amplitudes_old (tuple): Amplitudes from last iteration
        calculation (Calculation): Calculation object
    
    Returns:
        is_coupled_cluster_converged (bool): Has convergence been achieved

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    t_ia_old, t_ijab_old, _, _ = t_amplitudes_old

    # First checks for energy convergence

    if abs(delta_E) < calculation.correlated_energy_convergence:

        # Then checks for doubles amplitudes convergence

        if np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv:

            # Then checks for singles amplitudes convergence, if singles are calculated

            if t_ia is None or np.linalg.norm(t_ia - t_ia_old) < calculation.amp_conv:

                return True

    return False










def apply_damping(damping_factor: float, t_amplitudes: tuple, t_amplitudes_old: tuple) -> tuple:

    """
    
    Applies a damping factor to the coupled cluster amplitudes.

    Args:
        damping_factor (float): Coupled cluster damping factor
        t_amplitudes (tuple): Amplitudes
        t_amplitudes_old (tuple): Amplitudes from last iteration

    Returns:
        damped_amplitudes (tuple): New damped amplitudes
    
    """

    damped_amplitudes = []

    # Applies damping to the newly calculated t-amplitudes

    for amplitude, old_amplitude in zip(t_amplitudes, t_amplitudes_old):
        
        if amplitude is None:

            damped_amplitude = None

        else:

            # Updates amplitudes with corresponding old amplitudes

            damped_amplitude = damping_factor * old_amplitude + (1 - damping_factor) * amplitude

        damped_amplitudes.append(damped_amplitude)


    return tuple(damped_amplitudes)










def update_DIIS(t_vectors: tuple, DIIS_error_vector: list, calculation: Calculation, silent: bool) -> tuple:

    """
    
    Extrapolates the t-amplitudes using DIIS.

    Args:
        t_vectors (tuple): Tuple of lists t_ia_vector, t_ijab_vector, t_ijkabc_vector and t_ijklabcd vector
        DIIS_error_vector (list): Error vector for DIIS, same shape as t_vectors
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        t_amplitudes (tuple): Extrapolated amplitudes

    """

    # The first item in each vector should be deleted if too many matrices are stored

    if len(DIIS_error_vector) > calculation.max_DIIS_matrices:

        del DIIS_error_vector[0]

        for vec in t_vectors: 
            
            del vec[0]

    # Converts to array to easily construct B matrix

    DIIS_errors = np.array(DIIS_error_vector)
    n_DIIS = len(DIIS_error_vector)

    # Builds B matrix and right hand side of Pulay equations

    B = np.empty((n_DIIS + 1, n_DIIS + 1))
    B[:n_DIIS, :n_DIIS] = DIIS_errors @ DIIS_errors.T 
    B[:n_DIIS, -1] = -1
    B[-1, :n_DIIS] = -1
    B[-1, -1] = 0.0

    RHS = np.zeros(n_DIIS + 1)
    RHS[-1] = -1.0

    try:

        # Solves system of equations

        coeffs = np.linalg.solve(B, RHS)[:n_DIIS]

        extrapolated_amplitudes = []

        for t_vector in t_vectors:

            if t_vector is not None:

                extrapolated_amplitude = np.tensordot(coeffs, t_vector, axes=(0, 0))

            else:

                extrapolated_amplitude = None

            extrapolated_amplitudes.append(extrapolated_amplitude)


    except np.linalg.LinAlgError:

        # Clears all the vectors in t_vectors

        [vec.clear() for vec in t_vectors if vec is not None]

        DIIS_error_vector.clear()

        extrapolated_amplitudes = [None, None, None, None]

        log("   (Resetting DIIS)", calculation, 1, end="", silent=silent)


    return tuple(extrapolated_amplitudes)










def apply_DIIS(t_amplitudes: tuple, t_amplitudes_old: tuple, t_vectors: tuple, error_vector: ndarray, step: int, calculation: Calculation, silent: bool) -> tuple:

    """
    
    Updates the t-amplitudes using DIIS.

    Args:
        t_amplitudes (tuple): Amplitudes
        t_amplitudes_old (tuple): Amplitudes from last iteration
        t_vectors (list): Vectors of t-amplitudes
        error_vector (list): Error vector for DIIS
        step (int): Iteration of coupled cluster
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        t_amplitudes (tuple): Extrapolated amplitudes
        t_vectors (tuple): Updated vectors of t-amplitudes
        error_vector (array): Updated error vector for DIIS

    """

    t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes
    t_ia_old, t_ijab_old, t_ijkabc_old, t_ijklabcd_old = t_amplitudes_old
    t_ia_vector, t_ijab_vector, t_ijkabc_vector, t_ijklabcd_vector = t_vectors

    def store_block(t: ndarray, t_old: ndarray, history: list) -> ndarray:

        # Stores a block of t-amplitudes for extrapolation

        if t is None:

            history.append(np.zeros(1))

            return None
        
        history.append(t.copy())

        return (t - t_old).ravel()


    # Builds the full residual list

    residuals = [

        residual for residual in (

            store_block(t_ia, t_ia_old, t_ia_vector),
            store_block(t_ijab, t_ijab_old, t_ijab_vector),
            store_block(t_ijkabc, t_ijkabc_old, t_ijkabc_vector),
            store_block(t_ijklabcd, t_ijklabcd_old, t_ijklabcd_vector),

        ) if residual is not None
        
    ]

    error_vector.append(np.concatenate(residuals))

    t_vectors = t_ia_vector, t_ijab_vector, t_ijkabc_vector, t_ijklabcd_vector

    if step > 2 and calculation.DIIS:

        # Extrapolates the amplitudes with DIIS

        t_ia_DIIS, t_ijab_DIIS, t_ijkabc_DIIS, t_ijklabcd_DIIS = update_DIIS(t_vectors, error_vector, calculation, silent)

        # Updates the amplitudes with the extrapolated ones, if they're present

        if t_ia is not None and t_ia_DIIS is not None: t_ia = t_ia_DIIS
        if t_ijab is not None and t_ijab_DIIS is not None: t_ijab = t_ijab_DIIS
        if t_ijkabc is not None and t_ijkabc_DIIS is not None: t_ijkabc = t_ijkabc_DIIS
        if t_ijklabcd is not None and t_ijklabcd_DIIS is not None: t_ijklabcd = t_ijklabcd_DIIS


    t_amplitudes = t_ia, t_ijab, t_ijkabc, t_ijklabcd

    return t_amplitudes, t_vectors, error_vector










def calculate_coupled_cluster_linearised_density(t_ia: ndarray, t_ijab: ndarray, n_orbitals: int, n_occ: int, o: slice, v: slice, calculation: Calculation, molecular_orbitals: ndarray, silent: bool) -> tuple:

    """

    Calculates the coupled cluster linearised one-particle reduced density matrix.

    Args:

        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        n_orbitals (int): Number of spin orbitals
        molecular_orbitals (array): Spin-blocked molecular orbitals (SO) or molecular orbitals (spatial orbitals)
        n_occ (int): Number of occupied spin orbitals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        P (array): Full linearised density matrix in AO basis
        P_alpha (array): Alpha linearised density matrix in AO basis
        P_beta (array): Beta linearised density matrix in AO basis

    """

    log("\n  Constructing linearised density...    ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Correlated part of density matrix from squared connected doubles

    P_CC = np.zeros((n_orbitals, n_orbitals))
    
    
    if calculation.reference == "RHF":

        u_ijab = t_ijab * 2 - t_ijab.swapaxes(2, 3) 

        P_CC[v, v] += np.einsum('ijbc,ijac->ab', t_ijab, u_ijab, optimize=True)
        P_CC[o, o] += -np.einsum('ikab,jkab->ij', t_ijab, u_ijab, optimize=True)
        
        P_CC[o, v] += t_ia + np.einsum("ijab,jb->ia", u_ijab, t_ia, optimize=True) 

    else:

        P_CC[v, v] += (1 / 2) * np.einsum('ijbc,ijac->ab', t_ijab, t_ijab, optimize=True)
        P_CC[o, o] += - (1 / 2) * np.einsum('ikab,jkab->ij', t_ijab, t_ijab, optimize=True)
        
        P_CC[o, v] += t_ia + np.einsum("ijab,jb->ia", t_ijab, t_ia, optimize=True) 


    # Linearised coupled-cluster density, only includes up to double excitations I think

    P_CC[v, o] = P_CC[o, v].T

    P_CC[v, v] += np.einsum("ia,ib->ab", t_ia, t_ia, optimize=True) 
    P_CC[o, o] -= np.einsum("ia,ja->ij", t_ia, t_ia, optimize=True)


    # Builds reference density matrix in SO or spatial orbital basis, diagonal in ones - note we don't use "o" here as this is general for frozen core

    P_ref = np.zeros((n_orbitals, n_orbitals))
    P_ref[slice(0, n_occ), slice(0, n_occ)] = np.identity(n_occ)

    P = P_ref + P_CC 

    if calculation.reference == "UHF":

        # Transforms the density matrix from spin-orbital to atomic orbital basis

        P, P_alpha, P_beta = ci.transform_P_SO_to_AO(P, molecular_orbitals, n_orbitals) 

    else:

        # For RHF, all orbitals are doubly occupied

        P *= 2

        # Transforms the density matrix from the spatial orbital to atomic orbital basis

        P = molecular_orbitals @ P @ molecular_orbitals.T
        P_alpha = P_beta = P / 2

    log("     [Done]", calculation, 1, silent=silent)

    density_matrices = P, P_alpha, P_beta

    return density_matrices










def calculate_T1_diagnostic(molecule: Molecule, t_ia: ndarray, spin_labels_sorted: list, n_occ: int, n_alpha: int, n_beta: int, calculation: Calculation, silent: bool) -> None:

    """
    
    Calculates the T1 diagnostic for a coupled cluster calculation.

    Args:
        molecule (Molecule): Molecule object
        t_ia (array): Singles amplitudes
        spin_labels_sorted (list): List of sorted spin labels for spin orbitals
        n_occ (int): Number of occupied spin-orbitals
        n_alpha (int): Number of alpha electrons
        n_beta (int): Number of beta electrons
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging
    
    """


    if calculation.reference == "UHF":

        # Finds the alpha and beta indices that are occupied

        alpha_occupied_indices = [i for i, spin in enumerate(spin_labels_sorted) if spin == 'a' and i < n_occ]
        beta_occupied_indices = [i for i, spin in enumerate(spin_labels_sorted) if spin == 'b' and i < n_occ]

        # Removes first (core orbital)

        alpha_occupied_indices = np.array(alpha_occupied_indices[molecule.n_core_alpha_electrons:]) - molecule.n_core_spin_orbitals
        beta_occupied_indices = np.array(beta_occupied_indices[molecule.n_core_beta_electrons:]) - molecule.n_core_spin_orbitals

        # Separates the singles amplitudes into alpha and beta amplitudes

        t_ia_alpha = np.array([t_ia[i] for i in alpha_occupied_indices])
        t_ia_beta = np.array([t_ia[i] for i in beta_occupied_indices])

        n_alpha -= molecule.n_core_alpha_electrons
        n_beta -= molecule.n_core_beta_electrons
        n_occ -= molecule.n_core_alpha_electrons + molecule.n_core_beta_electrons

        # Finds the norm of both alpha and beta amplitudes, weighted by number of alpha and beta electrons

        t_ia_norm_alpha = n_alpha / n_occ * np.linalg.norm(t_ia_alpha)
        t_ia_norm_beta = n_beta / n_occ * np.linalg.norm(t_ia_beta)

        # Calculates total norm of singles amplitudes

        t_ia_norm = t_ia_norm_alpha + t_ia_norm_beta
    
    else:

        # Removes core orbitals from occupied count

        n_occ -= molecule.n_core_orbitals

        # The T1 diagnostic always wants the number of spin orbitals (at least to match ORCA)

        n_occ *= 2
       
        t_ia_norm = np.linalg.norm(t_ia)

    # Calculates the T1 diagnostic

    T1_diagnostic = t_ia_norm / np.sqrt(n_occ)

    log(f"\n  Norm of singles amplitudes:         {t_ia_norm:13.10f}", calculation, 1, silent=silent)
    log(f"  Value of T1 diagnostic:             {T1_diagnostic:13.10f}", calculation, 1, silent=silent)


    return










def find_and_print_largest_amplitudes(t_ia: ndarray, t_ijab: ndarray, n_occ: int, calculation: Calculation, spin_orbital_labels_sorted: list, silent: bool) -> None:
    
    """
    
    Searches for and prints the largest singles and doubles amplitudes.

    Args:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        n_occ (int): Number of occupied spin or spatial orbitals
        calculation (Calculation): Calculation object
        spin_orbital_labels_sorted (list): Energy ordering of alpha and beta spin orbitals
        silent (bool): Cancel logging
    
    """

    log("\n  Searching for largest amplitudes...        ", calculation, 2, end="", silent=silent); sys.stdout.flush()

    reference = calculation.reference

    # Flattened absolute amplitudes and their raw indices

    t_ijab_flat = np.abs(t_ijab).ravel()
    t_ia_flat = np.abs(t_ia).ravel()

    idx_ijab = np.vstack(np.unravel_index(np.arange(t_ijab_flat.size), t_ijab.shape)).T
    idx_ia = np.vstack(np.unravel_index(np.arange(t_ia_flat.size), t_ia.shape)).T

    # Adjust virtual indices for occupied offset

    idx_ijab[:, 2:] += n_occ
    idx_ia[:, 1] += n_occ

    # Build singles index rows to match doubles shape (use -1 for unused slots)

    singles = np.full((idx_ia.shape[0], 4), -1, dtype=int)
    singles[:, 0] = idx_ia[:, 0]
    singles[:, 2] = idx_ia[:, 1] 

    # Combine amplitudes and indices, then sort once

    amplitudes = np.concatenate([t_ijab_flat, t_ia_flat])
    indices = np.vstack([idx_ijab, singles])

    order = np.argsort(-amplitudes)       
    largest_amplitudes = amplitudes[order]
    indices_ordered = indices[order]

    if reference == "UHF":

        spin_orbital_labels_sorted.extend(["ERR"] * n_occ)
        spin_orbital_labels_sorted = np.array(spin_orbital_labels_sorted)

        # Lets me fiddle with the array

        indices_temporary = indices_ordered.astype(object)

        # Maps from orbital number to spin orbital label

        indices_temporary = spin_orbital_labels_sorted[indices_ordered]

        # Disallow alpha -> beta transitions

        mask = np.array([a[1][-1] == a[3][-1] and a[0][-1] == a[2][-1] for a in indices_temporary])

        indices_temporary = indices_temporary[mask]
        largest_amplitudes = largest_amplitudes[mask]

        # Make sure alpha transitions appear before beta transitions

        def fix_row(row):

            if row[1].endswith("a") or row[0].endswith("b"): 
                
                row[0], row[1] = row[1], row[0]
                row[2], row[3] = row[3], row[2]

            return row

        indices_temporary = np.array([fix_row(row) for row in indices_temporary])

        # Eliminates non-unique indices and corresponding values

        _, unique_idx = np.unique(indices_temporary, axis=0, return_index=True)

        indices_temporary = indices_temporary[np.sort(unique_idx)]
        largest_amplitudes = largest_amplitudes[np.sort(unique_idx)]

        indices_ordered = indices_temporary


    # Convert from computer counting to human counting

    if reference == "RHF": 
        
        indices_ordered += 1


    log(f"[Done]", calculation, 2, silent=silent)

    log("\n  Largest amplitudes:\n", calculation, 2, silent=silent)

    n_printing_indices = calculation.print_n_amplitudes

    # Prevents too high number of indices requested to be printed

    n_printing_indices = len(indices_ordered) if n_printing_indices > len(indices_ordered) else n_printing_indices

    for i in range(n_printing_indices):

        # Makes all the indices length three, pads with spaces

        idx_a_1, idx_b_1, idx_a_2, idx_b_2 = [f"{indices_ordered[i][j]:<3}" for j in (0, 1, 2, 3)]

        value = largest_amplitudes[i]

        stars = "~~~~~~~~  " 

        # Accounts for size difference with spin-appended indices

        space, antispace = (" ", "") if reference == "RHF" else ("", " ")

        # If its only a single excitation, use stars for the unchanged indices

        left_excitation = f"{idx_a_1}-> {space}{idx_a_2}{antispace}" if idx_a_1 != idx_a_2 else f"{stars}"
        right_excitation = f"{idx_b_1}-> {space}{idx_b_2}{antispace}" if idx_b_1 != idx_b_2 else f"{stars}"

        # Only print when the value is above zero

        if value > 1e-6:

            log(f"    {left_excitation}   {right_excitation}  :    {value:6f}", calculation, 2, silent=silent)


    return










def run_restricted_LCCD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple) -> tuple:

    """
    
    Updates the amplitudes for restricted LCCD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (array): Amplitudes
        e_denominators (array): Epsilons tensor

    Returns:
        t_amplitudes (array): Updated amplitudes

    """

    _, t_ijab, _, _ = t_amplitudes
    _, e_ijab, _, _ = e_denominators

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", g[o, v, v, o], t_ijab, optimize=True) 
    t_ijab_temporary -= np.einsum("ciak,kjcb->ijab", g[v, o, v, o], t_ijab, optimize=True) 
    t_ijab_temporary -= np.einsum("icak,kjbc->ijab", g[o, v, v, o], t_ijab, optimize=True) 
    t_ijab_temporary -= np.einsum("cibk,kjac->ijab", g[v, o, v, o], t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = None, t_ijab, None, None

    return t_amplitudes










def run_unrestricted_LCCD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple) -> tuple:

    """
    
    Updates the amplitudes for unrestricted LCCD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (array): Amplitudes
        e_denominators (array): Epsilons tensor

    Returns:
        t_amplitudes (array): Updated amplitudes

    """

    _, t_ijab, _, _ = t_amplitudes
    _, e_ijab, _, _ = e_denominators

    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = None, t_ijab, None, None

    return t_amplitudes










def run_restricted_CCD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, w: ndarray) -> tuple:

    """
    
    Updates the amplitudes for restricted CCD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        w (array): Antisymmetric spatial orbital integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    _, t_ijab, _, _ = t_amplitudes
    _, e_ijab, _, _ = e_denominators

    F_ik = np.einsum("cdkl,ilcd->ik", w[v, v, o, o], t_ijab, optimize=True)
    F_ca = -1 * np.einsum("cdkl,klad->ca", w[v, v, o, o], t_ijab, optimize=True) 

    # Intermediates based on two-electron integrals

    W_ijkl = g[o, o, o, o] + np.einsum("cdkl,ijcd->ijkl", g[v, v, o, o], t_ijab, optimize=True)
    W_icak = g[o, v, v, o] - (1 / 2) * np.einsum("dclk,ilda->icak", g[v, v, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum("dclk,ilad->icak", w[v, v, o, o], t_ijab, optimize=True)
    W_ciak = g[v, o, v, o] - (1 / 2) * np.einsum("cdlk,ilda->ciak", g[v, v, o, o], t_ijab, optimize=True) 

    # Updating doubles amplitudes

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", W_ijkl, t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += np.einsum("ca,ijcb->ijab", F_ca , t_ijab, optimize=True) - np.einsum("ik,kjab->ijab", F_ik, t_ijab, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", W_icak, t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", W_ciak, t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("icak,kjbc->ijab", W_icak, t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", W_ciak, t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = None, t_ijab, None, None

    return t_amplitudes










def run_unrestricted_CCD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple) -> tuple:

    """
    
    Updates the amplitudes for unrestricted CCD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    _, t_ijab, _, _ = t_amplitudes
    _, e_ijab, _, _ = e_denominators

    # Calculates contribution from LCCD

    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    # Calculates contribution from full CCD

    t_ijab_temporary += - (1 / 2) * permute(np.einsum("cdkl,ijac,klbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 2, 3) 
    t_ijab_temporary += - (1 / 2) * permute(np.einsum("cdkl,ikab,jlcd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)
    t_ijab_temporary += (1 / 4) * np.einsum("cdkl,ijcd,klab->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True)
    t_ijab_temporary += permute(np.einsum("cdkl,ikac,jlbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)

    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = None, t_ijab, None, None

    return t_amplitudes










def run_restricted_LCCSD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, w: ndarray) -> tuple:

    """
    
    Updates the amplitudes for restricted LCCSD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        w (array): Physicists notation antisymmetric spatial orbital integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    # Updates the singles amplitudes

    t_ia_temporary = np.einsum("icak,kc->ia", w[o, v, v, o], t_ia, optimize=True) 
    t_ia_temporary += np.einsum("cdak,ikcd->ia", w[v, v, v, o], t_ijab, optimize=True) 
    t_ia_temporary += -1 * np.einsum("ickl,klac->ia", w[o, v, o, o], t_ijab, optimize=True) 

    # Updates the doubles amplitudes

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += np.einsum("icab,jc->ijab", g[o, v, v, v], t_ia, optimize=True) - np.einsum("ijak,kb->ijab", g[o, o, v, o], t_ia, optimize=True) 
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", g[v, o, v, o], t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("icak,kjbc->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", g[v, o, v, o], t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_unrestricted_LCCSD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, F: ndarray) -> tuple:

    """
    
    Updates the amplitudes for unrestricted LCCSD. 

    Implementation of 10.1002/9780470125915.ch2, linearised.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        F (array): Fock matrix in spin orbital basis

    Returns:
        t_amplitudes (array): Updated amplitudes

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    # Equations from Crawford guide to coupled cluster, linearised, singles

    t_ia_temporary = F[o, v] + np.einsum("ac,ic->ia", F[v, v], t_ia, optimize=True) 
    t_ia_temporary += np.einsum("kc,ikac->ia", F[o, v], t_ijab, optimize=True) - np.einsum("ki,ka->ia", F[o, o], t_ia, optimize=True)
    t_ia_temporary += np.einsum("kaci,kc->ia", g[o, v, v, o], t_ia, optimize=True) 
    t_ia_temporary += (1 / 2) * np.einsum("kacd,kicd->ia", g[o, v, v, v], t_ijab, optimize=True) - (1 / 2) * np.einsum("klci,klca->ia", g[o, o, v, o], t_ijab, optimize=True)

    # Equations from Crawford guide to coupled cluster, linearised, connected doubles, shared with LCCD

    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    # Equations from Crawford guide to coupled cluster, linearised, doubles

    t_ijab_temporary += permute(np.einsum("bc,ijac->ijab", F[v, v], t_ijab, optimize=True), 2, 3) 
    t_ijab_temporary+= -1 * permute(np.einsum("kj,ikab->ijab", F[o, o], t_ijab, optimize=True), 0, 1)
    t_ijab_temporary += permute(np.einsum("abcj,ic->ijab", g[v, v, v, o], t_ia, optimize=True), 0, 1) 
    t_ijab_temporary += -1 * permute(np.einsum("kbij,ka->ijab", g[o, v, o, o], t_ia, optimize=True), 2, 3)

    t_ia += e_ia * t_ia_temporary
    t_ijab += e_ijab * t_ijab_temporary

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_restricted_QCISD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, w: ndarray) -> tuple:

    """
    
    Updates the amplitudes for restricted QCISD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        w (array): Physicists notation antisymmetric spatial orbital integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    # Curly F intermediates - formed from CCSD code

    F_ik = np.einsum("cdkl,ilcd->ik", w[v, v, o, o], t_ijab, optimize=True) 
    F_ca = - np.einsum("cdkl,klad->ca", w[v, v, o, o], t_ijab, optimize=True) 
    F_ck = np.einsum("cdkl,ld->ck", w[v, v, o, o], t_ia, optimize=True)

    # Curly W intermediates - formed from CCSD code

    W_ijkl = g[o, o, o, o] + np.einsum("cdkl,ijcd->ijkl", g[v, v, o, o], t_ijab, optimize=True) 
    W_icak = g[o, v, v, o] - (1 / 2) * np.einsum("dclk,ilda->icak", g[v, v, o, o], t_ijab, optimize=True)  + (1 / 2) * np.einsum("dclk,ilad->icak", w[v, v, o, o], t_ijab, optimize=True)
    W_ciak = g[v, o, v, o] - (1 / 2) * np.einsum("cdlk,ilda->ciak", g[v, v, o, o], t_ijab, optimize=True) 

    # Updating singles amplitudes

    t_ia_temporary = np.einsum("ca,ic->ia", F_ca, t_ia, optimize=True) - np.einsum("ik,ka->ia", F_ik, t_ia, optimize=True) + np.einsum("ck,kica->ia", F_ck, 2 * t_ijab - t_ijab.swapaxes(0, 1), optimize=True)
    t_ia_temporary += + np.einsum("icak,kc->ia", w[o, v, v, o], t_ia, optimize=True) + np.einsum("cdak,ikcd->ia", w[v, v, v, o], t_ijab, optimize=True)
    t_ia_temporary += - np.einsum("ickl,klac->ia", w[o, v, o, o], t_ijab, optimize=True)

    # Updating doubles amplitudes

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", W_ijkl, t_ijab, optimize=True) + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += np.einsum("ca,ijcb->ijab", F_ca, t_ijab, optimize=True) - np.einsum("ik,kjab->ijab", F_ik, t_ijab, optimize=True)
    t_ijab_temporary += np.einsum("icab,jc->ijab", g[o, v, v, v], t_ia, optimize=True) - np.einsum("ijak,kb->ijab", g[o, o, v, o], t_ia, optimize=True) 
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", W_icak, t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", W_ciak, t_ijab, optimize=True) - np.einsum("icak,kjbc->ijab", W_icak, t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", W_ciak, t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_unrestricted_QCISD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, F: ndarray) -> tuple:

    """
    
    Updates the amplitudes for unrestricted QCISD.

    All equations from Stanton paper on DPD coupled cluster (10.1063/1.460620) - curtailed by me into QCISD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        F (array): Fock matrix in spin orbital basis

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    kronecker_delta = np.eye(F.shape[1])

    # Builds curly F intermediates (Fock matrix intermediates)

    F_ae = F[v, v] - kronecker_delta[v, v] * F[v, v]  - (1 / 2) * np.einsum("mnaf,mnef->ae", t_ijab, g[o, o, v, v], optimize=True)
    F_mi = F[o, o] - kronecker_delta[o, o] * F[o, o]  + (1 / 2) * np.einsum("inef,mnef->mi", t_ijab, g[o, o, v, v], optimize=True)
    F_me = F[o, v] + np.einsum("nf,mnef->me", t_ia, g[o, o, v, v], optimize=True) 
    
    # Builds curly W intermediates (two-electron intermediates)

    W_mnij = g[o, o, o, o]  + (1 / 4) * np.einsum("ijef,mnef->mnij", t_ijab, g[o, o, v, v], optimize=True)
    W_abef = g[v, v, v, v] + (1 / 4) * np.einsum("mnab,mnef->abef", t_ijab, g[o, o, v, v], optimize=True)
    W_mbej = g[o, v, v, o]  - np.einsum("jnfb,mnef->mbej", (1 / 2) * t_ijab, g[o, o, v, v], optimize=True)

    # Builds t_ia tensor from intermediates

    t_ia_temporary = F[o, v] + np.einsum("ie,ae->ia", t_ia, F_ae, optimize=True) - np.einsum("ma,mi->ia", t_ia, F_mi, optimize=True) 
    t_ia_temporary += np.einsum("imae,me->ia", t_ijab, F_me, optimize=True) - np.einsum("nf,naif->ia", t_ia, g[o, v, o, v], optimize=True) 
    t_ia_temporary += - (1 / 2) * np.einsum("imef,maef->ia", t_ijab, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnae,nmei->ia", t_ijab, g[o, o, v, o], optimize=True)

    # Builds t_ijab tensor from intermediates, pairs of terms from Stanton

    t_ijab_temporary = g[o, o, v, v] + permute(np.einsum("ijae,be->ijab", t_ijab, F_ae, optimize=True), 2, 3) 
    t_ijab_temporary += -1 * permute(np.einsum("imab,mj->ijab", t_ijab, F_mi, optimize=True), 0, 1)
    t_ijab_temporary += (1 / 2) * np.einsum("mnab,mnij->ijab", t_ijab, W_mnij, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijef,abef->ijab", t_ijab, W_abef, optimize=True)
    t_ijab_temporary += permute(permute(np.einsum("ijmabe->ijab", np.einsum("imae,mbej->ijmabe", t_ijab, W_mbej, optimize=True), optimize=True), 2, 3), 0, 1)
    t_ijab_temporary += permute(np.einsum("ie,abej->ijab", t_ia, g[v, v, v, o], optimize=True), 0, 1) 
    t_ijab_temporary += -1 * permute(np.einsum("ma,mbij->ijab", t_ia, g[o, v, o, o], optimize=True), 2, 3)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary
    
    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_restricted_CCSD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, w: ndarray, F: ndarray) -> tuple:

    """
    
    Updates the amplitudes for restricted CCSD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        w (array): Physicists notation antisymmetric spatial orbital integrals
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """
    
    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    # Intermediates based on the Fock matrix

    F_ik = F[o, o] + np.einsum("cdkl,ilcd->ik", w[v, v, o, o], t_ijab, optimize=True) + np.einsum("cdkl,ic,ld->ik", w[v, v, o, o], t_ia, t_ia, optimize=True) 
    F_ca = F[v, v] - np.einsum("cdkl,klad->ca", w[v, v, o, o], t_ijab, optimize=True) - np.einsum("cdkl,ka,ld->ca", w[v, v, o, o], t_ia, t_ia, optimize=True) 
    F_ck = np.einsum("cdkl,ld->ck", w[v, v, o, o], t_ia, optimize=True)

    L_ik = F_ik + np.einsum("cilk,lc->ik", w[v, o, o, o], t_ia, optimize=True)
    L_ca = F_ca + np.einsum("dcka,kd->ca", w[v, v, o, v], t_ia, optimize=True)
    
    # Intermediates based on two-electron integrals

    W_ijkl = g[o, o, o, o] + np.einsum("cilk,jc->ijkl", g[v, o, o, o], t_ia, optimize=True) + np.einsum("cjkl,ic->ijkl", g[v, o, o, o], t_ia, optimize=True)
    W_ijkl += np.einsum("cdkl,ijcd->ijkl", g[v, v, o, o], t_ijab, optimize=True) + np.einsum("cdkl,ic,jd->ijkl", g[v, v, o, o], t_ia, t_ia, optimize=True)

    W_cdab = g[v, v, v, v] - np.einsum("dcka,kb->cdab", g[v, v, o, v], t_ia, optimize=True) - np.einsum("cdkb,ka->cdab", g[v, v, o, v], t_ia, optimize=True)

    W_icak = g[o, v, v, o] - np.einsum("cikl,la->icak", g[v, o, o, o], t_ia, optimize=True) + np.einsum("cdka,id->icak", g[v, v, o, v], t_ia, optimize=True)
    W_icak += - (1 / 2) * np.einsum("dclk,ilda->icak", g[v, v, o, o], t_ijab, optimize=True) - np.einsum("dclk,id,la->icak", g[v, v, o, o], t_ia, t_ia, optimize=True) 
    W_icak += (1 / 2) * np.einsum("dclk,ilad->icak", w[v, v, o, o], t_ijab, optimize=True)
    
    W_ciak = g[v, o, v, o] - np.einsum("cilk,la->ciak", g[v, o, o, o], t_ia, optimize=True) + np.einsum("dcka,id->ciak", g[v, v, o, v], t_ia, optimize=True)
    W_ciak += - (1 / 2) * np.einsum("cdlk,ilda->ciak", g[v, v, o, o], t_ijab, optimize=True) - np.einsum("cdlk,id,la->ciak", g[v, v, o, o], t_ia, t_ia, optimize=True)

    # Updating singles amplitudes

    t_ia_temporary = np.einsum("ca,ic->ia", F_ca - F[v, v], t_ia, optimize=True) - np.einsum("ik,ka->ia", F_ik - F[o, o], t_ia, optimize=True) 
    t_ia_temporary += -1 * np.einsum("ickl,klac->ia", w[o, v, o, o], t_ijab, optimize=True) - np.einsum("ickl,ka,lc->ia", w[o, v, o, o], t_ia, t_ia, optimize=True)
    t_ia_temporary += np.einsum("ck,kica->ia", F_ck, 2 * t_ijab - t_ijab.swapaxes(0, 1), optimize=True)
    t_ia_temporary += np.einsum("ck,ic,ka->ia", F_ck, t_ia, t_ia, optimize=True) 
    t_ia_temporary += np.einsum("icak,kc->ia", w[o, v, v, o], t_ia, optimize=True) 
    t_ia_temporary += np.einsum("cdak,ikcd->ia", w[v, v, v, o], t_ijab, optimize=True)
    t_ia_temporary += np.einsum("cdak,ic,kd->ia", w[v, v, v, o], t_ia, t_ia, optimize=True) 

    # Updating doubles amplitudes

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", W_ijkl, t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijkl,ka,lb->ijab", W_ijkl, t_ia, t_ia, optimize=True)
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", W_cdab, t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ic,jd->ijab", W_cdab, t_ia, t_ia, optimize=True)
    t_ijab_temporary += np.einsum("ca,ijcb->ijab", L_ca - F[v, v], t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("ik,kjab->ijab", L_ik - F[o, o], t_ijab, optimize=True)
    t_ijab_temporary += np.einsum("icab,jc->ijab", g[o, v, v, v], t_ia, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("ickb,ka,jc->ijab", g[o, v, o, v], t_ia, t_ia, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("ijak,kb->ijab", g[o, o, v, o], t_ia, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("icak,jc,kb->ijab", g[o, v, v, o], t_ia, t_ia, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", W_icak, t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("ciak,kjcb->ijab", W_ciak, t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("icak,kjbc->ijab", W_icak, t_ijab, optimize=True) 
    t_ijab_temporary += -1 * np.einsum("cibk,kjac->ijab", W_ciak, t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_unrestricted_CCSD_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, F: ndarray) -> tuple:

    """
    
    Updates the amplitudes for unrestricted CCSD.

    All equations from Stanton paper on DPD coupled cluster (10.1063/1.460620).

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilons tensors
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    kronecker_delta = np.eye(F.shape[1])

    # Build tau tensors, all equations from Stanton paper on DPD coupled cluster, referenced by Crawford tutorials

    tau_tilde_ijab = t_ijab + (1 / 2) * (np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True))
    tau_ijab = t_ijab + np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True)

    # Builds curly F intermediates (Fock matrix intermediates)

    F_ae = F[v, v] - kronecker_delta[v, v] * F[v, v] - (1 / 2) * np.einsum("me,ma->ae", F[o, v], t_ia, optimize=True) 
    F_ae += np.einsum("mf,mafe->ae", t_ia, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnaf,mnef->ae", tau_tilde_ijab, g[o, o, v, v], optimize=True)

    F_mi = F[o, o] - kronecker_delta[o, o] * F[o, o]  + (1 / 2) * np.einsum("ie,me->mi", t_ia, F[o, v], optimize=True)
    F_mi += np.einsum("ne,mnie->mi", t_ia, g[o, o, o, v], optimize=True) + (1 / 2) * np.einsum("inef,mnef->mi", tau_tilde_ijab, g[o, o, v, v], optimize=True)

    F_me = F[o, v] + np.einsum("nf,mnef->me", t_ia, g[o, o, v, v], optimize=True) 
    
    # Builds curly W intermediates (two-electron intermediates)

    W_mnij = g[o, o, o, o] + permute(np.einsum("je,mnie->mnij", t_ia, g[o, o, o, v], optimize=True), 2, 3) 
    W_mnij += (1 / 4) * np.einsum("ijef,mnef->mnij", tau_ijab, g[o, o, v, v], optimize=True)

    W_abef = g[v, v, v, v] - permute(np.einsum("mb,amef->abef", t_ia, g[v, o, v, v], optimize=True), 0, 1) 
    W_abef += (1 / 4) * np.einsum("mnab,mnef->abef", tau_ijab, g[o, o, v, v], optimize=True)

    W_mbej = g[o, v, v, o] + np.einsum("jf,mbef->mbej", t_ia, g[o, v, v, v], optimize=True) 
    W_mbej += -1 * np.einsum("nb,mnej->mbej", t_ia, g[o, o, v, o], optimize=True) 
    W_mbej += -1 * np.einsum("jnfb,mnef->mbej", (1 / 2) * t_ijab + np.einsum("jf,nb->jnfb", t_ia, t_ia, optimize=True), g[o, o, v, v], optimize=True)

    # Builds t_ia tensor from intermediates

    t_ia_temporary = F[o, v] + np.einsum("ie,ae->ia", t_ia, F_ae, optimize=True) - np.einsum("ma,mi->ia", t_ia, F_mi, optimize=True) 
    t_ia_temporary += np.einsum("imae,me->ia", t_ijab, F_me, optimize=True) - np.einsum("nf,naif->ia", t_ia, g[o, v, o, v], optimize=True) 
    t_ia_temporary += - (1 / 2) * np.einsum("imef,maef->ia", t_ijab, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnae,nmei->ia", t_ijab, g[o, o, v, o], optimize=True)
    
    # Builds t_ijab tensor from intermediates, pairs of terms from Stanton

    t_ijab_temporary = g[o, o, v, v] + permute(np.einsum("ijae,be->ijab", t_ijab, F_ae - (1 / 2) * np.einsum("mb,me->be", t_ia, F_me,optimize=True), optimize=True), 2, 3) 
    t_ijab_temporary += -1 * permute(np.einsum("imab,mj->ijab", t_ijab, F_mi + (1 / 2) * np.einsum("je,me->mj", t_ia, F_me, optimize=True),optimize=True), 0, 1)
    t_ijab_temporary += (1 / 2) * np.einsum("mnab,mnij->ijab", tau_ijab, W_mnij, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("ijef,abef->ijab", tau_ijab, W_abef, optimize=True)
    t_ijab_temporary += permute(permute(np.einsum("ijmabe->ijab", np.einsum("imae,mbej->ijmabe", t_ijab, W_mbej, optimize=True) - np.einsum("ie,ma,mbej->ijmabe", t_ia, t_ia, g[o, v, v, o], optimize=True), optimize=True), 2, 3), 0, 1)
    t_ijab_temporary += permute(np.einsum("ie,abej->ijab", t_ia, g[v, v, v, o], optimize=True), 0, 1) 
    t_ijab_temporary += -1 * permute(np.einsum("ma,mbij->ijab", t_ia, g[o, v, o, o], optimize=True), 2, 3)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary
    
    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_restricted_CC2_iteration(o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, molecular_orbitals: ndarray, integrals: Integrals) -> tuple:
    
    """
    
    Updates the amplitudes for restricted CC2. Uses T1 dressing.

    Taken from the Jiang2024 version of T1-dressed CCSDT then curtailed by me into CC2.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilon denominators
        molecular_orbitals (array): Molecular orbitals
        integrals (Integrals): Molecular integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes
    
    """

    t_ia, _, _, _ = t_amplitudes
    e_ia, e_ijab, _, _ = e_denominators

    # Combines molecular orbitals with T1 dressing

    X, Y = molecular_orbitals.copy(), molecular_orbitals.copy()

    X[:, v] -= molecular_orbitals[:, o] @ t_ia
    Y[:, o] += molecular_orbitals[:, v] @ t_ia.T
    
    # Full H_core transformation

    h_hat = np.einsum("ap,bq,ab->pq", X, Y, integrals.H_core, optimize=True)

    # Transform only the ERI blocks actually needed

    g_vovo = np.einsum("ap,bq,gr,ds,abgd->pqrs", X[:, v], Y[:, o], X[:, v], Y[:, o], integrals.ERI_AO, optimize=True)
    g_ovvv = np.einsum("ap,bq,gr,ds,abgd->pqrs", X[:, o], Y[:, v], X[:, v], Y[:, v], integrals.ERI_AO, optimize=True)
    g_ooov = np.einsum("ap,bq,gr,ds,abgd->pqrs", X[:, o], Y[:, o], X[:, o], Y[:, v], integrals.ERI_AO, optimize=True)
    g_oovo = np.einsum("ap,bq,gr,ds,abgd->pqrs", X[:, o], Y[:, o], X[:, v], Y[:, o], integrals.ERI_AO, optimize=True)
    g_ovoo = np.einsum("ap,bq,gr,ds,abgd->pqrs", X[:, o], Y[:, v], X[:, o], Y[:, o], integrals.ERI_AO, optimize=True)

    # Build only the needed F_hat blocks

    F_vo = h_hat[v, o] + 2 * np.einsum("kkai->ai", g_oovo, optimize=True) - np.einsum("kiak->ai", g_oovo, optimize=True)
    F_ov = h_hat[o, v] + 2 * np.einsum("kkia->ia", g_ooov, optimize=True) - np.einsum("kaik->ia", g_ovoo, optimize=True)

    # CC2 doubles amplitudes are perturbative, not iteratively updated like CCSD, uses MP2-like expression

    t_ijab = ci.build_MP2_t_amplitudes(g_vovo.transpose(1, 3, 0, 2), e_ijab)
    
    # Antisymmetrised doubles combination used in the CC2 singles residual

    u_ijab = 2 * t_ijab - t_ijab.swapaxes(2, 3)

    # Temporary arrays for CC2 singles

    A_ia = np.einsum("kicd,kcad->ia", u_ijab, g_ovvv, optimize=True)
    B_ia = -1 * np.einsum("klac,kilc->ia", u_ijab, g_ooov, optimize=True)
    C_ia = np.einsum("kc,ikac->ia", F_ov, u_ijab, optimize=True)

    # Contributions to CC2 singles amplitudes

    residual_ia = F_vo.swapaxes(0, 1) + A_ia + B_ia + C_ia

    # Updates amplitudes

    t_ia += e_ia * residual_ia

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_restricted_CC3_iteration(o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, molecular_orbitals: ndarray, integrals: Integrals) -> tuple:

    """
    
    Updates the amplitudes for restricted CC3. Uses T1 dressing.

    Taken from the Jiang2024 version of T1-dressed CCSDT then curtailed by me into CC3.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilon denominators
        molecular_orbitals (array): Molecular orbitals
        integrals (Integrals): Molecular integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes
    
    """

    t_ia, t_ijab, _, _ = t_amplitudes
    e_ia, e_ijab, e_ijkabc, _ = e_denominators

    # Combines molecular orbitals with T1 dressing

    X, Y = molecular_orbitals.copy(), molecular_orbitals.copy()

    X[:, v] -= molecular_orbitals[:, o] @ t_ia
    Y[:, o] += molecular_orbitals[:, v] @ t_ia.T
    
    # Transforms the atomic orbital basis molecular integrals with T1 dressing into chemists' notation

    g_hat = np.einsum("ap,bq,gr,ds,abgd->pqrs", X, Y, X, Y, integrals.ERI_AO, optimize=True)
    h_hat = np.einsum("ap,bq,ab->pq", X, Y, integrals.H_core, optimize=True)

    # Repeatedly used arrays in CCSD / CC3

    l_hat = 2 * g_hat - g_hat.swapaxes(1, 3)
    u_ijab = 2 * t_ijab - t_ijab.swapaxes(2, 3)

    # The T1 dressed Fock matrix - not using "o" as the full Fock matrix needs to be transformed

    F_hat = h_hat + np.einsum("kkpq->pq", l_hat[slice(0, o.stop), slice(0, o.stop), :, :], optimize=True)

    # Temporary arrays for CCSD singles

    A_ia = np.einsum("kicd,kcad->ia", u_ijab, g_hat[o, v, v, v], optimize=True)
    B_ia = -1 * np.einsum("klac,kilc->ia", u_ijab, g_hat[o, o, o, v], optimize=True)
    C_ia = np.einsum("kc,ikac->ia", F_hat[o, v], u_ijab, optimize=True)

    # Temporary arrays for CCSD doubles

    beta_ijkl = g_hat[o, o, o, o].transpose(1, 3, 0, 2) + np.einsum("ijcd,kcld->ijkl", t_ijab, g_hat[o, v, o, v], optimize=True)
    gamma_kiac = g_hat[o, o, v, v] - (1 / 2) * np.einsum("liad,kdlc->kiac", t_ijab, g_hat[o, v, o, v], optimize=True)

    delta_aikc = 2 * g_hat[v, o, o, v] - g_hat[o, o, v, v].transpose(2, 1, 0, 3) 
    delta_aikc += (1 / 2) * np.einsum("ilad,ldkc->aikc", u_ijab, 2 * g_hat[o, v, o, v] - g_hat[o, v, o, v].swapaxes(1, 3), optimize=True)
    
    F_tilde_tilde_bc = F_hat[v, v] - np.einsum("klbd,ldkc->bc", u_ijab, g_hat[o, v, o, v], optimize=True)
    F_tilde_tilde_kj = F_hat[o, o] + np.einsum("ljcd,kdlc->kj", u_ijab, g_hat[o, v, o, v], optimize=True)
    
    A_ijab = np.einsum("ijcd,acbd->ijab", t_ijab, g_hat[v, v, v, v], optimize=True)
    B_ijab = np.einsum("klab,ijkl->ijab", t_ijab, beta_ijkl, optimize=True)
    C_ijab = -1 * np.einsum("kjbc,kiac->ijab", t_ijab, gamma_kiac, optimize=True)
    D_ijab = (1 / 2) * np.einsum("jkbc,aikc->ijab", u_ijab, delta_aikc, optimize=True)
    E_ijab = np.einsum("ijac,bc->ijab", t_ijab, F_tilde_tilde_bc, optimize=True)
    G_ijab = -1 * np.einsum("ikab,kj->ijab", t_ijab, F_tilde_tilde_kj, optimize=True)

    # This is not iterative, but approximated from the doubles amplitudes

    t_ijkabc = mp.calculate_restricted_second_order_triples_amplitudes(e_ijkabc, t_ijab, g_hat, o, v)

    u_ijkabc = 2 * t_ijkabc - t_ijkabc.swapaxes(3, 4) - t_ijkabc.swapaxes(3, 5)

    # Triples contributions to the CCSD doubles residual

    temp_ijab = np.einsum("kc,ijkabc->ijab", F_hat[o, v], t_ijkabc - t_ijkabc.swapaxes(4, 5), optimize=True)
    temp_ijab += np.einsum("ackd,ijkcbd->ijab", g_hat[v, v, o, v], 2 * t_ijkabc - t_ijkabc.swapaxes(4, 5) - t_ijkabc.swapaxes(3, 5), optimize=True)
    temp_ijab += -1 * np.einsum("kilc,ljkcba->ijab", g_hat[o, o, o, v], u_ijkabc, optimize=True)

    # Contributions to CCSD amplitudes

    residual_ia = F_hat[v, o].swapaxes(0, 1) + A_ia + B_ia + C_ia

    residual_ijab = g_hat[v, o, v, o].transpose(1, 3, 0, 2) + A_ijab + B_ijab 
    residual_ijab += mp.permute_symmetric((1 / 2) * C_ijab + C_ijab.swapaxes(0, 1) + D_ijab + E_ijab + G_ijab, (0, 1), (2, 3))

    # Connected-triples contributions for CC3

    residual_ia += np.einsum("jbkc,ijkabc->ia", l_hat[o, v, o, v], t_ijkabc - t_ijkabc.swapaxes(3, 4), optimize=True) 
    residual_ijab += mp.permute_symmetric(temp_ijab, (0, 1), (2, 3)) 

    # Updates amplitudes

    t_ia += e_ia * residual_ia
    t_ijab += e_ijab * residual_ijab

    t_amplitudes = t_ia, t_ijab, None, None

    return t_amplitudes










def run_restricted_CCSDT_iteration(o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, molecular_orbitals: ndarray, integrals: Integrals) -> tuple:

    """
    
    Updates the amplitudes for restricted CCSDT. Uses T1 dressing.

    Implementation of 10.26434/chemrxiv-2024-xbnmh (CCSDT) via 10.26434/chemrxiv-2024-cvs8h (CCSD).

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple): Epsilon denominators
        molecular_orbitals (array): Molecular orbitals
        integrals (Integrals): Molecular integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes
        g_hat (array): Two-electron integrals with T1 dressing
        F_hat (array): Fock matrix with T1 dressing
        u_ijab (array): Antisymmetrised spin-restricted T2 amplitudes
    
    """
    
    def permute_short(array):

        # Corrected definition of the short permutation operator from 10.26434/chemrxiv-2024-xbnmh (the paper has a typo)

        return array + array.transpose(1, 0, 2, 4, 3, 5) + array.transpose(2, 1, 0, 5, 4, 3)


    t_ia, t_ijab, t_ijkabc, _ = t_amplitudes
    e_ia, e_ijab, e_ijkabc, _ = e_denominators

    # Combines molecular orbitals with T1 dressing

    X, Y = molecular_orbitals.copy(), molecular_orbitals.copy()

    X[:, v] -= molecular_orbitals[:, o] @ t_ia
    Y[:, o] += molecular_orbitals[:, v] @ t_ia.T
    
    # Transforms the atomic orbital basis molecular integrals with T1 dressing into chemists' notation
    g_hat = np.einsum("ap,bq,gr,ds,abgd->pqrs", X, Y, X, Y, integrals.ERI_AO, optimize=True)
    h_hat = np.einsum("ap,bq,ab->pq", X, Y, integrals.H_core, optimize=True)

    # Repeatedly used arrays in CCSDT

    l_hat = 2 * g_hat - g_hat.swapaxes(1, 3)
    u_ijab = 2 * t_ijab - t_ijab.swapaxes(2, 3)
    u_ijkabc = 2 * t_ijkabc - t_ijkabc.swapaxes(3, 4) - t_ijkabc.swapaxes(3, 5)

    # The T1 dressed Fock matrix - not using "o" as the full Fock matrix needs to be transformed

    F_hat = h_hat + np.einsum("kkpq->pq", l_hat[slice(0, o.stop), slice(0, o.stop), :, :], optimize=True)

    # Temporary arrays for CCSD singles

    A_ia = np.einsum("kicd,kcad->ia", u_ijab, g_hat[o, v, v, v], optimize=True)
    B_ia = -1 * np.einsum("klac,kilc->ia", u_ijab, g_hat[o, o, o, v], optimize=True)
    C_ia = np.einsum("kc,ikac->ia", F_hat[o, v], u_ijab, optimize=True)

    # Temporary arrays for CCSD doubles

    beta_ijkl = g_hat[o, o, o, o].transpose(1, 3, 0, 2) + np.einsum("ijcd,kcld->ijkl", t_ijab, g_hat[o, v, o, v], optimize=True)
    gamma_kiac = g_hat[o, o, v, v] - (1 / 2) * np.einsum("liad,kdlc->kiac", t_ijab, g_hat[o, v, o, v], optimize=True)

    delta_aikc = 2 * g_hat[v, o, o, v] - g_hat[o, o, v, v].transpose(2, 1, 0, 3) 
    delta_aikc += (1 / 2) * np.einsum("ilad,ldkc->aikc", u_ijab, 2 * g_hat[o, v, o, v] - g_hat[o, v, o, v].swapaxes(1, 3), optimize=True)
    
    F_tilde_tilde_bc = F_hat[v, v] - np.einsum("klbd,ldkc->bc", u_ijab, g_hat[o, v, o, v], optimize=True)
    F_tilde_tilde_kj = F_hat[o, o] + np.einsum("ljcd,kdlc->kj", u_ijab, g_hat[o, v, o, v], optimize=True)
    
    A_ijab = np.einsum("ijcd,acbd->ijab", t_ijab, g_hat[v, v, v, v], optimize=True)
    B_ijab = np.einsum("klab,ijkl->ijab", t_ijab, beta_ijkl, optimize=True)
    C_ijab = -1 * np.einsum("kjbc,kiac->ijab", t_ijab, gamma_kiac, optimize=True)
    D_ijab = (1 / 2) * np.einsum("jkbc,aikc->ijab", u_ijab, delta_aikc, optimize=True)
    E_ijab = np.einsum("ijac,bc->ijab", t_ijab, F_tilde_tilde_bc, optimize=True)
    G_ijab = -1 * np.einsum("ikab,kj->ijab", t_ijab, F_tilde_tilde_kj, optimize=True)

    # Temporary arrays for CCSDT triples

    chi_li = F_hat[o, o] + np.einsum("meld,imde->li", g_hat[o, v, o, v], u_ijab, optimize=True)
    chi_ad = F_hat[v, v] - np.einsum("meld,lmae->ad", g_hat[o, v, o, v], u_ijab, optimize=True)
    
    chi_ljmk = g_hat[o, o, o, o] + np.einsum("ldme,jkde->ljmk", g_hat[o, v, o, v], t_ijab, optimize=True)
    chi_bdce = g_hat[v, v, v, v] + np.einsum("ldme,lmbc->bdce", g_hat[o, v, o, v], t_ijab, optimize=True)
    chi_adli = g_hat[v, v, o, o] - np.einsum("lemd,miae->adli", g_hat[o, v, o, v], t_ijab, optimize=True)

    chi_aild = g_hat[v, o, o, v] - np.einsum("lemd,imae->aild", g_hat[o, v, o, v], t_ijab, optimize=True)
    chi_aild += np.einsum("ldme,imae->aild", g_hat[o, v, o, v], u_ijab, optimize=True)

    xi_cklj = g_hat[v, o, o, o] + np.einsum("ljmd,mkdc->cklj", g_hat[o, o, o, v], u_ijab, optimize=True)
    xi_cklj += -1 * np.einsum("ldmj,mkdc->cklj", g_hat[o, v, o, o], t_ijab, optimize=True)
    xi_cklj += np.einsum("cdle,kjde->cklj", g_hat[v, v, o, v], t_ijab, optimize=True)
    xi_cklj += -1 * np.einsum("ldmk,mjcd->cklj", g_hat[o, v, o, o], t_ijab, optimize=True)
    xi_cklj += np.einsum("ldme,mkjecd->cklj", g_hat[o, v, o, v], u_ijkabc, optimize=True)

    xi_ckbd = g_hat[v, o, v, v] - np.einsum("ld,lkbc->ckbd", F_hat[o, v], t_ijab, optimize=True)
    xi_ckbd += np.einsum("lkmd,lmcb->ckbd", g_hat[o, o, o, v], t_ijab, optimize=True) 
    xi_ckbd += -1 * np.einsum("beld,lkec->ckbd", g_hat[v, v, o, v], t_ijab, optimize=True)
    xi_ckbd += np.einsum("bdle,lkec->ckbd", g_hat[v, v, o, v], u_ijab, optimize=True) 
    xi_ckbd += -1 * np.einsum("celd,lkbe->ckbd", g_hat[v, v, o, v], t_ijab, optimize=True)
    xi_ckbd += -1 * np.einsum("ldme,mklecb->ckbd", g_hat[o, v, o, v], u_ijkabc, optimize=True)

    temp_ijab = np.einsum("kc,ijkabc->ijab", F_hat[o, v], t_ijkabc - t_ijkabc.swapaxes(4, 5), optimize=True)
    temp_ijab += np.einsum("ackd,ijkcbd->ijab", g_hat[v, v, o, v], 2 * t_ijkabc - t_ijkabc.swapaxes(4, 5) - t_ijkabc.swapaxes(3, 5), optimize=True)
    temp_ijab += -1 * np.einsum("kilc,ljkcba->ijab", g_hat[o, o, o, v], u_ijkabc, optimize=True)

    temp_ijkabc = np.einsum("ad,ijkdbc->ijkabc", chi_ad, t_ijkabc, optimize=True) 
    temp_ijkabc += -1 * np.einsum("li,ljkabc->ijkabc", chi_li, t_ijkabc, optimize=True) 
    temp_ijkabc += np.einsum("ljmk,ilmabc->ijkabc", chi_ljmk, t_ijkabc, optimize=True) 
    temp_ijkabc += -1 * np.einsum("adli,ljkdbc->ijkabc", chi_adli, t_ijkabc, optimize=True) 
    temp_ijkabc += np.einsum("bdce,ijkade->ijkabc", chi_bdce, t_ijkabc, optimize=True) 
    temp_ijkabc += -1 * np.einsum("bdli,ljkadc->ijkabc", chi_adli, t_ijkabc, optimize=True) 
    temp_ijkabc += -1 * np.einsum("cdli,ljkabd->ijkabc", chi_adli, t_ijkabc, optimize=True)
    temp_ijkabc += np.einsum("aild,ljkdbc->ijkabc", chi_aild, u_ijkabc, optimize=True)
    
    # Contributions to CCSD amplitudes

    residual_ia = F_hat[v, o].swapaxes(0, 1) + A_ia + B_ia + C_ia

    residual_ijab = g_hat[v, o, v, o].transpose(1, 3, 0, 2) + A_ijab + B_ijab 
    residual_ijab += mp.permute_symmetric((1 / 2) * C_ijab + C_ijab.swapaxes(0, 1) + D_ijab + E_ijab + G_ijab, (0, 1), (2, 3))

    # Contributions to CCSDT amplitudes

    residual_ia += np.einsum("jbkc,ijkabc->ia", l_hat[o, v, o, v], t_ijkabc - t_ijkabc.swapaxes(3, 4), optimize=True) 
    residual_ijab += mp.permute_symmetric(temp_ijab, (0, 1), (2, 3)) 

    residual_ijkabc = mp.permute_three_column_indices(np.einsum("ijad,ckbd->ijkabc", t_ijab, xi_ckbd, optimize=True) - np.einsum("ilab,cklj->ijkabc", t_ijab, xi_cklj, optimize=True)) 
    residual_ijkabc += permute_short(temp_ijkabc)

    # Updates amplitudes

    t_ia += e_ia * residual_ia
    t_ijab += e_ijab * residual_ijab
    t_ijkabc += e_ijkabc * residual_ijkabc 

    t_amplitudes = t_ia, t_ijab, t_ijkabc, None

    return t_amplitudes, g_hat, F_hat, u_ijab










def run_unrestricted_CCSDT_iteration(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, F: ndarray) -> tuple:

    """
    
    Updates the amplitudes for unrestricted CCSDT.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Singles amplitudes
        e_denominators (tuple): Epsilons tensor
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_amplitudes (tuple): Updated amplitudes

    """

    t_ia, t_ijab, t_ijkabc, _ = t_amplitudes
    e_ia, e_ijab, e_ijkabc, _ = e_denominators


    # Contributions from singles
    t_ia_temporary = np.einsum('ia->ia', F[o, v], optimize=True) + np.einsum('ab,ib->ia', F[v, v], t_ia, optimize=True) - np.einsum('ji,ja->ia', F[o, o], t_ia, optimize=True)
    t_ia_temporary += np.einsum('ajib,jb->ia', g[v, o, o, v], t_ia, optimize=True)

    # Contributions from connected doubles
    t_ia_temporary += np.einsum('jb,ijab->ia', F[o, v], t_ijab, optimize=True)

    # Contributions from connected doubles
    t_ia_temporary += (1 / 2) * np.einsum('ajbc,ijbc->ia', g[v, o, v, v], t_ijab, optimize=True) - (1 / 2) * np.einsum('jkib,jkab->ia', g[o, o, o, v], t_ijab, optimize=True)
    
    # Contributions from disconnected doubles
    t_ia_temporary += -np.einsum('jb,ja,ib->ia', F[o, v], t_ia, t_ia, optimize=True)
    t_ia_temporary += np.einsum('jkib,ka,jb->ia', g[o, o, o, v], t_ia, t_ia, optimize=True) - np.einsum('ajbc,jb,ic->ia', g[v, o, v, v], t_ia, t_ia, optimize=True)
    
    # Contributions from connected triples
    t_ia_temporary += (1 / 4) * np.einsum('jkbc,ijkabc->ia', g[o, o, v, v], t_ijkabc, optimize=True)

    # Contributions from disconnected triples
    t_ia_temporary += -np.einsum('jkbc,ka,jb,ic->ia', g[o, o, v, v], t_ia, t_ia, t_ia, optimize=True)
    t_ia_temporary += np.einsum('jkbc,jb,ikac->ia', g[o, o, v, v], t_ia, t_ijab, optimize=True)
    t_ia_temporary += -(1 / 2) * np.einsum('jkbc,ja,ikbc->ia', g[o, o, v, v], t_ia, t_ijab, optimize=True) - (1 / 2) * np.einsum('jkbc,ib,jkac->ia', g[o, o, v, v], t_ia, t_ijab, optimize=True)





    # Contributions from singles
    t_ijab_temporary = permute(np.einsum('abic,jc->ijab', g[v, v, o, v], t_ia, optimize=True), 1, 0) - permute(np.einsum('akij,kb->ijab', g[v, o, o, o], t_ia, optimize=True), 3, 2)
    t_ijab_temporary += np.einsum('ijab->ijab', g[o, o, v, v], optimize=True)

    # Contributions from connected doubles
    t_ijab_temporary += (1 / 2) * np.einsum('klij,klab->ijab', g[o, o, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum('abcd,ijcd->ijab', g[v, v, v, v], t_ijab, optimize=True)
    t_ijab_temporary += permute(np.einsum('ki,jkab->ijab', F[o, o], t_ijab, optimize=True), 1, 0) - permute(np.einsum('ac,ijbc->ijab', F[v, v], t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute(permute(np.einsum('akic,jkbc->ijab', g[v, o, o, v], t_ijab, optimize=True), 0, 1), 3, 2)

    # Contributions from disconnected doubles
    t_ijab_temporary += np.einsum('abcd,ic,jd->ijab', g[v, v, v, v], t_ia, t_ia, optimize=True)
    t_ijab_temporary += np.einsum('klij,ka,lb->ijab', g[o, o, o, o], t_ia, t_ia, optimize=True)
    t_ijab_temporary += permute(permute(-np.einsum('akic,kb,jc->ijab', g[v, o, o, v], t_ia, t_ia, optimize=True), 0, 1), 3, 2)

    # Contributions from connected triples
    t_ijab_temporary += np.einsum('kc,ijkabc->ijab', F[o, v], t_ijkabc, optimize=True)
    t_ijab_temporary += permute((1 / 2) * np.einsum('klic,jklabc->ijab', g[o, o, o, v], t_ijkabc, optimize=True), 1, 0)
    t_ijab_temporary += permute(-(1 / 2) * np.einsum('akcd,ijkbcd->ijab', g[v, o, v, v], t_ijkabc, optimize=True), 3, 2)

    # Contributions from disconnected triples
    t_ijab_temporary += permute(np.einsum('akcd,kc,ijbd->ijab', g[v, o, v, v], t_ia, t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute((1 / 2) * np.einsum('klic,jc,klab->ijab', g[o, o, o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute(-np.einsum('klic,kc,jlab->ijab', g[o, o, o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute(-(1 / 2) * np.einsum('akcd,kb,ijcd->ijab', g[v, o, v, v], t_ia, t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute(permute(np.einsum('akcd,ic,jkbd->ijab', g[v, o, v, v], t_ia, t_ijab, optimize=True), 0, 1), 3, 2)
    t_ijab_temporary += permute(permute(-np.einsum('klic,ka,jlbc->ijab', g[o, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 3, 2)
    t_ijab_temporary += permute(np.einsum('kc,ka,ijbc->ijab', F[o, v], t_ia, t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute(np.einsum('kc,ic,jkab->ijab', F[o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute(np.einsum('klic,ka,lb,jc->ijab', g[o, o, o, v], t_ia, t_ia, t_ia, optimize=True), 1, 0)
    t_ijab_temporary += permute(-np.einsum('akcd,kb,ic,jd->ijab', g[v, o, v, v], t_ia, t_ia, t_ia, optimize=True), 3, 2)

    # Contributions from disconnected quadruples
    t_ijab_temporary += np.einsum('klcd,kc,ijlabd->ijab', g[o, o, v, v], t_ia, t_ijkabc, optimize=True)
    t_ijab_temporary += permute((1 / 2) * np.einsum('klcd,ic,jklabd->ijab', g[o, o, v, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijab_temporary += permute((1 / 2) * np.einsum('klcd,ka,ijlbcd->ijab', g[o, o, v, v], t_ia, t_ijkabc, optimize=True), 3, 2)
    t_ijab_temporary += (1 / 4) * np.einsum('klcd,klab,ijcd->ijab', g[o, o, v, v], t_ijab, t_ijab, optimize=True)
    t_ijab_temporary += permute(np.einsum('klcd,ikac,jlbd->ijab', g[o, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute((1 / 2) * np.einsum('klcd,ilab,jkcd->ijab', g[o, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute(-(1 / 2) * np.einsum('klcd,klac,ijbd->ijab', g[o, o, v, v], t_ijab, t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute(np.einsum('klcd,la,kc,ijbd->ijab', g[o, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 3, 2)
    t_ijab_temporary += permute(np.einsum('klcd,kc,id,jlab->ijab', g[o, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijab_temporary += permute(permute(-np.einsum('klcd,ka,ic,jlbd->ijab', g[o, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 0, 1), 3, 2)
    t_ijab_temporary += (1 / 2) * np.einsum('klcd,ka,lb,ijcd->ijab', g[o, o, v, v], t_ia, t_ia, t_ijab, optimize=True)
    t_ijab_temporary += (1 / 2) * np.einsum('klcd,ic,jd,klab->ijab', g[o, o, v, v], t_ia, t_ia, t_ijab, optimize=True)
    t_ijab_temporary += np.einsum('klcd,ka,lb,ic,jd->ijab', g[o, o, v, v], t_ia, t_ia, t_ia, t_ia, optimize=True)





    # Contributions from connected doubles
    t_ijkabc_temporary = permute(np.einsum('ackd,ijbd->ijkabc', g[v, v, o, v], t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('alij,klbc->ijkabc', g[v, o, o, o], t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += -np.einsum('abkd,ijcd->ijkabc', g[v, v, o, v], t_ijab, optimize=True)
    t_ijkabc_temporary += np.einsum('clij,klab->ijkabc', g[v, o, o, o], t_ijab, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('abid,jkcd->ijkabc', g[v, v, o, v], t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('clik,jlab->ijkabc', g[v, o, o, o], t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(np.einsum('acid,jkbd->ijkabc', g[v, v, o, v], t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('alik,jlbc->ijkabc', g[v, o, o, o], t_ijab, optimize=True), 1, 0), 4, 3)
    
    # Contributions from connected triples
    t_ijkabc_temporary += permute(np.einsum('alkd,ijlbcd->ijkabc', g[v, o, o, v], t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('clid,jklabd->ijkabc', g[v, o, o, v], t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('ad,ijkbcd->ijkabc', F[v, v], t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += -np.einsum('lk,ijlabc->ijkabc', F[o, o], t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('abde,ijkcde->ijkabc', g[v, v, v, v], t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmij,klmabc->ijkabc', g[o, o, o, o], t_ijkabc, optimize=True)
    t_ijkabc_temporary += np.einsum('clkd,ijlabd->ijkabc', g[v, o, o, v], t_ijkabc, optimize=True)
    t_ijkabc_temporary += np.einsum('cd,ijkabd->ijkabc', F[v, v], t_ijkabc, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('li,jklabc->ijkabc', F[o, o], t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('acde,ijkbde->ijkabc', g[v, v, v, v], t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmik,jlmabc->ijkabc', g[o, o, o, o], t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(np.einsum('alid,jklbcd->ijkabc', g[v, o, o, v], t_ijkabc, optimize=True), 1, 0), 4, 3)

    # Contributions from disconnected triples
    t_ijkabc_temporary += -np.einsum('abde,kd,ijce->ijkabc', g[v, v, v, v], t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += -np.einsum('lmij,lc,kmab->ijkabc', g[o, o, o, o], t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(np.einsum('acde,kd,ijbe->ijkabc', g[v, v, v, v], t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('alkd,lb,ijcd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('clid,jd,klab->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('clkd,la,ijbd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('clkd,id,jlab->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(-np.einsum('alid,lc,jkbd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute( -np.einsum('alid,kd,jlbc->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmik,lc,jmab->ijkabc', g[o, o, o, o], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('abde,id,jkce->ijkabc', g[v, v, v, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('alkd,lc,ijbd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('clid,kd,jlab->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmij,la,kmbc->ijkabc', g[o, o, o, o], t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('acde,id,jkbe->ijkabc', g[v, v, v, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('alid,lb,jkcd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('alid,jd,klbc->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('alkd,id,jlbc->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('clid,la,jkbd->ijkabc', g[v, o, o, v], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmik,la,jmbc->ijkabc', g[o, o, o, o], t_ia, t_ijab, optimize=True), 1, 0), 4, 3)

    # Contributions from disconnected quadruples
    t_ijkabc_temporary += (1 / 2) * np.einsum('clde,klab,ijde->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += np.einsum('clde,kd,ijlabe->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += np.einsum('lmkd,ld,ijmabc->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += np.einsum('ld,klab,ijcd->ijkabc', F[o, v], t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += -np.einsum('ld,lc,ijkabd->ijkabc', F[o, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -np.einsum('ld,kd,ijlabc->ijkabc', F[o, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -np.einsum('clde,ld,ijkabe->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -np.einsum('lmkd,lc,ijmabd->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += permute(np.einsum('ld,ijad,klbc->ijkabc', F[o, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += -(1 / 2) * np.einsum('lmkd,lmab,ijcd->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(np.einsum('alde,kd,ijlbce->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += np.einsum('clde,id,je,klab->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(permute(-np.einsum('alde,lc,id,jkbe->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('alde,id,ke,jlbc->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmid,la,jd,kmbc->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmkd,la,id,jmbc->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-(1 / 2) * np.einsum('lmid,jkad,lmbc->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('alde,lc,kd,ijbe->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('alde,ilbd,jkce->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('clde,id,ke,jlab->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmid,la,mb,jkcd->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmid,la,jkmbcd->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('lmid,lc,jd,kmab->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmid,klad,jmbc->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('lmkd,lc,id,jmab->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute((1 / 2) * np.einsum('alde,ilbc,jkde->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('clde,id,jklabe->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('clde,ikad,jlbe->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmid,ld,jkmabc->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmid,jlab,kmcd->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmkd,ilad,jmbc->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('alde,lc,ijkbde->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('alde,klbc,ijde->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('clde,ilab,jkde->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmid,jd,klmabc->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmkd,id,jlmabc->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('ld,la,ijkbcd->ijkabc', F[o, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('ld,id,jklabc->ijkabc', F[o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('ld,ikad,jlbc->ijkabc', F[o, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('alde,ld,ijkbce->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('alde,ijbd,klce->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('alde,klbd,ijce->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('clde,ijad,klbe->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('clde,ilad,jkbe->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmid,lc,jkmabd->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmid,klab,jmcd->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += -np.einsum('lmkd,la,mb,ijcd->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('lmkd,la,ijmbcd->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('alde,lb,ijkcde->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('clde,la,ijkbde->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmid,kd,jlmabc->ijkabc', g[o, o, o, v], t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmid,lmab,jkcd->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1 , 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmkd,ijad,lmbc->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('ld,ilab,jkcd->ijkabc', F[o, v], t_ijab, t_ijab, optimize=True), 1, 0), 5, 4)
    t_ijkabc_temporary += permute(np.einsum('alde,lb,kd,ijce->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('alde,id,je,klbc->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('alde,id,jklbce->ijkabc', g[v, o, v, v], t_ia, t_ijkabc, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('alde,ikbd,jlce->ijkabc', g[v, o, v, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('clde,la,kd,ijbe->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmid,lc,kd,jmab->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(np.einsum('lmid,jlad,kmbc->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmkd,la,mc,ijbd->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmkd,ilab,jmcd->ijkabc', g[o, o, o, v], t_ijab, t_ijab, optimize=True), 1, 0), 5, 4)

    # Contributions from disconnected quintuples
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,klab,ijmcde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,ijcd,klmabe->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,lmcd,ijkabe->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,klde,ijmabc->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += np.einsum('lmde,klcd,ijmabe->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 4) * np.einsum('lmde,lmab,ijkcde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 4) * np.einsum('lmde,ijde,klmabc->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('lmde,la,mb,id,jkce->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmde,la,id,je,kmbc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmde,la,id,jkmbce->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary +=permute(permute(-np.einsum('lmde,la,ikbd,jmce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-np.einsum('lmde,id,klae,jmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-(1 / 2) * np.einsum('lmde,la,imbc,jkde->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-(1 / 2) * np.einsum('lmde,id,jkae,lmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,la,mc,id,jkbe->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,la,id,ke,jmbc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,la,mc,kd,ijbe->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,la,imbd,jkce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,lc,id,ke,jmab->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,id,jlae,kmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,kd,ilab,jmce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 5, 4)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,ld,imab,jkce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0), 5, 4)
    t_ijkabc_temporary += permute(permute(np.einsum('alde,lb,id,jkce->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('clde,la,id,jkbe->ijkabc', g[v, o, v, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmid,la,mc,jkbd->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(np.einsum('lmid,la,kd,jmbc->ijkabc', g[o, o, o, v], t_ia, t_ia, t_ijab, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,la,mc,ijkbde->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,la,kmbc,ijde->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,lc,imab,jkde->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,id,ke,jlmabc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,id,lmab,jkce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,kd,ijae,lmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(permute(-(1 / 2) * np.einsum('lmde,ilac,jkmbde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += permute(permute(-(1 / 2) * np.einsum('lmde,ikad,jlmbce->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0), 4, 3)
    t_ijkabc_temporary += -np.einsum('lmde,la,mb,kd,ijce->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('lmde,la,kd,ijmbce->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('lmde,ma,ld,ijkbce->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += -np.einsum('lmde,lc,id,je,kmab->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ia, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(-np.einsum('lmde,lc,id,jkmabe->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmde,lc,ikad,jmbe->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmde,id,klab,jmce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-np.einsum('lmde,ld,ie,jkmabc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute( -np.einsum('lmde,ld,kmac,ijbe->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-np.einsum('lmde,ld,ikae,jmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += np.einsum('lmde,ld,kmab,ijce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(np.einsum('lmde,klad,ijmbce->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,ilcd,jkmabe->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,la,mb,ijkcde->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += (1 / 2) * np.einsum('lmde,id,je,klmabc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmde,ilab,jkmcde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmde,ijad,klmbce->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmde,lmad,ijkbce->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute((1 / 2) * np.einsum('lmde,ilde,jkmabc->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += -np.einsum('lmde,lc,kd,ijmabe->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -np.einsum('lmde,mc,ld,ijkabe->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -np.einsum('lmde,ld,ke,ijmabc->ijkabc', g[o, o, v, v], t_ia, t_ia, t_ijkabc, optimize=True)
    t_ijkabc_temporary += -(1 / 2) * np.einsum('lmde,lc,kmab,ijde->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += -(1 / 2) * np.einsum('lmde,kd,lmab,ijce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,klac,ijmbde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 2) * np.einsum('lmde,ikcd,jlmabe->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(-(1 / 4) * np.einsum('lmde,lmac,ijkbde->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(-(1 / 4) * np.einsum('lmde,ikde,jlmabc->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmde,la,ijbd,kmce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,la,kmbd,ijce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,lc,imad,jkbe->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmde,lc,kmad,ijbe->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 4, 3)
    t_ijkabc_temporary += permute(np.einsum('lmde,id,jlab,kmce->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(np.einsum('lmde,kd,ilae,jmbc->ijkabc', g[o, o, v, v], t_ia, t_ijab, t_ijab, optimize=True), 1, 0)
    t_ijkabc_temporary += permute(permute(np.einsum('lmde,ilad,jkmbce->ijkabc', g[o, o, v, v], t_ijab, t_ijkabc, optimize=True), 1, 0), 4, 3)


    # Updates t-amplitudes with epsilons tensors

    t_ia += e_ia * t_ia_temporary 
    t_ijab += e_ijab * t_ijab_temporary 
    t_ijkabc += e_ijkabc * t_ijkabc_temporary 

    t_amplitudes = t_ia, t_ijab, t_ijkabc, None

    return t_amplitudes










def run_restricted_CCSDTQ_iteration(o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, molecular_orbitals: ndarray, integrals: Integrals) -> tuple:
  
    """
    
    Updates the amplitudes for restricted CCSDTQ.

    Implementation of 10.26434/chemrxiv-2025-qgc1q (CCSDTQ).

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (tuple): Amplitudes
        e_denominators (tuple):Epsilon denominators
        molecular_orbitals (array): Molecular orbitals
        integrals (Integrals): Molecular integrals

    Returns:
        t_amplitudes (tuple): Updated amplitudes
    
    """

    t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes
    e_ia, e_ijab, e_ijkabc, e_ijklabcd = e_denominators

    # Updates the amplitudes to include CCSD and CCSDT terms

    (t_ia_ccsdt, t_ijab_ccsdt, t_ijkabc_ccsdt, _), g_hat, F_hat, u_ijab = run_restricted_CCSDT_iteration(o, v, (t_ia.copy(), t_ijab.copy(), t_ijkabc.copy(), None), e_denominators, molecular_orbitals, integrals)

    # Unpacks the residuals from the CCSDT update

    residual_ia = (t_ia_ccsdt - t_ia) / e_ia
    residual_ijab = (t_ijab_ccsdt - t_ijab) / e_ijab
    residual_ijkabc = (t_ijkabc_ccsdt - t_ijkabc) / e_ijkabc

    # Builds intermediates for CCSDTQ - direct implementation from 10.26434/chemrxiv-2025-qgc1q

    alpha = 2 * t_ijklabcd - t_ijklabcd.swapaxes(4, 5) - t_ijklabcd.swapaxes(4, 6) - t_ijklabcd.transpose(0, 1, 2, 3, 7, 5, 6, 4)
    beta = 2 * alpha - alpha.swapaxes(5, 6) - alpha.swapaxes(5, 7)

    z_ijkabc = 2 * t_ijkabc - t_ijkabc.swapaxes(3, 4) - t_ijkabc.swapaxes(3, 5)

    A_aebj = g_hat[v, v, v, o] + np.einsum("menj,mnab->aebj", g_hat[o, v, o, o], t_ijab, optimize=True)
    A_aebj += (1 / 2) * (np.einsum("mfae,mjfb->aebj", 2 * g_hat[o, v, v, v], u_ijab, optimize=True) - np.einsum("afme,mjfb->aebj", g_hat[v, v, o, v], u_ijab, optimize=True))
    A_aebj += -(1 / 2) * np.einsum("meaf,jmfb->aebj", g_hat[o, v, v, v], t_ijab, optimize=True) - np.einsum("meaf,jmfb->aebj", g_hat[o, v, v, v], t_ijab, optimize=True).swapaxes(0, 2)
    A_aebj += -1 * np.einsum("menf,nmjfab->aebj", g_hat[o, v, o, v], z_ijkabc, optimize=True)
    A_aebj += -1 * np.einsum("me,mjab->aebj", F_hat[o, v], t_ijab, optimize=True)

    B_aimj = g_hat[v, o, o, o] + np.einsum("aemf,ijef->aimj", g_hat[v, v, o, v], t_ijab, optimize=True)
    B_aimj += (1 / 2) * (np.einsum("nemj,niea->aimj", 2 * g_hat[o, v, o, o], u_ijab, optimize=True) - np.einsum("njme,niea->aimj", g_hat[o, o, o, v], u_ijab, optimize=True))
    B_aimj += -(1 / 2) * np.einsum("njme,inea->aimj", g_hat[o, o, o, v], t_ijab, optimize=True) - np.einsum("njme,inea->aimj", g_hat[o, o, o, v], t_ijab, optimize=True).swapaxes(1, 3)
    B_aimj += np.einsum("me,ijae->aimj", F_hat[o, v], t_ijab, optimize=True)
    B_aimj += np.einsum("menf,nijfae->aimj", g_hat[o, v, o, v], z_ijkabc, optimize=True)

    F_tilde_tilde_ae = F_hat[v, v] - np.einsum("nfme,nmfa->ae", 2 * g_hat[o, v, o, v], t_ijab, optimize=True) + np.einsum("nemf,nmfa->ae", g_hat[o, v, o, v], t_ijab, optimize=True)
    F_tilde_tilde_mi = F_hat[o, o] + np.einsum("nfme,nife->mi", 2 * g_hat[o, v, o, v], t_ijab, optimize=True) - np.einsum("nemf,nife->mi", g_hat[o, v, o, v], t_ijab, optimize=True)

    E_meai = 2 * g_hat[o, v, v, o] - g_hat[o, o, v, v].swapaxes(1, 3)
    E_meai += np.einsum("nfme,nifa->meai", 2 * g_hat[o, v, o, v], u_ijab, optimize=True) - np.einsum("nemf,nifa->meai", g_hat[o, v, o, v], u_ijab, optimize=True)

    F_miae = g_hat[o, o, v, v] - np.einsum("nemf,infa->miae", g_hat[o, v, o, v], t_ijab, optimize=True)
    G_minj = g_hat[o, o, o, o] + np.einsum("menf,ijef->minj", g_hat[o, v, o, v], t_ijab, optimize=True)
    H_aebf = g_hat[v, v, v, v] + np.einsum("menf,mnab->aebf", g_hat[o, v, o, v], t_ijab, optimize=True)

    I_ejimba = 2 * np.einsum("meaf,jibf->ejimba", g_hat[o, v, v, v], t_ijab, optimize=True)
    I_ejimba += -1 * np.einsum("mfae,jibf->ejimba", g_hat[o, v, v, v], t_ijab, optimize=True)
    I_ejimba += -2 * np.einsum("meni,njab->ejimba", g_hat[o, v, o, o], t_ijab, optimize=True)
    I_ejimba += np.einsum("mine,njab->ejimba", g_hat[o, o, o, v], t_ijab, optimize=True)
    I_ejimba += (1 / 2) * np.einsum("nfme,nijfab->ejimba", g_hat[o, v, o, v], z_ijkabc, optimize=True)
    I_ejimba += -1 *(1 / 4) * np.einsum("nemf,nijfab->ejimba", g_hat[o, v, o, v], z_ijkabc, optimize=True)
    I_eijmab = I_ejimba + I_ejimba.swapaxes(1, 2).swapaxes(4, 5)

    J_iejmab = np.einsum("mfae,jibf->iejmab", g_hat[o, v, v, v], t_ijab, optimize=True)
    J_iejmab += -1* np.einsum("mine,njab->iejmab", g_hat[o, o, o, v], t_ijab, optimize=True)
    J_iejmab += -1 * (1 / 2) * np.einsum("nemf,injfab->iejmab", g_hat[o, v, o, v], t_ijkabc, optimize=True)

    K_ikjanm = np.einsum("menk,ijae->ikjanm", g_hat[o, v, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum("menf,ijkaef->ikjanm", g_hat[o, v, o, v], t_ijkabc, optimize=True)
    K_ijkamn = K_ikjanm + K_ikjanm.swapaxes(1, 2).swapaxes(4, 5)

    L_jikbam = np.einsum("aemf,ijkebf->jikbam", g_hat[v, v, o, v], t_ijkabc, optimize=True)
    L_jikbam += (1 / 2) * np.einsum("meai,jkbe->jikbam", E_meai, t_ijab, optimize=True)
    L_jikbam += (1 / 2) * np.einsum("miae,jkbe->jikbam", F_miae, t_ijab, optimize=True)
    L_jikbam += np.einsum("mkae,jibe->jikbam", F_miae, t_ijab, optimize=True)
    L_jikbam += -1 * (1 / 2) * np.einsum("mkni,njab->jikbam", G_minj, t_ijab, optimize=True)
    L_jikbam += (1 / 2) * np.einsum("menf,nijkfabe->jikbam", g_hat[o, v, o, v], alpha, optimize=True)
    L_ijkabm = L_jikbam + L_jikbam.swapaxes(0, 1).swapaxes(3, 4)

    M_ekjacb = (1 / 2) * np.einsum("aebf,jkfc->ekjacb", H_aebf, t_ijab, optimize=True) - (1 / 2) * np.einsum("menf,nmjkfabc->ekjacb", g_hat[o, v, o, v], alpha, optimize=True)
    M_ejkabc = M_ekjacb + M_ekjacb.swapaxes(1, 2).swapaxes(4, 5)
    
    # The quadruple amplitudes don't mix with the singles - first to be updated are the doubles

    residual_ijab += mp.permute_symmetric((1 / 4) * np.einsum("menf,mnijefab->ijab", g_hat[o, v, o, v], beta, optimize=True), (0, 1), (2, 3))
    
    # Triples residuals updated with quadruples influence

    residual_ijkabc += mp.permute_three_column_indices((1 / 6) * np.einsum("me,mijkeabc->ijkabc", F_hat[o, v], alpha, optimize=True) + (1 / 2) * np.einsum("aemf,mijkfebc->ijkabc", g_hat[v, v, o, v], alpha, optimize=True) - (1 / 2) * np.einsum("menj,minkeabc->ijkabc", g_hat[o, v, o, o], alpha, optimize=True))

    # Builds complicated quadruples residual

    residual_ijklabcd = (1 / 2) * np.einsum("aebj,iklecd->ijklabcd", A_aebj, t_ijkabc, optimize=True) 
    residual_ijklabcd += - (1 / 2) * np.einsum("aimj,mklbcd->ijklabcd", B_aimj, t_ijkabc, optimize=True)
    residual_ijklabcd += (1 / 6) * np.einsum("ae,ijklebcd->ijklabcd", F_tilde_tilde_ae, t_ijklabcd, optimize=True)
    residual_ijklabcd += -1 * (1 / 6) * np.einsum("mi,mjklabcd->ijklabcd", F_tilde_tilde_mi, t_ijklabcd, optimize=True)
    residual_ijklabcd += (1 / 12) * np.einsum("meai,mjklebcd->ijklabcd", E_meai, alpha, optimize=True)
    residual_ijklabcd += -1 * (1 / 4) * np.einsum("miae,jmklebcd->ijklabcd", F_miae, t_ijklabcd, optimize=True) 
    residual_ijklabcd += - (1 / 2) * np.einsum("miae,jmklebcd->ijklabcd", F_miae, t_ijklabcd, optimize=True).swapaxes(4, 5)
    residual_ijklabcd += (1 / 4) * np.einsum("minj,mnklabcd->ijklabcd", G_minj, t_ijklabcd, optimize=True)
    residual_ijklabcd += (1 / 4) * np.einsum("aebf,ijklefcd->ijklabcd", H_aebf, t_ijklabcd, optimize=True)
    residual_ijklabcd += (1 / 8) * np.einsum("eijmab,mklecd->ijklabcd", I_eijmab, z_ijkabc, optimize=True)
    residual_ijklabcd += -1 * (1 / 2) * np.einsum("iejmab,kmlecd->ijklabcd", J_iejmab, t_ijkabc, optimize=True) 
    residual_ijklabcd += -1 * np.einsum("iejmab,kmlecd->ijklabcd", J_iejmab, t_ijkabc, optimize=True).swapaxes(4, 6)
    residual_ijklabcd += (1 / 2) * np.einsum("ijkamn,mnlbcd->ijklabcd", K_ijkamn, t_ijkabc, optimize=True)
    residual_ijklabcd += -1 * (1 / 2) * np.einsum("ijkabm,mlcd->ijklabcd", L_ijkabm, t_ijab, optimize=True)
    residual_ijklabcd += (1 / 2) * np.einsum("ejkabc,iled->ijklabcd", M_ejkabc, t_ijab, optimize=True)

    # Applies final 24-fold permutation to the quadruples residual

    residual_ijklabcd = mp.permute_four_column_indices(residual_ijklabcd)

    # Updates amplitudes

    t_ia += e_ia * residual_ia
    t_ijab += e_ijab * residual_ijab
    t_ijkabc += e_ijkabc * residual_ijkabc
    t_ijklabcd += e_ijklabcd * residual_ijklabcd

    t_amplitudes = t_ia, t_ijab, t_ijkabc, t_ijklabcd

    return t_amplitudes










def calculate_restricted_CCSD_T_energy(g: ndarray, e_ijkabc: ndarray, t_ia: ndarray, t_ijab: ndarray, o: slice, v: slice, method: Method, calculation: Calculation, silent: bool) -> float:


    """ 
    
    Calculates the perturbative triples energy for restricted CCSD(T).

    Args:
        g (array): Non-antisymmetrised ERI in spatial MO basis
        e_ijkabc (array): Triples epsilon tensor
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        E_CCSD_T (float): Restricted CCSD(T) energy

    """

    method.name = method.name.replace("[", "(").replace("]", ")")

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                    {method.name} Energy ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)


    def P_ijkabc(array):

        # Three index permutation per Lee

        return array + array.transpose(1, 0, 2, 4, 3, 5) + array.transpose(2, 1, 0, 5, 4, 3) + array.transpose(0, 2, 1, 3, 5, 4) + array.transpose(2, 0, 1, 5, 3, 4) + array.transpose(1, 2, 0, 4, 5, 3)


    log("  Forming disconnected amplitudes...         ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Calculation of key intermediate tensors

    V_ijkabc = np.einsum("jkbc,ia->ijkabc", g[o, o, v, v], t_ia, optimize=True)
    V_ijkabc += np.einsum("ikac,jb->ijkabc", g[o, o, v, v], t_ia, optimize=True)
    V_ijkabc += np.einsum("ijab,kc->ijkabc", g[o, o, v, v], t_ia, optimize=True)

    space = " "

    if "QCISD" in method.name: 
        
        # This factor of two arises because part of the MP5 disconnected triples are included in the CCSD equations, but not the QCISD equations

        V_ijkabc *= 2
        space = ""

    log(f"[Done]", calculation, 1, silent=silent)

    log("  Forming connected amplitudes...            ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    W_ijkabc = P_ijkabc(np.einsum("ibaf,kjcf->ijkabc", g[o, v, v, v], t_ijab, optimize=True) - np.einsum("ijam,mkbc->ijkabc", g[o, o, v, o], t_ijab, optimize=True))

    W = 4 * W_ijkabc + W_ijkabc.transpose(2, 0, 1, 3, 4, 5) + W_ijkabc.transpose(1, 2, 0, 3, 4, 5) - 4 * W_ijkabc.transpose(2, 1, 0, 3, 4, 5) - W_ijkabc.transpose(0, 2, 1, 3, 4, 5) - W_ijkabc.transpose(1, 0, 2, 3, 4, 5)
    
    log(f"[Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating {method.name} correlation energy... {space}", calculation, 1, end="", silent=silent); sys.stdout.flush()

    E_CCSD_T = (1 / 3) * np.einsum("ijkabc,ijkabc,ijkabc->", W_ijkabc + V_ijkabc, W, e_ijkabc, optimize=True)
    
    log(f"[Done]\n\n  {method.name} correlation energy:       {space} {E_CCSD_T:13.10f}", calculation, 1, silent=silent) 

    return E_CCSD_T










def calculate_unrestricted_CCSD_T_energy(g: ndarray, e_ijkabc: ndarray, t_ia: ndarray, t_ijab: ndarray, o: slice, v: slice, method: Method, calculation: Calculation, silent: bool) -> float:

    """ 
    
    Calculates the perturbative triples energy for CCSD(T).

    Args:
        g (array): Spin orbital ERI tensor
        e_ijkabc (array): Triples epsilon tensor
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        E_CCSD_T (float): CCSD(T) energy

    """

    method.name = method.name.replace("[", "(").replace("]", ")")

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                   {method.name} Energy  ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    def permute_three_indices(array_ijab, idx1, idx2, idx3):
        
        # Three-index permutation operator per Crawford

        return array_ijab - array_ijab.swapaxes(idx1, idx2) - array_ijab.swapaxes(idx1, idx3)

    log("  Forming disconnected amplitudes...         ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    # Temporary disconnected (d_ijkabc) and connected (c_ijkabc) triples tensors before permutation, from Crawford

    d_ijkabc = np.einsum("ia,jkbc->ijkabc", t_ia, g[o, o, v, v], optimize=True)
    t_ijkabc_d = np.einsum("ijkabc,ijkabc->ijkabc", e_ijkabc, permute_three_indices(permute_three_indices(d_ijkabc, 3, 4, 5), 0, 1, 2), optimize=True)
    
    space = " "

    # This factor of two arises because part of the MP5 disconnected triples are included in the CCSD equations, but not the QCISD equations

    if "QCISD" in method.name: 
        
        t_ijkabc_d *= 2
        space = ""

    log(f"[Done]", calculation, 1, silent=silent)

    log("  Forming connected amplitudes...            ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    c_ijkabc = np.einsum("jkae,eibc->ijkabc", t_ijab, g[v, o, v, v], optimize=True) - np.einsum("imbc,majk->ijkabc", t_ijab, g[o, v, o, o], optimize=True)
    t_ijkabc_c = np.einsum("ijkabc,ijkabc->ijkabc", e_ijkabc, permute_three_indices(permute_three_indices(c_ijkabc, 3, 4, 5), 0, 1, 2), optimize=True)

    log(f"[Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating {method.name} correlation energy... {space}", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    # Final contraction for the CCSD(T) energy using the connected and disconnected approximate triples amplitudes

    E_CCSD_T = (1 / 36) * np.einsum("ijkabc,ijkabc->", t_ijkabc_c / e_ijkabc, t_ijkabc_c + t_ijkabc_d, optimize=True)

    log(f"[Done]\n\n  {method.name} correlation energy:       {space} {E_CCSD_T:13.10f}", calculation, 1, silent=silent) 
    

    return E_CCSD_T










def calculate_restricted_CCSDT_Q_energy(g: ndarray, e_ijklabcd: ndarray, t_ijab: ndarray, t_ijkabc: ndarray, o: slice, v: slice, calculation: Calculation, silent: bool) -> float:

    """ 
    
    Calculates the perturbative quadruples energy for CCSDT(Q).

    Args:
        g (array): Spatial orbital ERI tensor
        e_ijklabcd (array): Quadruples epsilon tensor
        t_ijab (array): Converged doubles amplitudes
        t_ijkabc (array): Converged triples amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        E_CCSDT_Q (float): CCSDT(Q) energy

    """

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                   CCSDT(Q) Energy ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log("  Forming quadruples amplitudes...           ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Now shape <pr|qs> -> (pq|rs) in chemist's notation

    g = g.swapaxes(1, 2)

    u_ijab = 2 * t_ijab - t_ijab.swapaxes(2, 3)

    K_ijab = g[o, o, v, v]  # (ij|ab)
    L_ijab = 2 * K_ijab - K_ijab.swapaxes(2, 3)

    G = np.einsum("iabe,jklecd->ijklabcd", g[o, v, v, v], t_ijkabc, optimize=True) 
    G += -1 * np.einsum("iamj,mklbcd->ijklabcd", g[o, v, o, o], t_ijkabc, optimize=True)
    G += np.einsum("minj,mkac,nlbd->ijklabcd", g[o, o, o, o], t_ijab, t_ijab, optimize=True) 
    G += - 2 * np.einsum("iame,kjeb,mlcd->ijklabcd", g[o, v, o, v], t_ijab, t_ijab, optimize=True)
    G += np.einsum("cfae,ijeb,klfd->ijklabcd", g[v, v, v, v], t_ijab, t_ijab, optimize=True) 
    G += - 2 * np.einsum("bemi,kjce,mlad->ijklabcd", g[v, v, o, o], t_ijab, t_ijab, optimize=True)

    G = (1 / 2) * mp.permute_four_column_indices(G)

    t_ijklabcd = G * e_ijklabcd

    log(f"[Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating MP5 contribution to energy...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Determines the MP5 contribution

    E_CCSDT_Q_MP5 = np.einsum("ijklcdab,klcd,ijab->", t_ijklabcd, u_ijab, K_ijab, optimize=True)
    E_CCSDT_Q_MP5 += -2 * np.einsum("ijklbdac,kldc,ijba->", t_ijklabcd, u_ijab, L_ijab, optimize=True)
    E_CCSDT_Q_MP5 += np.einsum("ijklabcd,klcd,ijab->", t_ijklabcd, u_ijab, L_ijab, optimize=True)

    log(f"[Done]", calculation, 1, silent=silent) 

    log(f"  Calculating MP6 contribution to energy...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    t_bar_ijklabcd = -2 * t_ijklabcd - t_ijklabcd.swapaxes(4, 6).swapaxes(5, 7) + t_ijklabcd.swapaxes(4, 5)
    t_tilde_ijklabcd = 2 * t_ijklabcd.transpose(0, 1, 2, 3, 7, 5, 4, 6) - t_ijklabcd.transpose(0, 1, 2, 3, 5, 7, 4, 6)
    t_tilde_ijklabcd += t_tilde_ijklabcd.swapaxes(2, 3).swapaxes(6, 7)

    term = np.einsum("mjicba,ldkm->ijklabcd", t_ijkabc, g[o, v, o, o], optimize=True)
    term2 = np.einsum("kjieba,ldce->ijklabcd", t_ijkabc, g[o, v, v, v], optimize=True)

    alpha = 2 * term - term.swapaxes(6, 7) - 2 * term2 + term2.swapaxes(2, 3)
    
    term = np.einsum("mjicba,kdlm->ijklabcd", t_ijkabc, g[o, v, o, o], optimize=True)
    term2 = np.einsum("ljieba,kdce->ijklabcd", t_ijkabc, g[o, v, v, v], optimize=True)
 
    beta = 2 * term - term.swapaxes(6, 7) - 2 * term2 + term2.swapaxes(2, 3)

    # Determines the MP6 contribution

    E_CCSDT_Q_MP6 = 2 * np.einsum("ijklabcd,ijklabcd->", alpha, t_bar_ijklabcd, optimize=True)
    E_CCSDT_Q_MP6 += 2 * np.einsum("ijklabcd,ijklabcd->", beta, t_tilde_ijklabcd, optimize=True)

    E_CCSDT_Q = E_CCSDT_Q_MP5 + E_CCSDT_Q_MP6

    log(f"[Done]", calculation, 1, silent=silent) 

    log(f"\n  Contribution from MP5:              {E_CCSDT_Q_MP5:13.10f}", calculation, 2, silent=silent) 
    log(f"  Contribution from MP6:              {E_CCSDT_Q_MP6:13.10f}", calculation, 2, silent=silent) 

    log(f"\n  CCSDT(Q) correlation energy:        {E_CCSDT_Q:13.10f}", calculation, 1, silent=silent) 
    

    return E_CCSDT_Q










def calculate_coupled_cluster_energy(g: ndarray, o: slice, v: slice, t_amplitudes: tuple, e_denominators: tuple, F: ndarray, method: Method, calculation: Calculation, silent: bool, SCF_output: Output, integrals: Integrals) -> tuple:

    """
    
    Calculates the coupled cluster energy in an iterative procedure.

    Args:
        g (array): Two-electron integrals in spin or spatial orbital basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_amplitudes (array): Guess  amplitudes
        e_denominators (array): Epsilons tensors
        F (array): Fock matrix in spin or spatial orbital basis
        method (Method): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging
        SCF_output (Output): Output object
        integrals (Integrals): Molecular integrals

    Returns:
        E_CC (float): Coupled cluster energy
        t_amplitudes (tuple): Converged amplitudes
    
    """

    E_CC = 0.0

    calculate_iterative_singles = not "CCD" in method.name
    calculate_iterative_triples = "CCSDT" in method.name
    calculate_iterative_quadruples = "CCSDTQ" in method.name

    # Chops of "[T]" or "[Q]" in the method name 
    
    original_method_name = method.name

    method.name = method.name.split("[T]")[0] if "[T]" in method.name else method.name
    method.name = method.name.split("[Q]")[0] if "[Q]" in method.name else method.name

    # Sets up DIIS vectors

    DIIS_t_vector, DIIS_error_vector = ([], [], [], []), []

    t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes

    # Common printing for all coupled cluster calculations

    coupled_cluster_initial_print(g, o, v, t_amplitudes, calculation.reference, method, calculation, silent)

    if calculation.reference == "RHF":
        
        # Useful intermediate quantity for restricted coupled cluster

        w = 2 * g - g.swapaxes(0, 1)

    for step in range(1, calculation.correlated_max_iter + 1):

        E_old = E_CC

        # Easier just to set singles amplitudes to zero than keep checking if it's present

        t_ia_old = t_ia.copy() if calculate_iterative_singles else np.zeros_like(e_denominators[0]) 
        t_ijab_old = t_ijab.copy()
        t_ijkabc_old = t_ijkabc.copy() if calculate_iterative_triples else None
        t_ijklabcd_old = t_ijklabcd.copy() if calculate_iterative_quadruples else None

        t_amplitudes = t_ia, t_ijab, t_ijkabc, t_ijklabcd
        t_amplitudes_old = t_ia_old, t_ijab_old, t_ijkabc_old, t_ijklabcd_old

        if calculation.reference == "RHF":

            match method.name:

                case "LCCD":

                    t_amplitudes = run_restricted_LCCD_iteration(g, o, v, t_amplitudes, e_denominators)

                case "CCD":

                    t_amplitudes = run_restricted_CCD_iteration(g, o, v, t_amplitudes, e_denominators, w)

                case "LCCSD":

                    t_amplitudes = run_restricted_LCCSD_iteration(g, o, v, t_amplitudes, e_denominators, w)

                case "CC2":

                    t_amplitudes = run_restricted_CC2_iteration(o, v, t_amplitudes, e_denominators, SCF_output.molecular_orbitals, integrals)

                case "CC3":

                    t_amplitudes = run_restricted_CC3_iteration(o, v, t_amplitudes, e_denominators, SCF_output.molecular_orbitals, integrals)

                case "CCSD":

                    t_amplitudes = run_restricted_CCSD_iteration(g, o, v, t_amplitudes, e_denominators, w, F)
                    
                case "QCISD":

                    t_amplitudes = run_restricted_QCISD_iteration(g, o, v, t_amplitudes, e_denominators, w)

                case "CCSDT":

                    t_amplitudes, _, _, _ = run_restricted_CCSDT_iteration(o, v, t_amplitudes, e_denominators, SCF_output.molecular_orbitals, integrals)
            
                case "CCSDTQ":

                    t_amplitudes = run_restricted_CCSDTQ_iteration(o, v, t_amplitudes, e_denominators, SCF_output.molecular_orbitals, integrals)
            

            # Use the energy expression from restricted coupled cluster

            E_CC, E_CC_singles, E_CC_connected_doubles, E_CC_disconnected_doubles = calculate_restricted_coupled_cluster_energy(o, v, w, t_amplitudes, method, F)
            

        elif calculation.reference == "UHF":

            match method.name:

                case "LCCD":

                    t_amplitudes = run_unrestricted_LCCD_iteration(g, o, v, t_amplitudes, e_denominators)

                case "CCD":

                    t_amplitudes = run_unrestricted_CCD_iteration(g, o, v, t_amplitudes, e_denominators)

                case "LCCSD":

                    t_amplitudes = run_unrestricted_LCCSD_iteration(g, o, v, t_amplitudes, e_denominators, F)

                case "CCSD":

                    t_amplitudes = run_unrestricted_CCSD_iteration(g, o, v, t_amplitudes, e_denominators, F)

                case "QCISD":

                    t_amplitudes = run_unrestricted_QCISD_iteration(g, o, v, t_amplitudes, e_denominators, F)

                case "CCSDT":

                    t_amplitudes = run_unrestricted_CCSDT_iteration(g, o, v, t_amplitudes, e_denominators, F)
                

            # Use the energy expression from unrestricted coupled cluster

            E_CC, E_CC_singles, E_CC_connected_doubles, E_CC_disconnected_doubles = calculate_unrestricted_coupled_cluster_energy(o, v, g, t_amplitudes, method, F)


        t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes

        # Makes sure all amplitudes are finite

        if E_CC > 1000 or any(not np.isfinite(amplitude).all() for amplitude in t_amplitudes if amplitude is not None):

            error(f"Non-finite encountered in {method.name} iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy

        delta_E = E_CC - E_old

        # Prints the correlation energy at the current iteration

        log(f"  {step:3.0f}           {E_CC:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop

        if is_coupled_cluster_converged(delta_E, t_amplitudes, t_amplitudes_old, calculation): 
            
            break

        if step >= calculation.correlated_max_iter: 
            
            error(f"The {method.name} iterations failed to converge! Try increasing the maximum iterations with CCMAXITER?")

        # Update amplitudes with DIIS                   

        t_amplitudes, DIIS_t_vector, DIIS_error_vector = apply_DIIS(t_amplitudes, t_amplitudes_old, DIIS_t_vector, DIIS_error_vector, step, calculation, silent)

        # Damps the t-amplitudes if this is requested

        t_amplitudes = apply_damping(calculation.correlated_damping_parameter, t_amplitudes, t_amplitudes_old)

        # Need to unpack again within the loop

        t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes


    log_spacer(calculation, silent=silent)

    log(f"\n  Singles contribution:               {E_CC_singles:13.10f}", calculation, 1, silent=silent)
    log(f"  Connected doubles contribution:     {E_CC_connected_doubles:13.10f}", calculation, 1, silent=silent)
    log(f"  Disconnected doubles contribution:  {E_CC_disconnected_doubles:13.10f}", calculation, 1, silent=silent)

    log(f"\n  {method.name} correlation energy:  {" " * (10 - len(method.name))}    {E_CC:.10f}", calculation, 1, silent=silent)

    method.name = original_method_name


    return E_CC, t_amplitudes










def begin_coupled_cluster_calculation(method: Method, molecule: Molecule, SCF_output: Output, integrals: Integrals, X: ndarray, calculation: Calculation, silent: bool) -> tuple:

    """
    
    Sets off a coupled cluster calculation.

    Depending on whether a RHF or UHF reference is present, calculates the relevant intermediates and sets off the iterative calculation, then runs
    the post-processing including the T1 diagnostic, linaerised density and optional natural orbitals and occupancies.

    Args:
        method (Method): Electronic structure method
        molecule (Molecule): Molecule object
        SCF_output (Output): Output object
        integrals (Integrals): Molecular integrals in AO basis
        calculation (Calculation): Calculation object
        silent (bool): Cancel logging

    Returns:
        E_CC (float): Coupled cluster energy
        E_perturbative (float): Energy from perturbative triples or quadruples
        density_matrices (tuple): Total, alpha and beta density matrices in AO basis
        occupancies (array): Natural orbital occupancies
        natural_orbitals (array): Natural orbitals

    """

    E_CC, E_perturbative = 0, 0
    occupancies, natural_orbitals = None, None

    calculate_triples = method.name in ["CCSDT", "CCSD[T]", "QCISD[T]", "CCSDT[Q]", "CCSDTQ", "CC3"]
    calculate_quadruples = method.name in ["CCSDT[Q]", "CCSDTQ"]


    if calculation.reference == "RHF":

        n_occ = molecule.n_doubly_occ

        g, molecular_orbitals, epsilons, o, v = ci.begin_spatial_orbital_calculation(molecule, integrals.ERI_AO, SCF_output, n_occ, calculation, silent=silent)
        
        # All coupled cluster calculations use interleaved physicists' notation, (pq|rs) -> <pr|qs>

        g = g.swapaxes(1, 2)

        # Builds spatial orbital Fock matrix

        F = np.diag(epsilons)

        # Don't need to worry about these for spatial orbital calculations

        spin_labels_sorted, spin_orbital_labels_sorted = None, None


    else:
        
        n_occ = molecule.n_occ
        
        g, molecular_orbitals, epsilons, _, o, v, spin_labels_sorted, spin_orbital_labels_sorted = ci.begin_spin_orbital_calculation(molecule, integrals.ERI_AO, SCF_output, n_occ, calculation, silent=silent)
        
        # Builds spin orbital core Hamiltonian

        H_core_spin_block = ci.spin_block_core_Hamiltonian(integrals.H_core)
        H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, molecular_orbitals)

        # Combines the spin-orbital core Hamiltonian and ERIs to get the spin-orbital basis Fock matrix

        F = ci.build_spin_orbital_Fock_matrix(H_core_SO, g, slice(0, n_occ))


    log("\n Preparing arrays for coupled cluster...     ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Builds the inverse epsilon tensors - skips triples and quadruples if not required

    e_ia = ci.build_singles_epsilons_tensor(epsilons, o, v)
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
    e_ijkabc = ci.build_triples_epsilons_tensor(epsilons, o, v) if calculate_triples else np.zeros_like(e_ijab)
    e_ijklabcd = ci.build_quadruples_epsilons_tensor(epsilons, o, v) if calculate_quadruples else np.zeros_like(e_ijab)

    # Defines the guess t-amplitudes

    t_ia = np.einsum("ia,ia->ia", e_ia, F[o, v], optimize=True)
    t_ijab = ci.build_MP2_t_amplitudes(g[o, o, v, v], e_ijab)
    t_ijkabc = np.zeros_like(e_ijkabc)
    t_ijklabcd = np.zeros_like(e_ijklabcd)

    t_amplitudes = t_ia, t_ijab, t_ijkabc, t_ijklabcd
    e_denominators = e_ia, e_ijab, e_ijkabc, e_ijklabcd

    log("[Done]", calculation, 1, silent=silent)

    # Calculates the coupled cluster energy

    E_CC, t_amplitudes = calculate_coupled_cluster_energy(g, o, v, t_amplitudes, e_denominators, F, method, calculation, silent, SCF_output, integrals)

    t_ia, t_ijab, t_ijkabc, t_ijklabcd = t_amplitudes

    # Easier to do post-processing with zeroed arrays

    t_ia = np.zeros_like(e_ia) if t_ia is None else t_ia

    # Determines and prints the T1 diagnostic and norm of the singles

    calculate_T1_diagnostic(molecule, t_ia, spin_labels_sorted, n_occ, molecule.n_alpha, molecule.n_beta, calculation, silent)

    # Determines and prints the largest amplitudes

    find_and_print_largest_amplitudes(t_ia, t_ijab, n_occ, calculation, spin_orbital_labels_sorted, silent)

    # Calculates the unrelaxed density matrix in the AO basis

    density_matrices = calculate_coupled_cluster_linearised_density(t_ia, t_ijab, molecule.n_orbitals, n_occ, o, v, calculation, molecular_orbitals, silent=silent)
    
    # If "NATORBS" is used, calculate and print the natural orbitals

    if calculation.natural_orbitals: 
        
        occupancies, natural_orbitals = mp.calculate_natural_orbitals(density_matrices[0], X, calculation, silent=silent)

    if "[T]" in method.name:

        if calculation.reference == "UHF":

            E_perturbative = calculate_unrestricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent)

        else:

            E_perturbative = calculate_restricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent)


    elif "[Q]" in method.name:

        E_perturbative = calculate_restricted_CCSDT_Q_energy(g, e_ijklabcd, t_ijab, t_ijkabc, o, v, calculation, silent)


    log_spacer(calculation, silent=silent)


    return E_CC, E_perturbative, density_matrices, occupancies, natural_orbitals