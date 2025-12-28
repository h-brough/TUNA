import numpy as np
import tuna_ci as ci
import tuna_mp as mp
from tuna_util import *
import sys




def calculate_restricted_coupled_cluster_energy(o, v, w, t_ijab, method, t_ia=None, F=None):
   
    """

    Calculates the spin-restricted coupled cluster energy.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        t_ia (array, optional): Singles amplitudes
        F (array, optional): Spatial-orbital Fock matrix

    Returns:
        E_CC (float): Restricted coupled cluster energy
        E_singles (float): Restricted coupled cluster energy from single excitations
        E_connected_doubles (float): Restricted coupled cluster energy from connected double excitations
        E_disconnected_doubles (float): Restricted coupled cluster energy from disconnected double excitations

    """


    # Contribution to coupled cluster energy from single excitations (should be zero)
    E_singles = np.einsum("ia,ia->", F[o, v], t_ia, optimize=True) if t_ia is not None and F is not None else 0

    # Contribution to coupled cluster energy from connected double excitations (should be large at normal bond lengths)
    E_connected_doubles = np.einsum("abij,ijab->", w[v, v, o, o], t_ijab, optimize=True)

    # Contribution to coupled cluster energy from disconnected double excitations (should be small at normal bond lengths)
    E_disconnected_doubles = np.einsum("abij,ia,jb->", w[v, v, o, o], t_ia, t_ia, optimize=True) if t_ia is not None else 0

    # In linearised methods, the total energy does not have a disconnected contribution
    if method not in ["CCSD", "CCSD[T]"]:

        E_disconnected_doubles = 0

    E_CC = E_singles + E_connected_doubles + E_disconnected_doubles


    return E_CC, E_singles, E_connected_doubles, E_disconnected_doubles









def calculate_unrestricted_coupled_cluster_energy(o, v, g, t_ijab, method, t_ia=None, F=None):

    """
    
    Calculates the spin-unrestricted coupled cluster energy.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        t_ia (array, optional): Singles amplitudes
        F (array, optional): Spin-orbital Fock matrix
    
    Returns:
        E_CC (float): Unrestricted coupled cluster energy
        E_singles (float): Unrestricted coupled cluster energy from single excitations
        E_connected_doubles (float): Unrestricted coupled cluster energy from connected double excitations
        E_disconnected_doubles (float): Unrestricted coupled cluster energy from disconnected double excitations
    
    """


    # Contribution to coupled cluster energy from single excitations (should be zero)
    E_singles = np.einsum("ia,ia->", F[o, v], t_ia, optimize=True) if t_ia is not None and F is not None else 0

    # Contribution to coupled cluster energy from connected double excitations (should be large at normal bond lengths)
    E_connected_doubles = (1 / 4) * np.einsum("ijab,ijab->", g[o, o, v, v], t_ijab, optimize=True)

    # Contribution to coupled cluster energy from disconnected double excitations (should be small at normal bond lengths)
    E_disconnected_doubles = (1 / 2) * np.einsum("ijab,ia,jb->", g[o, o, v, v], t_ia, t_ia, optimize=True) if t_ia is not None else 0
    
    # In linearised methods, the total energy does not have a disconnected contribution
    if method not in ["CCSD", "CCSD[T]"]:

        E_disconnected_doubles = 0

    E_CC = E_singles + E_connected_doubles + E_disconnected_doubles


    return E_CC, E_singles, E_connected_doubles, E_disconnected_doubles









def coupled_cluster_initial_print(g, o, v, t_ijab, reference, method, calculation, silent=False):

    """
    
    Prints common and prerequisite information for coupled cluster calculations.

    Args:
        g (array): Two-electron integrals in spin or spatial orbital basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        reference (str): Either RHF or UHF
        method (str): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging
    
    """

    log_spacer(calculation, silent=silent, start="\n")
    log(f"              {method:>5} Energy and Density ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log(f"  Energy convergence tolerance:        {calculation.CC_conv:.10f}", calculation, 1, silent=silent)
    log(f"  Amplitude convergence tolerance:     {calculation.amp_conv:.10f}", calculation, 1, silent=silent)

    
    # Calculates the initial guess MP2 energy

    if reference == "RHF":
        
        E_MP2 = mp.calculate_restricted_t_amplitude_energy(t_ijab, g[o, o, v, v])

    else:
            
        E_MP2 = mp.calculate_t_amplitude_energy(t_ijab, g[o, o, v, v])

    
    log(f"\n  Guess t-amplitude MP2 energy:       {E_MP2:.10f}\n", calculation, 1, silent=silent)

    if calculation.coupled_cluster_damping_parameter != 0 : 
        
        log(f"  Using damping parameter of {calculation.coupled_cluster_damping_parameter:.2f} for convergence.", calculation, 1, silent=silent)

    if calculation.DIIS: 

        log(f"  Using DIIS, storing {calculation.max_DIIS_matrices} matrices, for convergence.", calculation, 1, silent=silent)

    log(f"\n  Starting {method} iterations...\n", calculation, 1, silent=silent)


    log_spacer(calculation, silent=silent)
    log("  Step          Correlation E               DE", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)


    return









def permute(array, idx_1, idx_2):

    """

    Incorporates antisymmetric permutation into an array. This is the definition of P- from the Stanton paper on CCSD.

    Args:
        array (array): Array wanted to be permuted
        idx_1 (int): First index
        idx_2 (int): Second index

    Returns:
        permuted_array (array): Antisymmetrically permuted array

    """

    permuted_array = array - array.swapaxes(idx_1, idx_2)

    return permuted_array











def apply_damping(damping_factor, t_ia, t_ia_old, t_ijab, t_ijab_old, t_ijkabc=None, t_ijkabc_old=None):

    """
    
    Applies a damping factor to the coupled cluster amplitudes.

    Args:
        damping_factor (float): Coupled cluster damping factor
        t_ia (array): Singles amplitudes
        t_ia_old (array): Singles amplitudes from last iteration
        t_ijab (array): Doubles amplitudes
        t_ijab_old (array): Doubles amplitudes from last iteration
        t_ijkabc (array, optional): Triples amplitudes
        t_ijkabc_old (array, optional): Triples amplitudes from last iteration

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        t_ijkabc (array): Triples amplitudes
    
    """

    t_ia = damping_factor * t_ia_old + (1 - damping_factor) * t_ia
    t_ijab = damping_factor * t_ijab_old + (1 - damping_factor) * t_ijab

    # Only damps the triples if they are there
    if t_ijkabc is not None:

        t_ijkabc = damping_factor * t_ijkabc_old + (1 - damping_factor) * t_ijkabc


    return t_ia, t_ijab, t_ijkabc










def update_DIIS(t_vectors, DIIS_error_vector, calculation, silent=False):

    """
    
    Extrapolates the t-amplitudes using DIIS.

    Args:
        t_vectors (list): List of t_ia_vector, t_ijab_vector and possibly t_ijkabc_vector
        DIIS_error_vector (list): Error vector for DIIS, same shape as t_vectors
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        t_ia (array): Extrapolated singles amplitudes
        t_ijab (array): Extrapolated doubles amplitudes
        t_ijkabc (array): Extrapolated triples amplitudes

    """


    # The first item in each vector should be deleted
    if len(t_vectors[0]) > calculation.max_DIIS_matrices:

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

        # Extrapolate t_ijab and t_ia, and t_ijkabc if there are all three vectors
        t_ia = np.tensordot(coeffs, t_vectors[0], axes=(0, 0))
        t_ijab = np.tensordot(coeffs, t_vectors[1], axes=(0, 0))
        t_ijkabc = np.tensordot(coeffs, t_vectors[2], axes=(0, 0)) if len(t_vectors) == 3 else None


    except np.linalg.LinAlgError:

        # Clears all the vectors in t_vectors
        [vec.clear() for vec in t_vectors]

        DIIS_error_vector.clear()

        t_ijab = t_ia = t_ijkabc = None

        log("   (Resetting DIIS)", calculation, 1, end="", silent=silent)


    return t_ia, t_ijab, t_ijkabc













def apply_DIIS(t_ia, t_ijab, t_ia_old, t_ijab_old, t_ia_vector, t_ijab_vector, DIIS_error_vector, step, calculation, t_ijkabc=None, t_ijkabc_old=None, t_ijkabc_vector=None, silent=False):
    
    """
    
    Applies DIIS to provide updated amplitudes.

    Args:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        t_ia_old (array): Singles amplitudes from previous iteration
        t_ijab_old (array): Doubles amplitudes from previous iteration
        t_ia_vector (array): Vector of all t_ia 
        t_ijab_vector (array): Vector of all t_ijab
        DIIS_error_vector (array): Vector for DIIS error matrices
        step (int): Coupled cluster iteration
        calculation (Calculation): Calculation object
        t_ijkabc (array, optional): Triples amplitudes
        t_ijkabc_old (array, optional): Triples amplitudes from previous iteration
        t_ijkabc_vector (array, optional): Vector of all t_ijkabc
        silent (bool, optional): Cancel logging

    Returns:
        t_ia (array): Extrapolated singles amplitudes
        t_ijab (array): Extrapolated doubles amplitudes
        t_ijkabc (array): Extrapolated triples amplitudes
        t_ia_vector (array): Vector of all t_ia
        t_ijab_vector (array): Vector of all t_ijab
        t_ijkabc_vector (array): Vector of all t_ijkabc
        DIIS_error_vector (array): Updated vector for DIIS error matrices
    
    """

    calculate_triples = t_ijkabc is not None

    t_ijab_residual = (t_ijab - t_ijab_old).ravel()
    t_ia_residual = (t_ia - t_ia_old).ravel()

    # Builds up vector of anmplitudes
    t_ijab_vector.append(t_ijab.copy())
    t_ia_vector.append(t_ia.copy())

    if calculate_triples:

        t_ijkabc_residual = (t_ijkabc - t_ijkabc_old).ravel()
        t_ijkabc_vector.append(t_ijkabc.copy())
        DIIS_error_vector.append(np.concatenate((t_ia_residual, t_ijab_residual, t_ijkabc_residual)))

    else:
        
        # Adds residuals to error vector
        DIIS_error_vector.append(np.concatenate((t_ia_residual, t_ijab_residual)))

    # Only starts DIIS after step 2 to prevent premature extrapolation
    if step > 2 and calculation.DIIS: 

        if calculate_triples:

            t_ia_DIIS, t_ijab_DIIS, t_ijkabc_DIIS = update_DIIS((t_ia_vector, t_ijab_vector, t_ijkabc_vector), DIIS_error_vector, calculation, silent=silent)

            t_ijkabc = t_ijkabc_DIIS if t_ijkabc_DIIS is not None else t_ijkabc

        else:

            t_ia_DIIS, t_ijab_DIIS, _ = update_DIIS((t_ia_vector, t_ijab_vector), DIIS_error_vector, calculation, silent=silent)

        # This will be "None" if there's a linear algebra error, so t_ijab not extrapolated in this case
        t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab
        t_ia = t_ia_DIIS if t_ia_DIIS is not None else t_ia

    return t_ia, t_ijab, t_ijkabc, t_ia_vector, t_ijab_vector, t_ijkabc_vector, DIIS_error_vector













def calculate_coupled_cluster_linearised_density(t_ia, t_ijab, n_orbitals, n_occ, o, v, calculation, molecular_orbitals, silent=False):

    """

    Calculates the coupled cluster linearised one-particle reduced density matrix.

    Args:

        t_ia (array, optional): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        n_orbitals (int): Number of spin orbitals
        molecular_orbitals (array): Spin-blocked molecular orbitals (SO) or molecular orbitals (spatial orbitals)
        n_occ (int): Number of occupied spin orbitals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        P (array): Full linearised density matrix in AO basis
        P_alpha (array): Alpha linearised density matrix in AO basis
        P_beta (array): Beta linearised density matrix in AO basis

    """

    log("\n  Constructing linearised density...    ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    
    reference = calculation.reference

    # Correlated part of density matrix from squared connected doubles
    P_CC = np.zeros((n_orbitals, n_orbitals))
    
    
    if reference == "RHF":

        l_ijab = t_ijab * 2 - t_ijab.swapaxes(0, 1) 

        P_CC[v, v] += np.einsum('ijbc,ijac->ab', t_ijab, l_ijab, optimize=True)
        P_CC[o, o] += -np.einsum('ikab,jkab->ij', t_ijab, l_ijab, optimize=True)
        
        P_CC[o, v] += t_ia + np.einsum("ijab,jb->ia", l_ijab, t_ia, optimize=True) 

    else:

        P_CC[v, v] +=  (1 / 2) * np.einsum('ijbc,ijac->ab', t_ijab, t_ijab, optimize=True)
        P_CC[o, o] += - (1 / 2) * np.einsum('ikab,jkab->ij', t_ijab, t_ijab, optimize=True)
        
        P_CC[o, v] += t_ia + np.einsum("ijab,jb->ia", t_ijab, t_ia, optimize=True) 


    # Linearised coupled-cluster density, only includes up to double excitations I think
    P_CC[v, o] = P_CC[o, v].T

    P_CC[v, v] += np.einsum("ia,ib->ab", t_ia, t_ia, optimize=True) 
    P_CC[o, o] -= np.einsum("ia,ja->ij", t_ia, t_ia, optimize=True)


    # Builds reference density matrix in SO or spatial orbital basis, diagonal in ones
    P_ref = np.zeros((n_orbitals, n_orbitals))
    P_ref[slice(0, n_occ), slice(0, n_occ)] = np.identity(n_occ)

    P = P_ref + P_CC 

    if reference == "UHF" or calculation.method == "CCSDT":

        # Transforms the density matrix from spin-orbital to atomic orbital basis
        P, P_alpha, P_beta = ci.transform_P_SO_to_AO(P, molecular_orbitals, n_orbitals) 

    else:

        # For RHF, all orbitals are doubly occupied
        P *= 2

        # Transforms the density matrix from the spatial orbital to atomic orbital basis
        P = molecular_orbitals @ P @ molecular_orbitals.T
        P_alpha = P_beta = P / 2

    log("     [Done]", calculation, 1, silent=silent)


    return P, P_alpha, P_beta










def calculate_T1_diagnostic(molecule, t_ia, spin_labels_sorted, n_occ, n_alpha, n_beta, calculation, silent=False):

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

        # Finds the alpha and beta indices that are occupied`
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











def find_and_print_largest_amplitudes(t_ia, t_ijab, n_occ, calculation, spin_orbital_labels_sorted=None, silent=False):
    
    """
    
    Searches for and prints the largest singles and doubles amplitudes.

    Args:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        n_occ (int): Number of occupied spin or spatial orbitals
        calculation (Calculation): Calculation object
        spin_orbital_labels_sorted (list, optional): Energy ordering of alpha and beta spin orbitals
        silent (bool, optional): Cancel logging
    
    
    """

    log("\n  Searching for largest amplitudes...        ", calculation, 2, end="", silent=silent); sys.stdout.flush()

    reference = calculation.reference

    # Fflattened absolute amplitudes and their raw indices
    t_ijab_flat = np.abs(t_ijab).ravel()
    t_ia_flat   = np.abs(t_ia).ravel()

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








def run_restricted_LCCD_iteration(g, o, v, t_ijab, e_ijab):

    """
    
    Updates the amplitudes for restricted LCCD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        e_ijab (array): Doubles epsilons tensor

    Returns:
        t_ijab (array): Doubles amplitudes

    """

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", g[v, o, v, o], t_ijab, optimize=True) - np.einsum("icak,kjbc->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", g[v, o, v, o], t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ijab = e_ijab * t_ijab_temporary


    return t_ijab








def run_unrestricted_LCCD_iteration(g, o, v, t_ijab, e_ijab):

    """
    
    Updates the amplitudes for unrestricted LCCD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        e_ijab (array): Doubles epsilons tensor

    Returns:
        t_ijab (array): Doubles amplitudes

    """

    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    t_ijab = e_ijab * t_ijab_temporary

    return t_ijab








def run_restricted_CCD_iteration(g, o, v, t_ijab, e_ijab, w):

    """
    
    Updates the amplitudes for restricted CCD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        e_ijab (array): Doubles epsilons tensor
        w (array): Chemists notation antisymmetric spatial orbital integrals

    Returns:
        t_ijab (array): Doubles amplitudes

    """

    F_ik = + np.einsum("cdkl,ilcd->ik", w[v, v, o, o], t_ijab, optimize=True)
    F_ca = - np.einsum("cdkl,klad->ca", w[v, v, o, o], t_ijab, optimize=True) 

    # Intermediates based on two-electron integrals
    W_ijkl = g[o, o, o, o] + np.einsum("cdkl,ijcd->ijkl", g[v, v, o, o], t_ijab, optimize=True)
    W_icak = g[o, v, v, o] -(1 / 2) * np.einsum("dclk,ilda->icak", g[v, v, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum("dclk,ilad->icak", w[v, v, o, o], t_ijab, optimize=True)
    W_ciak = g[v, o, v, o] - (1 / 2) * np.einsum("cdlk,ilda->ciak", g[v, v, o, o], t_ijab, optimize=True) 

    # Updating doubles amplitudes
    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", W_ijkl, t_ijab, optimize=True) 
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v] , t_ijab, optimize=True) 
    t_ijab_temporary += np.einsum("ca,ijcb->ijab", F_ca , t_ijab, optimize=True) - np.einsum("ik,kjab->ijab", F_ik, t_ijab, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", W_icak, t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", W_ciak, t_ijab, optimize=True) - np.einsum("icak,kjbc->ijab", W_icak, t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", W_ciak, t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ijab = e_ijab * t_ijab_temporary

    return t_ijab







def run_unrestricted_CCD_iteration(g, o, v, t_ijab, e_ijab):

    """
    
    Updates the amplitudes for unrestricted CCD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ijab (array): Doubles amplitudes
        e_ijab (array): Doubles epsilons tensor

    Returns:
        t_ijab (array): Doubles amplitudes

    """

    # Calculates contribution from LCCD
    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    # Calculates contribution from full CCD
    t_ijab_temporary += - (1 / 2) * permute(np.einsum("cdkl,ijac,klbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 2, 3) - (1 / 2) * permute(np.einsum("cdkl,ikab,jlcd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)
    t_ijab_temporary += (1 / 4) * np.einsum("cdkl,ijcd,klab->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True)
    t_ijab_temporary += permute(np.einsum("cdkl,ikac,jlbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)

    t_ijab = e_ijab * t_ijab_temporary


    return t_ijab











def run_restricted_LCCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w):

    """
    
    Updates the amplitudes for restricted LCCSD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        w (array): Chemists notation antisymmetric spatial orbital integrals

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

    t_ia_temporary = np.einsum("icak,kc->ia", w[o, v, v, o], t_ia, optimize=True) + np.einsum("cdak,ikcd->ia", w[v, v, v, o], t_ijab, optimize=True) - np.einsum("ickl,klac->ia", w[o, v, o, o], t_ijab, optimize=True) 

    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) 
    t_ijab_temporary += np.einsum("icab,jc->ijab", g[o, v, v, v], t_ia, optimize=True) - np.einsum("ijak,kb->ijab", g[o, o, v, o], t_ia, optimize=True) 
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", g[v, o, v, o], t_ijab, optimize=True) - np.einsum("icak,kjbc->ijab", g[o, v, v, o], t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", g[v, o, v, o], t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary

    return t_ia, t_ijab








def run_unrestricted_LCCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F):

    """
    
    Updates the amplitudes for unrestricted LCCSD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        F (array): Fock matrix in spin orbital basis

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

    # Equations from Crawford guide to coupled cluster, linearised, singles
    t_ia_temporary = F[o, v] + np.einsum("ac,ic->ia", F[v, v], t_ia, optimize=True) - np.einsum("ki,ka->ia", F[o, o], t_ia, optimize=True) + np.einsum("kc,ikac->ia", F[o, v], t_ijab, optimize=True)
    t_ia_temporary += np.einsum("kaci,kc->ia", g[o, v, v, o], t_ia, optimize=True) + (1 / 2) * np.einsum("kacd,kicd->ia", g[o, v, v, v], t_ijab, optimize=True) - (1 / 2) * np.einsum("klci,klca->ia", g[o, o, v, o], t_ijab, optimize=True)

    # Equations from Crawford guide to coupled cluster, linearised, connected doubles, shared with LCCD
    t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
    t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)

    # Equations from Crawford guide to coupled cluster, linearised, doubles
    t_ijab_temporary += permute(np.einsum("bc,ijac->ijab", F[v, v], t_ijab, optimize=True), 2, 3) - permute(np.einsum("kj,ikab->ijab", F[o, o], t_ijab, optimize=True), 0, 1)
    t_ijab_temporary += permute(np.einsum("abcj,ic->ijab", g[v, v, v, o], t_ia, optimize=True), 0, 1) - permute(np.einsum("kbij,ka->ijab", g[o, v, o, o], t_ia, optimize=True), 2, 3)

    t_ia += e_ia * t_ia_temporary
    t_ijab += e_ijab * t_ijab_temporary

    return t_ia, t_ijab









def run_restricted_QCISD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w):

    """
    
    Updates the amplitudes for restricted QCISD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        w (array): Chemists notation antisymmetric spatial orbital integrals

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

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

    return t_ia, t_ijab








def run_unrestricted_QCISD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F):

    """
    
    Updates the amplitudes for unrestricted QCISD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

    kronecker_delta = np.eye(F.shape[1])

    # Builds curly F intermediates (Fock matrix intermediates), all equations from Stanton paper on DPD coupled cluster, referenced by Crawford tutorials - curtailed by me into QCISD
    F_ae = F[v, v] - kronecker_delta[v, v] * F[v, v]  - (1 / 2) * np.einsum("mnaf,mnef->ae", t_ijab, g[o, o, v, v], optimize=True)
    F_mi = F[o, o] - kronecker_delta[o, o] * F[o, o]  + (1 / 2) * np.einsum("inef,mnef->mi", t_ijab, g[o, o, v, v], optimize=True)
    F_me = F[o, v] + np.einsum("nf,mnef->me", t_ia, g[o, o, v, v], optimize=True) 
    
    # Builds curly W intermediates (two-electron intermediates)
    W_mnij = g[o, o, o, o]  + (1 / 4) * np.einsum("ijef,mnef->mnij", t_ijab, g[o, o, v, v], optimize=True)
    W_abef = g[v, v, v, v] + (1 / 4) * np.einsum("mnab,mnef->abef", t_ijab, g[o, o, v, v], optimize=True)
    W_mbej = g[o, v, v, o]  - np.einsum("jnfb,mnef->mbej", (1 / 2) * t_ijab, g[o, o, v, v], optimize=True)


    # Builds t_ia tensor from intermediates
    t_ia_temporary = F[o, v] + np.einsum("ie,ae->ia", t_ia, F_ae, optimize=True) - np.einsum("ma,mi->ia", t_ia, F_mi, optimize=True) 
    t_ia_temporary += np.einsum("imae,me->ia", t_ijab, F_me, optimize=True) - np.einsum("nf,naif->ia", t_ia, g[o, v, o, v], optimize=True) - (1 / 2) * np.einsum("imef,maef->ia", t_ijab, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnae,nmei->ia", t_ijab, g[o, o, v, o], optimize=True)

    
    # Builds t_ijab tensor from intermediates, pairs of terms from Stanton
    t_ijab_temporary = g[o, o, v, v] + permute(np.einsum("ijae,be->ijab", t_ijab, F_ae, optimize=True), 2, 3) - permute(np.einsum("imab,mj->ijab", t_ijab, F_mi, optimize=True), 0, 1)
    t_ijab_temporary += (1 / 2) * np.einsum("mnab,mnij->ijab", t_ijab, W_mnij, optimize=True) + (1 / 2) * np.einsum("ijef,abef->ijab", t_ijab, W_abef, optimize=True)
    t_ijab_temporary += permute(permute(np.einsum("ijmabe->ijab", np.einsum("imae,mbej->ijmabe", t_ijab, W_mbej, optimize=True), optimize=True), 2, 3), 0, 1)
    t_ijab_temporary += permute(np.einsum("ie,abej->ijab", t_ia, g[v, v, v, o], optimize=True), 0, 1) - permute(np.einsum("ma,mbij->ijab", t_ia, g[o, v, o, o], optimize=True), 2, 3)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary
    
    return t_ia, t_ijab









def run_restricted_CCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w, F):

    """
    
    Updates the amplitudes for restricted CCSD.

    Args:
        g (array): Spatial orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        w (array): Chemists notation antisymmetric spatial orbital integrals
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

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
    W_icak += - (1 / 2) * np.einsum("dclk,ilda->icak", g[v, v, o, o], t_ijab, optimize=True) - np.einsum("dclk,id,la->icak", g[v, v, o, o], t_ia, t_ia, optimize=True) + (1 / 2) * np.einsum("dclk,ilad->icak", w[v, v, o, o], t_ijab, optimize=True)

    W_ciak = g[v, o, v, o] - np.einsum("cilk,la->ciak", g[v, o, o, o], t_ia, optimize=True) + np.einsum("dcka,id->ciak", g[v, v, o, v], t_ia, optimize=True)
    W_ciak += - (1 / 2) * np.einsum("cdlk,ilda->ciak", g[v, v, o, o], t_ijab, optimize=True) - np.einsum("cdlk,id,la->ciak", g[v, v, o, o], t_ia, t_ia, optimize=True)

    # Updating singles amplitudes
    t_ia_temporary = np.einsum("ca,ic->ia", F_ca - F[v, v], t_ia, optimize=True) - np.einsum("ik,ka->ia", F_ik - F[o, o], t_ia, optimize=True) + np.einsum("ck,kica->ia", F_ck, 2 * t_ijab - t_ijab.swapaxes(0, 1), optimize=True)
    t_ia_temporary += np.einsum("ck,ic,ka->ia", F_ck, t_ia, t_ia, optimize=True) + np.einsum("icak,kc->ia", w[o, v, v, o], t_ia, optimize=True) + np.einsum("cdak,ikcd->ia", w[v, v, v, o], t_ijab, optimize=True)
    t_ia_temporary += np.einsum("cdak,ic,kd->ia", w[v, v, v, o], t_ia, t_ia, optimize=True) - np.einsum("ickl,klac->ia", w[o, v, o, o], t_ijab, optimize=True) - np.einsum("ickl,ka,lc->ia", w[o, v, o, o], t_ia, t_ia, optimize=True)

    # Updating doubles amplitudes
    t_ijab_temporary = (1 / 2) * g[o, o, v, v] + (1 / 2) * np.einsum("ijkl,klab->ijab", W_ijkl, t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,ka,lb->ijab", W_ijkl, t_ia, t_ia, optimize=True)
    t_ijab_temporary += (1 / 2) * np.einsum("cdab,ijcd->ijab", W_cdab, t_ijab, optimize=True) + (1 / 2) * np.einsum("cdab,ic,jd->ijab", W_cdab, t_ia, t_ia, optimize=True)
    t_ijab_temporary += np.einsum("ca,ijcb->ijab", L_ca - F[v, v], t_ijab, optimize=True) - np.einsum("ik,kjab->ijab", L_ik - F[o, o], t_ijab, optimize=True)
    t_ijab_temporary += np.einsum("icab,jc->ijab", g[o, v, v, v], t_ia, optimize=True) - np.einsum("ickb,ka,jc->ijab", g[o, v, o, v], t_ia, t_ia, optimize=True) - np.einsum("ijak,kb->ijab", g[o, o, v, o], t_ia, optimize=True) - np.einsum("icak,jc,kb->ijab", g[o, v, v, o], t_ia, t_ia, optimize=True)
    t_ijab_temporary += 2 * np.einsum("icak,kjcb->ijab", W_icak, t_ijab, optimize=True) - np.einsum("ciak,kjcb->ijab", W_ciak, t_ijab, optimize=True) - np.einsum("icak,kjbc->ijab", W_icak, t_ijab, optimize=True) - np.einsum("cibk,kjac->ijab", W_ciak, t_ijab, optimize=True)

    t_ijab_temporary += t_ijab_temporary.transpose(1, 0, 3, 2)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary

    return t_ia, t_ijab









def run_unrestricted_CCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F):

    """
    
    Updates the amplitudes for unrestricted CCSD.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes

    """

    kronecker_delta = np.eye(F.shape[1])

    # Build tau tensors, all equations from Stanton paper on DPD coupled cluster, referenced by Crawford tutorials
    tau_tilde_ijab = t_ijab + (1 / 2) * (np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True))
    tau_ijab = t_ijab + np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True)


    # Builds curly F intermediates (Fock matrix intermediates)
    F_ae = F[v, v] - kronecker_delta[v, v] * F[v, v] - (1 / 2) * np.einsum("me,ma->ae", F[o, v], t_ia, optimize=True) + np.einsum("mf,mafe->ae", t_ia, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnaf,mnef->ae", tau_tilde_ijab, g[o, o, v, v], optimize=True)
    F_mi = F[o, o] - kronecker_delta[o, o] * F[o, o]  + (1 / 2) * np.einsum("ie,me->mi", t_ia, F[o, v], optimize=True) + np.einsum("ne,mnie->mi", t_ia, g[o, o, o, v], optimize=True) + (1 / 2) * np.einsum("inef,mnef->mi", tau_tilde_ijab, g[o, o, v, v], optimize=True)
    F_me = F[o, v] + np.einsum("nf,mnef->me", t_ia, g[o, o, v, v], optimize=True) 
    

    # Builds curly W intermediates (two-electron intermediates)
    W_mnij = g[o, o, o, o] + permute(np.einsum("je,mnie->mnij", t_ia, g[o, o, o, v], optimize=True), 2, 3) + (1 / 4) * np.einsum("ijef,mnef->mnij", tau_ijab, g[o, o, v, v], optimize=True)
    W_abef = g[v, v, v, v] - permute(np.einsum("mb,amef->abef", t_ia, g[v, o, v, v], optimize=True), 0, 1) + (1 / 4) * np.einsum("mnab,mnef->abef", tau_ijab, g[o, o, v, v], optimize=True)
    W_mbej = g[o, v, v, o] + np.einsum("jf,mbef->mbej", t_ia, g[o, v, v, v], optimize=True) - np.einsum("nb,mnej->mbej", t_ia, g[o, o, v, o], optimize=True) - np.einsum("jnfb,mnef->mbej", (1 / 2) * t_ijab + np.einsum("jf,nb->jnfb", t_ia, t_ia, optimize=True), g[o, o, v, v], optimize=True)


    # Builds t_ia tensor from intermediates
    t_ia_temporary = F[o, v] + np.einsum("ie,ae->ia", t_ia, F_ae, optimize=True) - np.einsum("ma,mi->ia", t_ia, F_mi, optimize=True) 
    t_ia_temporary += np.einsum("imae,me->ia", t_ijab, F_me, optimize=True) - np.einsum("nf,naif->ia", t_ia, g[o, v, o, v], optimize=True) - (1 / 2) * np.einsum("imef,maef->ia", t_ijab, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnae,nmei->ia", t_ijab, g[o, o, v, o], optimize=True)

    
    # Builds t_ijab tensor from intermediates, pairs of terms from Stanton
    t_ijab_temporary = g[o, o, v, v] + permute(np.einsum("ijae,be->ijab", t_ijab, F_ae - (1 / 2) * np.einsum("mb,me->be", t_ia, F_me,optimize=True), optimize=True), 2, 3) - permute(np.einsum("imab,mj->ijab", t_ijab, F_mi + (1 / 2) * np.einsum("je,me->mj", t_ia, F_me, optimize=True),optimize=True), 0, 1)
    t_ijab_temporary += (1 / 2) * np.einsum("mnab,mnij->ijab", tau_ijab, W_mnij, optimize=True) + (1 / 2) * np.einsum("ijef,abef->ijab", tau_ijab, W_abef, optimize=True)
    t_ijab_temporary += permute(permute(np.einsum("ijmabe->ijab", np.einsum("imae,mbej->ijmabe", t_ijab, W_mbej, optimize=True) - np.einsum("ie,ma,mbej->ijmabe", t_ia, t_ia, g[o, v, v, o], optimize=True), optimize=True), 2, 3), 0, 1)
    t_ijab_temporary += permute(np.einsum("ie,abej->ijab", t_ia, g[v, v, v, o], optimize=True), 0, 1) - permute(np.einsum("ma,mbij->ijab", t_ia, g[o, v, o, o], optimize=True), 2, 3)

    t_ia = e_ia * t_ia_temporary
    t_ijab = e_ijab * t_ijab_temporary
    
    return t_ia, t_ijab









def run_unrestricted_CCSDT_iteration(g, o, v, t_ia, t_ijab, t_ijkabc, e_ia, e_ijab, e_ijkabc, F):

    """
    
    Updates the amplitudes for unrestricted CCSDT.

    Args:
        g (array): Spin orbital two-electron integrals
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        t_ijkabc (array): Triples amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        e_ijkabc (array): Triples epsilons tensor
        F (array): Fock matrix in spatial orbital basis

    Returns:
        t_ia (array): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        t_ijkabc (array): Triples amplitudes

    """

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


    return t_ia, t_ijab, t_ijkabc









def calculate_restricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent=False):


    """ 
    
    Calculates the perturbative triples energy for restricted CCSD(T).

    Args:
        g (array): Non-antisymmetrised ERI in spatial MO basis
        e_ijkabc (array): Triples epsilon tensor
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        method (str): Electronic structure method
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_CCSD_T (float): Restricted CCSD(T) energy

    """

    method = method.replace("[", "(").replace("]", ")")

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                   {method} Energy ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)


    def P_ijkabc(array):

        # Three index permutation per Lee

        return array + array.transpose(1, 0, 2, 4, 3, 5) + array.transpose(2, 1, 0, 5, 4, 3) + array.transpose(0, 2, 1, 3, 5, 4) + array.transpose(2, 0, 1, 5, 3, 4) + array.transpose(1, 2, 0, 4, 5, 3)


    log("  Forming disconnected amplitudes...         ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Calculation of key intermediate tensors
    V_ijkabc = np.einsum("jkbc,ia->ijkabc", g[o, o, v, v], t_ia, optimize=True) + np.einsum("ikac,jb->ijkabc", g[o, o, v, v], t_ia, optimize=True) + np.einsum("ijab,kc->ijkabc", g[o, o, v, v], t_ia, optimize=True)

    space = " "

    if "QCISD" in method: 
        
        # This factor of two arises because part of the MP5 disconnected triples are included in the CCSD equations, but not the QCISD equations
        V_ijkabc *= 2
        space = ""

    log(f"[Done]", calculation, 1, silent=silent)

    log("  Forming connected amplitudes...            ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    W_ijkabc = P_ijkabc(np.einsum("ibaf,kjcf->ijkabc", g[o, v, v, v], t_ijab, optimize=True) - np.einsum("ijam,mkbc->ijkabc", g[o, o, v, o], t_ijab, optimize=True))

    W = 4 * W_ijkabc + W_ijkabc.transpose(2, 0, 1, 3, 4, 5) + W_ijkabc.transpose(1, 2, 0, 3, 4, 5) - 4 * W_ijkabc.transpose(2, 1, 0, 3, 4, 5) - W_ijkabc.transpose(0, 2, 1, 3, 4, 5) - W_ijkabc.transpose(1, 0, 2, 3, 4, 5)
    
    log(f"[Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating {method} correlation energy... {space}", calculation, 1, end="", silent=silent); sys.stdout.flush()

    E_CCSD_T = (1 / 3) * np.einsum("ijkabc,ijkabc,ijkabc->", W_ijkabc + V_ijkabc, W, e_ijkabc, optimize=True)
    
    log(f"[Done]\n\n  {method} correlation energy:       {space} {E_CCSD_T:13.10f}", calculation, 1, silent=silent) 


    return E_CCSD_T









def calculate_unrestricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent=False):

    """ 
    
    Calculates the perturbative triples energy for CCSD(T).

    Args:
        g (array): Spin orbital ERI tensor
        e_ijkabc (array): Triples epsilon tensor
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_CCSD_T (float): CCSD(T) energy

    """

    method = method.replace("[", "(").replace("]", ")")
    method = method.split("U")[1] if "U" in method else method

    log_spacer(calculation, silent=silent, start="\n")
    log(f"                   {method} Energy  ", calculation, 1, silent=silent, colour="white")
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
    if "QCISD" in method: 
        
        t_ijkabc_d *= 2
        space = ""

    log(f"[Done]", calculation, 1, silent=silent)

    log("  Forming connected amplitudes...            ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    c_ijkabc = np.einsum("jkae,eibc->ijkabc", t_ijab, g[v, o, v, v], optimize=True) - np.einsum("imbc,majk->ijkabc", t_ijab, g[o, v, o, o], optimize=True)
    t_ijkabc_c = np.einsum("ijkabc,ijkabc->ijkabc", e_ijkabc, permute_three_indices(permute_three_indices(c_ijkabc, 3, 4, 5), 0, 1, 2), optimize=True)

    log(f"[Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating {method} correlation energy... {space}", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    # Final contraction for the CCSD(T) energy using the connected and disconnected approximate triples amplitudes
    E_CCSD_T = (1 / 36) * np.einsum("ijkabc,ijkabc->", t_ijkabc_c / e_ijkabc, t_ijkabc_c + t_ijkabc_d, optimize=True)

    log(f"[Done]\n\n  {method} correlation energy:       {space} {E_CCSD_T:13.10f}", calculation, 1, silent=silent) 


    return E_CCSD_T










def calculate_coupled_cluster_energy(g, o, v, t_ia, t_ijab, t_ijkabc, e_ia, e_ijab, e_ijkabc, F, method, reference, calculation, silent=False):

    """
    
    Calculates the coupled cluster energy in an iterative procedure.

    Args:
        g (array): Two-electron integrals in spin or spatial orbital basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        t_ia (array): Guess singles amplitudes
        t_ijab (array): Guess doubles amplitudes
        t_ijkabc (array): Guess triples amplitudes
        e_ia (array): Singles epsilons tensor
        e_ijab (array): Doubles epsilons tensor
        e_ijkabc (array): Triples epsilons tensor
        F (array): Fock matrix in spin or spatial orbital basis
        method (string): Electronic structure method
        reference (string): Either RHF or UHF
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        E_CC (float): Coupled cluster energy
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        t_ijkabc (array): Converged triples amplitudes
    
    """

    E_CC = 0.0

    CC_max_iter = calculation.CC_max_iter
    calculate_triples = "CCSDT" in method

    # Chops of "U" in front of method
    method = method.split("U")[1] if "U" in method else method
    method = method.split("[T]")[0] if "T" in method else method

    # Sets up DIIS vectors
    t_ia_vector, t_ijab_vector, t_ijkabc_vector, DIIS_error_vector = [], [], [], []

    # Common printing for all coupled cluster calculations
    coupled_cluster_initial_print(g, o, v, t_ijab, reference, method, calculation, silent=silent)


    if reference == "RHF":
        
        # Useful intermediate quantity for restricted coupled cluster
        w = 2 * g - g.swapaxes(0, 1)


    for step in range(1, CC_max_iter + 1):

        E_old = E_CC

        t_ia_old = t_ia.copy()
        t_ijab_old = t_ijab.copy()
        
        # Only bother with DIIS and damping on triples if CCSDT is requested
        if calculate_triples: 
            
            t_ijkabc_old = t_ijkabc.copy()



        if reference == "RHF" and not calculate_triples:

            if "LCCD" in method:

                t_ijab = run_restricted_LCCD_iteration(g, o, v, t_ijab, e_ijab)

            elif "CCD" in method:

                t_ijab = run_restricted_CCD_iteration(g, o, v, t_ijab, e_ijab, w)

            elif "LCCSD" in method:

                t_ia, t_ijab = run_restricted_LCCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w)

            elif "QCISD" in method:

                t_ia, t_ijab = run_restricted_QCISD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w)

            elif "CCSD" in method:

                t_ia, t_ijab = run_restricted_CCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, w, F)


            # Use the energy expression from restricted coupled cluster
            E_CC, E_CC_singles, E_CC_connected_doubles, E_CC_disconnected_doubles = calculate_restricted_coupled_cluster_energy(o, v, w, t_ijab, method, t_ia=t_ia, F=F)
            


        elif reference == "UHF" or calculate_triples:

            if "LCCD" in method:

                t_ijab = run_unrestricted_LCCD_iteration(g, o, v, t_ijab, e_ijab)

            elif "CCD" in method:

                t_ijab = run_unrestricted_CCD_iteration(g, o, v, t_ijab, e_ijab)

            elif "LCCSD" in method:

                t_ia, t_ijab = run_unrestricted_LCCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F)

            elif "CCSDT" in method:

                t_ia, t_ijab, t_ijkabc = run_unrestricted_CCSDT_iteration(g, o, v, t_ia, t_ijab, t_ijkabc, e_ia, e_ijab, e_ijkabc, F)
            
            elif "QCISD" in method:

                t_ia, t_ijab = run_unrestricted_QCISD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F)

            elif "CCSD" in method:

                t_ia, t_ijab = run_unrestricted_CCSD_iteration(g, o, v, t_ia, t_ijab, e_ia, e_ijab, F)


            # Use the energy expression from unrestricted coupled cluster
            E_CC, E_CC_singles, E_CC_connected_doubles, E_CC_disconnected_doubles = calculate_unrestricted_coupled_cluster_energy(o, v, g, t_ijab, method, t_ia=t_ia, F=F)



        # Makes sure all amplitudes are finite
        if E_CC > 1000 or any(not np.isfinite(x).all() for x in (t_ia, t_ijab, t_ijkabc)):

            error(f"Non-finite encountered in {method} iteration. Try stronger damping with the CCDAMP keyword?.")


        # Calculates the change in energy
        delta_E = E_CC - E_old

        # Prints the correlation energy at the current iteration
        log(f"  {step:3.0f}           {E_CC:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv and np.linalg.norm(t_ia - t_ia_old) < calculation.amp_conv: break


        elif step >= CC_max_iter: error(f"The {method} iterations failed to converge! Try increasing the maximum iterations with CCMAXITER?")


        if calculate_triples:
            
            # Update amplitudes with DIIS
            t_ia, t_ijab, t_ijkabc, t_ia_vector, t_ijab_vector, t_ijkabc_vector, DIIS_error_vector = apply_DIIS(t_ia, t_ijab, t_ia_old, t_ijab_old, t_ia_vector, t_ijab_vector, DIIS_error_vector, step, calculation, t_ijkabc=t_ijkabc, t_ijkabc_old=t_ijkabc_old, t_ijkabc_vector=t_ijkabc_vector, silent=silent)

            # Applies damping to amplitudes
            t_ia, t_ijab, t_ijkabc = apply_damping(calculation.coupled_cluster_damping_parameter, t_ia, t_ia_old, t_ijab, t_ijab_old, t_ijkabc=t_ijkabc, t_ijkabc_old=t_ijkabc_old)

        else:

            # Update amplitudes with DIIS           
            t_ia, t_ijab, _, t_ia_vector, t_ijab_vector, _, DIIS_error_vector = apply_DIIS(t_ia, t_ijab, t_ia_old, t_ijab_old, t_ia_vector, t_ijab_vector, DIIS_error_vector, step, calculation, t_ijkabc=None, t_ijkabc_old=None, t_ijkabc_vector=None, silent=silent)
            
            # Applies damping to amplitudes
            t_ia, t_ijab, _ = apply_damping(calculation.coupled_cluster_damping_parameter, t_ia, t_ia_old, t_ijab, t_ijab_old, t_ijkabc=None, t_ijkabc_old=None)


    log_spacer(calculation, silent=silent)

    log(f"\n  Singles contribution:               {E_CC_singles:13.10f}", calculation, 1, silent=silent)
    log(f"  Connected doubles contribution:     {E_CC_connected_doubles:13.10f}", calculation, 1, silent=silent)
    log(f"  Disconnected doubles contribution:  {E_CC_disconnected_doubles:13.10f}", calculation, 1, silent=silent)

    log(f"\n  {method} correlation energy:  {" " * (10 - len(method))}    {E_CC:.10f}", calculation, 1, silent=silent)


    return E_CC, t_ia, t_ijab, t_ijkabc









def begin_coupled_cluster_calculation(method, molecule, SCF_output, ERI_AO, X, H_core, calculation, silent=False):

    """
    
    Sets off a coupled cluster calculation.

    Args:
        method (str): Electronic structure method
        molecule (Molecule): Molecule object
        SCF_Output (Output): Output object
        ERI_AO (array): Electron repulsion integrals in AO basis
        X (array): Fock transformation matrix in AO basis
        H_core (array): Core Hamiltonian matrix in AO basis
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        E_CC (float): Coupled cluster energy
        E_perturbative_triples (float): Energy from perturbative triples
        P (array): Density matrix in AO basis
        P_alpha (array): Alpha spin density matrix in AO basis
        P_beta (array): Beta spin density matrix in AO basis
        occupancies (array): Natural orbital occupancies
        natural_orbitals (array): Natural orbitals

    """

    E_CC = 0
    E_perturbative_triples = 0
    occupancies, natural_orbitals = None, None

    reference = calculation.reference

    # All CCSDT calculations go via spin orbitals, so n_orbitals needs to be n_SO even for RHF references
    n_orbitals = molecule.n_orbitals if "CCSDT" not in method else molecule.n_SO

    calculate_triples = method in ["CCSDT", "UCCSDT", "CCSD[T]", "UCCSD[T]", "QCISD[T]", "UQCISD[T]"]

    # All CCSDT calculations must go via spin orbital route
    if reference == "RHF" and "CCSDT" not in method:

        n_occ = molecule.n_doubly_occ

        g, molecular_orbitals, epsilons, o, v = ci.begin_spatial_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=silent)
        
        # All coupled cluster calculations use chemists' notation
        g = g.transpose(0, 2, 1, 3)
        
        # Builds spatial orbital Fock matrix
        F = np.diag(epsilons)

        # Don't need to worry about these for spatial orbital calculations
        spin_labels_sorted = None
        spin_orbital_labels_sorted = None


    else:
        
        n_occ = molecule.n_occ
        
        g, molecular_orbitals, epsilons, _, o, v, spin_labels_sorted, spin_orbital_labels_sorted = ci.begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=silent)
        
        # Builds spin orbital core Hamiltonian
        H_core_spin_block = ci.spin_block_core_Hamiltonian(H_core)
        H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, molecular_orbitals)

        # Combines the spin-orbital core Hamiltonian and ERIs to get the spin-orbital basis Fock matrix
        F = ci.build_spin_orbital_Fock_matrix(H_core_SO, g, slice(0, n_occ))


    log("\n Preparing arrays for coupled cluster...     ", calculation, 1, end="", silent=silent); sys.stdout.flush()


    # Builds the inverse epsilon tensors - skips triples if not required
    e_ia = ci.build_singles_epsilons_tensor(epsilons, o, v)
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
    e_ijkabc = ci.build_triples_epsilons_tensor(epsilons, o, v) if calculate_triples else np.zeros_like(e_ijab)

    # Defines the guess t-amplitudes
    t_ia = np.einsum("ia,ia->ia", e_ia, F[o, v], optimize=True)
    t_ijab = ci.build_MP2_t_amplitudes(g[o, o, v, v], e_ijab)
    t_ijkabc = np.zeros_like(e_ijkabc)

    log("[Done]", calculation, 1, silent=silent)

    # Calculates the coupled cluster energy
    E_CC, t_ia, t_ijab, t_ijkabc = calculate_coupled_cluster_energy(g, o, v, t_ia, t_ijab, t_ijkabc, e_ia, e_ijab, e_ijkabc, F, method, reference, calculation, silent=silent)

    # Determines and prints the T1 diagnostic and norm of the singles
    calculate_T1_diagnostic(molecule, t_ia, spin_labels_sorted, n_occ, molecule.n_alpha, molecule.n_beta, calculation, silent=silent)

    # Determines and prints the largest amplitudes
    find_and_print_largest_amplitudes(t_ia, t_ijab, n_occ, calculation, spin_orbital_labels_sorted=spin_orbital_labels_sorted, silent=silent)

    # Calculates the unrelaxed density matrix in the AO basis
    P, P_alpha, P_beta = calculate_coupled_cluster_linearised_density(t_ia, t_ijab, n_orbitals, n_occ, o, v, calculation, molecular_orbitals, silent=silent)
    
    # If NATORBS is used, calculate and print the natural orbitals
    if calculation.natural_orbitals: 
        
        occupancies, natural_orbitals = mp.calculate_natural_orbitals(P, X, calculation, silent=silent)


    if "CCSD[T]" in method or "QCISD[T]" in method:

        if reference == "UHF":

            E_perturbative_triples = calculate_unrestricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent=silent)

        else:

            E_perturbative_triples = calculate_restricted_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, method, calculation, silent=silent)


    log_spacer(calculation, silent=silent)


    return E_CC, E_perturbative_triples, P, P_alpha, P_beta, occupancies, natural_orbitals
