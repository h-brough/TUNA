import numpy as np
import tuna_ci as ci
import tuna_mp as mp
from tuna_util import *
import sys




def calculate_coupled_cluster_energy(o, v, g, t_ijab, t_ia=np.zeros(1), F=None):

    """
    
    Calculates the coupled cluster energy.

    Args:
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        g (array): Antisymmetrised two-electron integrals
        t_ijab (array): Doubles amplitudes
        t_ia (array, optional): Singles amplitudes
        F (array, optional): Spin-orbital Fock matrix

    Returns:
        E_CC (float): Coupled cluster energy
        E_singles (float): Energy due to single excitations
        E_connected_doubles (float): Energy due to connected doubles
        E_disconnected_doubles (float): Energy due to disconnected doubles
    
    """

    E_singles = np.einsum("ia,ia->", F[o, v], t_ia, optimize=True) if F is not None else 0
    E_connected_doubles = (1 / 4) * np.einsum("ijab,ijab->", g[o, o, v, v], t_ijab, optimize=True)
    E_disconnected_doubles = (1 / 2) * np.einsum("ijab,ia,jb->", g[o, o, v, v], t_ia, t_ia, optimize=True) if t_ia.all() != 0 else 0

    E_CC = E_singles + E_connected_doubles + E_disconnected_doubles

    return E_CC, E_singles, E_connected_doubles, E_disconnected_doubles






def coupled_cluster_initial_print(t_ijab, g, method, o, v, calculation, silent=False):

    """

    Prints out common logging between coupled cluster methods at the start of the iterations.

    Args:
        t_ijab (array): Doubles amplitudes
        g (array): Spin-orbital ERI tensor
        method (str): Coupled cluster method
        o (slice): Occupied spin-orbital slice
        v (slice): Virtual spin-orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        None

    """

    log_spacer(calculation, silent=silent)

    log(f"  Energy convergence tolerance:        {calculation.CC_conv:.10f}", calculation, 1, silent=silent)
    log(f"  Amplitude convergence tolerance:     {calculation.amp_conv:.10f}", calculation, 1, silent=silent)

    # Calculates the initial guess MP2 energy
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










def apply_damping(t_ijab, t_ijab_old, calculation, t_ia=None, t_ia_old=None, t_ijkabc=None, t_ijkabc_old=None):

    """

    Damps the t-amplitude tensors based on the damping parameter.

    Args:
        t_ijab (array): Doubles amplitudes
        t_ijab_old (array): Old doubles amplitudes
        calculation (Calculation): Calculation object
        t_ia (array, optional): Singles amplitudes
        t_ia_old (array, optional): Old singles amplitudes
        t_ijkabc (array, optional): Triples amplitudes
        t_ijkabc_old (array, optional): Old triples amplitudes

    Returns:
        t_ijab (array): Damped doubles amplitudes
        t_ia (array): Damped singles amplitudes
        t_ijkabc (array): Damped triples amplitudes

    """

    damp_param = calculation.coupled_cluster_damping_parameter

    t_ijab = damp_param * t_ijab_old + (1 - damp_param) * t_ijab

    # Accounts for methods like CCD with no singles or triples
    if t_ia is not None:

        t_ia = damp_param * t_ia_old + (1 - damp_param) * t_ia

    if t_ijkabc is not None:

        t_ijkabc = damp_param * t_ijkabc_old + (1 - damp_param) * t_ijkabc


    return t_ijab, t_ia, t_ijkabc










def update_DIIS(t_vectors, DIIS_error_vector, calculation, silent=False):

    """
    
    Extrapolates the t-amplitudes using DIIS.

    Args:
        t_vectors (list): List of either only t_ijab_vector, or this appended with t_ia_vector and/or t_ijkabc_vector
        DIIS_error_vector (list): Error vector for DIIS, same shape as t_vectors
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        t_ijab (array): Extrapolated doubles amplitudes
        t_ia (array): Extrapolated singles amplitudes
        t_ijkabc (array): Extrapolated triples amplitudes

    """


    # Checks if there are multiple vectors inside t_vectors
    is_multiple = isinstance(t_vectors, tuple) and isinstance(t_vectors[0], (list, np.ndarray))

    # If there are multiple vectors (t_ijab_vector, t_ia_vector) then the first item in each vector should be deleted
    if is_multiple and len(t_vectors[0]) > calculation.max_DIIS_matrices:

        del DIIS_error_vector[0]

        for vec in t_vectors: del vec[0]

    # If there is one vector, the first item in that vector should be deleted
    elif len(t_vectors) > calculation.max_DIIS_matrices:

        del DIIS_error_vector[0]
        del t_vectors[0]

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

        if is_multiple: 
            
            # If there are multiple, extrapolate t_ijab and t_ia, and t_ijkabc if there are all three vectors
            t_ijab = np.tensordot(coeffs, t_vectors[0], axes=(0, 0))
            t_ia = np.tensordot(coeffs, t_vectors[1], axes=(0, 0))
            t_ijkabc = np.tensordot(coeffs, t_vectors[2], axes=(0, 0)) if len(t_vectors) == 3 else None

        else:

            # If there's only one vector, it will be t_ijab so extrapolate this and set the other arrays to None
            t_ijab = np.tensordot(coeffs, t_vectors, axes=(0, 0))

            t_ia = t_ijkabc = None


    except np.linalg.LinAlgError:

        # Clears all the vectors in t_vectors
        [vec.clear() for vec in t_vectors] if is_multiple else t_vectors.clear()

        DIIS_error_vector.clear()

        t_ijab = t_ia = t_ijkabc = None

        log("   (Resetting DIIS)", calculation, 1, end="", silent=silent)


    return t_ijab, t_ia, t_ijkabc










def calculate_coupled_cluster_linearised_density(t_ia, t_ijab, n_SO, C_spin_block, n_occ, o, v, calculation, silent=False):

    """

    Calculates the coupled cluster linearised one-particle reduced density matrix.

    Args:

        t_ia (array, optional): Singles amplitudes
        t_ijab (array): Doubles amplitudes
        n_SO (int): Number of spin orbitals
        C_spin_block (array): Spin-blocked molecular orbitals
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

    # Correlated part of density matrix from squared connected doubles
    P_CC = mp.build_t_amplitude_density_contribution(n_SO, t_ijab, o, v)


    # Linearised coupled-cluster density, only includes up to double excitations I think
    P_CC[o, v] += t_ia + np.einsum("ijab,jb->ia", t_ijab, t_ia, optimize=True)
    P_CC[v, o] = P_CC[o, v].T

    P_CC[v, v] += np.einsum("ia,ib->ab", t_ia, t_ia, optimize=True)
    P_CC[o, o] -= np.einsum("ia,ja->ij", t_ia, t_ia, optimize=True)


    # Builds reference density matrix in SO basis, diagonal in ones
    P_ref = np.zeros((n_SO, n_SO))
    P_ref[slice(0, n_occ), slice(0, n_occ)] = np.identity(n_occ)

    P = P_ref + P_CC

    # Transforms the density matrix from spin-orbital to atomic orbital basis
    P, P_alpha, P_beta = ci.transform_P_SO_to_AO(P, C_spin_block, n_SO)

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
    
    # Calculates the T1 diagnostic
    T1_diagnostic = t_ia_norm / np.sqrt(n_occ)

    log(f"\n  Norm of singles amplitudes:         {t_ia_norm:13.10f}", calculation, 1, silent=silent)
    log(f"  Value of T1 diagnostic:             {T1_diagnostic:13.10f}", calculation, 1, silent=silent)


    return











def find_largest_amplitudes(t_ijab, t_ia, spin_orbital_labels_sorted, calculation, molecule, silent=False):

    """
    
    Finds, formats and prints the largest single and double excitation amplitudes from a converged coupled cluster calculation.

    Args:
        t_ijab (array): Doubles amplitudes
        t_ia (array): Singles amplitudes
        spin_orbital_labels_sorted (list): Sorted list of spin orbital labels
        calculation (Calculation): Calculation object
        molecule (Molecule): Molecule object
        silent (bool, optional): Cancel logging
    
    """


    t_ijab = np.pad(t_ijab, ((molecule.n_core_spin_orbitals, 0), (molecule.n_core_spin_orbitals,0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    t_ia = np.pad(t_ia, ((molecule.n_core_spin_orbitals,0), (0, 0)), mode='constant', constant_values=0)

    log("\n  Searching for largest amplitudes...        ", calculation, 2, end="", silent=silent); sys.stdout.flush()

    n_occ = molecule.n_occ

    # Flattens and absolutises the arrays
    t_ijab_abs = np.abs(t_ijab).ravel()
    t_ia_abs = np.abs(t_ia).ravel()
    top_idx_ijab = np.argpartition(-t_ijab_abs, t_ijab_abs.size - 1)[:t_ijab_abs.size]
    top_idx_ia = np.argpartition(-t_ia_abs, t_ia_abs.size - 1)[:t_ia_abs.size]

    top_idx_ijab = top_idx_ijab[np.argsort(-t_ijab_abs[top_idx_ijab])]
    top_idx_ia = top_idx_ia[np.argsort(-t_ia_abs[top_idx_ia])]

    idx_arrays_ijab = np.unravel_index(top_idx_ijab, t_ijab.shape)
    idx_arrays_ia = np.unravel_index(top_idx_ia, t_ia.shape)

    # This is the arrays ordered from largest to smallest values
    top_t_ijab = t_ijab[idx_arrays_ijab]  
    top_t_ia = t_ia[idx_arrays_ia]  



    def format_indices(idxs, spin_orbital_labels_sorted, n_occ):
        
        """

        Formats some indices, into spin-orbital labelling.

        Args:
            idxs (tuple): Indices
            spin_orbital_labels_sorted (list): List of spin orbital labels, sorted
            n_occ (int): Number of occupied orbitals

        Returns:
            indices_formatted (list): List of formatted indices, ie 1a, 4b, etc. instead of 3, 5
        
        """

        indices_formatted = []

        for k, idx in enumerate(idxs):
            
            # Length of 2 means t_ia, so different counting
            if len(idxs) == 2:

                idx += n_occ if k > 0 else 0

            else:

                idx += n_occ if k > 1 else 0

            indices_formatted.append(spin_orbital_labels_sorted[idx])

        return indices_formatted






    def screen_indices(idx_arrays_ijab, idx_arrays_ia, top_t_ijab, top_t_ia, spin_orbital_labels_sorted, n_occ):

        """

        Formats and screens out indices of forbidden transitions.

        Args:
            idx_arrays_ijab (array): Indices of t_ijab arrays
            idx_arrays_ia (array): Indices of t_ia arrays
            top_t_ijab (array): Sorted list of largest t_ijab array elements
            top_t_ia (array): Sorted list of largest t_ia array elements
            spin_orbital_labels_sorted (list): List of spin orbital labels, sorted
            n_occ (int): Number of occupied orbitals

        Returns:
            screened_rows_ijab (array): Screened and formatted indices of ijab shape
            screened_rows_ia (array): Screened and formatted indices of ia shape
            screened_vals_ijab (array): Values of screened and formatted indices of ijab shape
            screened_vals_ia (array): Values of screened and formatted indices of ia shape
        
        """

        screened_rows_ijab, screened_rows_ia, screened_vals_ijab, screened_vals_ia = [], [], [], [] 

        for i, j, a, b, val in zip(*idx_arrays_ijab, top_t_ijab):

            row = format_indices((i, j, a, b), spin_orbital_labels_sorted, n_occ)

            # Only allows alpha-alpha or alpha-beta transitions
            if row[0][-1]  == row[2][-1] and row[1][-1] == row[3][-1]:
                
                screened_rows_ijab.append(row)
                screened_vals_ijab.append(val)


        for i, a, val in zip(*idx_arrays_ia, top_t_ia):

            row = format_indices((i, a), spin_orbital_labels_sorted, n_occ)

            # Only allows alpha-alpha or alpha-beta transitions
            if row[0][-1] == row[1][-1]:
                
                screened_rows_ia.append(row)
                screened_vals_ia.append(val)

   
        return screened_rows_ijab, screened_rows_ia, screened_vals_ijab, screened_vals_ia





    def find_unique_indices(screened_rows_ijab, screened_rows_ia, screened_vals_ijab, screened_vals_ia):
        
        """

        Finds the unique indices, from the screened indices.

        Args:
            screened_rows_ijab (array): Screened t_ijab indices
            screened_rows_ia (array): Screened t_ia indices
            screened_vals_ijab (array): Values from t_ijab indices
            screened_vals_ia (array): Values from t_ia indices

        Returns:
            unique_rows_ijab (array): Unique indices of ijab shape
            unique_rows_ia (array): Unique indices of ia shape
            unique_vals_ijab (array): Values of unique indices of ijab shape
            unique_vals_ia (array): Values of unique indices of ia shape
        
        """

        unique_ijab, unique_ia = {}, {}
        
        for row, val in zip(screened_rows_ijab, screened_vals_ijab):
            
            key = (row[0], row[2], row[1], row[3]) 
            
            excitation_1 = (row[0], row[2]) 
            excitation_2 = (row[1], row[3]) 
            
            # Makes sure the two independent transitions in t_ijab are only included once
            key = tuple(sorted([excitation_1, excitation_2]))
            
            if key not in unique_ijab: 
                
                unique_ijab[key] = (row, val) 
                

        for row, val in zip(screened_rows_ia, screened_vals_ia): 
            
            # Makes sure the two transitions in t_ia are only included once
            key = tuple(sorted(row)) 
                    
            if key not in unique_ia: 
                
                unique_ia[key] = (row, val) 
                

        # Lists of unique rows and values for each tensor
        unique_rows_ijab = [row for row, _ in unique_ijab.values()] 
        unique_vals_ijab = [val for _, val in unique_ijab.values()] 

        unique_rows_ia = [row for row, _ in unique_ia.values()] 
        unique_vals_ia = [val for _, val in unique_ia.values()]


        return unique_rows_ijab, unique_rows_ia, unique_vals_ijab, unique_vals_ia






    def prepare_printing_indices(unique_rows_ijab, unique_rows_ia, unique_vals_ijab, unique_vals_ia):
        
        """

        Makes the indices suitable for logging.

        Args:
            unique_rows_ijab (array): Unique indices of ijab shape
            unique_rows_ia (array): Unique indices of ia shape
            unique_vals_ijab (array): Values of unique indices of ijab shape
            unique_vals_ia (array): Values of unique indices of ia shape

        Returns:
            printing_indices_ijab (array): Indices of shape ijab for printing
            printing_values_ijab (array): Values of indices of shape ijab for printing
            printing_indices_ia (array): Indices of shape ia for printing
            printing_values_ia (array): Values of indices of shape ia for printing
        
        """

        printing_indices_ijab, printing_values_ijab, printing_indices_ia, printing_values_ia  = [], [], [], []

        for row, val in zip(unique_rows_ijab, unique_vals_ijab):

            a, b, c, d = row[0], row[2], row[1], row[3]

            # Makes sure printing always has excitations of alpha orbitals on the left, beta on the right
            if row[0][1] == "b":

                a, b, c, d = row[1], row[3], row[0], row[2]

            space = " "
            if len(a) == 3 or len(b) == 3: space = ""

            entry = f"{a} -> {b},  {space} {c} -> {d}"

            printing_indices_ijab.append(entry)
            printing_values_ijab.append(val)


        for row, val in zip(unique_rows_ia, unique_vals_ia):

            space = " "
            if len(row[0]) == 3 or len(row[1]) == 3: space = ""

            # Adds stars to opaque out the unused spin, keeps alpha on the left, beta on the right
            entry = f"{row[0]} -> {row[1]}, {space}  ********"
            if entry[1] == "b":  entry = f"********,  {space} {row[0]} -> {row[1]}"

            printing_indices_ia.append(entry)
            printing_values_ia.append(val)
        

        return printing_indices_ijab, printing_values_ijab, printing_indices_ia, printing_values_ia
        

    # Screens out forbidden transitions, formats indices
    screened_rows_ijab, screened_rows_ia, screened_vals_ijab, screened_vals_ia = screen_indices(idx_arrays_ijab, idx_arrays_ia, top_t_ijab, top_t_ia, spin_orbital_labels_sorted, n_occ)

    # Removes non-unique indices
    unique_rows_ijab, unique_rows_ia, unique_vals_ijab, unique_vals_ia = find_unique_indices(screened_rows_ijab, screened_rows_ia, screened_vals_ijab, screened_vals_ia)

    # Adds padding for printing, depending on length of indices and keeps alpha spins and beta spins together, separately
    printing_indices_ijab, printing_values_ijab, printing_indices_ia, printing_values_ia  = prepare_printing_indices(unique_rows_ijab, unique_rows_ia, unique_vals_ijab, unique_vals_ia)

    # Builds final indices and values list
    final_indices = printing_indices_ijab + printing_indices_ia
    final_values = np.array(printing_values_ijab + printing_values_ia)

    # Reorders them all together
    order = np.argsort(-np.abs(final_values))
    final_indices = [final_indices[i] for i in order]
    final_values = final_values[order].tolist()

    log(f"[Done]", calculation, 2, silent=silent)

    log("\n  Largest amplitudes:\n", calculation, 2, silent=silent)

    # Print the 10 largest doubles and singles amplitudes, with their corresponding excitations, formatted correctly
    for counter, indices in enumerate(final_indices):

        if counter < 10 and np.abs(final_values[counter]) > 0.0000001:

            log(f"    {indices:<22.22}   :   {np.abs(final_values[counter]):9.6f}", calculation, 2, silent=silent)


    return












def calculate_LCCD_energy(g, e_ijab, t_ijab, o, v, calculation, silent=False):

    """ 
    
    Calculates the linearised coupled cluster doubles energy.

    Args:
        g (array): Spin orbital ERI tensor
        e_ijab (array): Doubles epsilon tensor
        t_ijab (array): Guess doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_LCCD (float): Linearised CCD energy
        t_ijab (array): Converged doubles amplitudes

    """

    E_LCCD = 0.0

    CC_max_iter = calculation.CC_max_iter

    DIIS_error_vector = []
    t_ijab_vector = []


    log_spacer(calculation, silent=silent, start="\n")
    log("          Linearised CCD Energy and Density ", calculation, 1, silent=silent, colour="white")


    coupled_cluster_initial_print(t_ijab, g, "LCCD", o, v, calculation, silent=silent)


    for step in range(1, CC_max_iter + 1):

        # Stores the old energy to calculate the change in energy
        E_old = E_LCCD
        t_ijab_old = t_ijab.copy()

        # Updates doubles amplitudes
        t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
        t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)
    
        t_ijab = e_ijab * t_ijab_temporary

        # Calculates the LCCD correlation energy
        E_LCCD = mp.calculate_t_amplitude_energy(t_ijab, g[o, o, v, v])

        if E_LCCD > 1000 or not np.isfinite(t_ijab).all():

            error("Non-finite encountered in LCCD iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy
        delta_E = E_LCCD - E_old

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv: break

        elif step >= CC_max_iter: error("The LCCD iterations failed to converge! Try increasing the maximum iterations?")

        # Formats output lines nicely
        log(f"  {step:3.0f}           {E_LCCD:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # Update amplitudes with DIIS
        t_ijab_residual = (t_ijab - t_ijab_old).ravel()

        t_ijab_vector.append(t_ijab.copy())
        DIIS_error_vector.append(t_ijab_residual)

        if step > 2 and calculation.DIIS: 

            t_ijab_DIIS, _, _ = update_DIIS(t_ijab_vector, DIIS_error_vector, calculation, silent=silent)

            t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab

        # Applies damping to amplitudes
        t_ijab, _, _ = apply_damping(t_ijab, t_ijab_old, calculation)


    log_spacer(calculation, silent=silent)

    log(f"\n  LCCD correlation energy:            {E_LCCD:13.10f}", calculation, 1, silent=silent)


    return E_LCCD, t_ijab







def calculate_CCD_energy(g, e_ijab, t_ijab, o, v, calculation, silent=False):

    """ 
    
    Calculates the coupled cluster doubles energy.

    Args:
        g (array): Spin orbital ERI tensor
        e_ijab (array): Doubles epsilon tensor
        t_ijab (array): Guess doubles amplitudes
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_CCD (float): CCD energy
        array (float): Converged doubles amplitudes

    """

    E_CCD = 0.0

    CC_max_iter = calculation.CC_max_iter

    DIIS_error_vector = []
    t_ijab_vector = []

    log_spacer(calculation, silent=silent, start="\n")
    log("                CCD Energy and Density ", calculation, 1, silent=silent, colour="white")


    coupled_cluster_initial_print(t_ijab, g, "CCD", o, v, calculation, silent=silent)


    for step in range(1, CC_max_iter + 1):

        # Stores the old energy to calculate the change in energy
        E_old = E_CCD
        t_ijab_old = t_ijab.copy()

        # Calculates contribution from LCCD
        t_ijab_temporary = g[o, o, v, v] + (1 / 2) * np.einsum("cdab,ijcd->ijab", g[v, v, v, v], t_ijab, optimize=True) + (1 / 2) * np.einsum("ijkl,klab->ijab", g[o, o, o, o], t_ijab, optimize=True) 
        t_ijab_temporary += permute(permute(np.einsum("icak,jkbc->ijab", g[o, v, v, o], t_ijab, optimize=True), 2, 3), 0, 1)
    
        # Calculates contribution from full CCD
        t_ijab_temporary += - (1 / 2) * permute(np.einsum("cdkl,ijac,klbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 2, 3) - (1 / 2) * permute(np.einsum("cdkl,ikab,jlcd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)
        t_ijab_temporary += (1 / 4) * np.einsum("cdkl,ijcd,klab->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True)
        t_ijab_temporary += permute(np.einsum("cdkl,ikac,jlbd->ijab", g[v, v, o, o], t_ijab, t_ijab, optimize=True), 0, 1)

        t_ijab = e_ijab * t_ijab_temporary

        # Calculates the CCD correlation energy
        E_CCD = mp.calculate_t_amplitude_energy(t_ijab, g[o, o, v, v])

        if E_CCD > 1000 or not np.isfinite(t_ijab).all():

            error("Non-finite encountered in CCD iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy
        delta_E = E_CCD - E_old

        # Formats output lines nicely
        log(f"  {step:3.0f}           {E_CCD:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv: break

        elif step >= CC_max_iter: error("The CCD iterations failed to converge! Try increasing the maximum iterations?")


        # Update amplitudes with DIIS
        t_ijab_residual = (t_ijab - t_ijab_old).ravel()

        t_ijab_vector.append(t_ijab.copy())
        DIIS_error_vector.append(t_ijab_residual)

        if step > 2 and calculation.DIIS: 

            t_ijab_DIIS, _, _ = update_DIIS(t_ijab_vector, DIIS_error_vector, calculation, silent=silent)

            t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab

        # Applies damping to amplitudes
        t_ijab, _, _ = apply_damping(t_ijab, t_ijab_old, calculation)
    

    log_spacer(calculation, silent=silent)

    log(f"\n  CCD correlation energy:             {E_CCD:.10f}", calculation, 1, silent=silent)


    return E_CCD, t_ijab







def calculate_LCCSD_energy(g, e_ia, e_ijab, t_ia, t_ijab, F, o, v, calculation, silent=False):

    """ 
    
    Calculates the linearised coupled cluster singles and doubles energy.

    Args:
        g (array): Spin orbital ERI tensor
        e_ia (array): Singles epsilon tensor
        e_ijab (array): Doubles epsilon tensor
        t_ia (array): Guess singles amplitudes
        t_ijab (array): Guess doubles amplitudes
        F (array): Spin orbital Fock matrix
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_LCCSD (float): LCCSD energy
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes

    """

    E_LCCSD = 0.0

    CC_max_iter = calculation.CC_max_iter

    DIIS_error_vector = []
    t_ijab_vector = []
    t_ia_vector = []

    log_spacer(calculation, silent=silent, start="\n")
    log("         Linearised CCSD Energy and Density ", calculation, 1, silent=silent, colour="white")


    coupled_cluster_initial_print(t_ijab, g, "LCCSD", o, v, calculation, silent=silent)


    for step in range(1, CC_max_iter + 1):

        # Stores the old energy to calculate the change in energy
        E_old = E_LCCSD
        t_ijab_old = t_ijab.copy()
        t_ia_old = t_ia.copy()

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

        # Separate components of energy, singles terms should be zero for HF reference
        E_LCCSD, E_LCCSD_singles, E_LCCSD_connected_doubles, _ = calculate_coupled_cluster_energy(o, v, g, t_ijab)

        if E_LCCSD > 1000 or any(not np.isfinite(x).all() for x in (t_ia, t_ijab)):

            error("Non-finite encountered in LCCSD iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy
        delta_E = E_LCCSD - E_old

        # Formats output lines nicely
        log(f"  {step:3.0f}           {E_LCCSD:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv and np.linalg.norm(t_ia - t_ia_old) < calculation.amp_conv: break

        elif step >= CC_max_iter: error("The LCCSD iterations failed to converge! Try increasing the maximum iterations?")

        # Update amplitudes with DIIS
        t_ijab_residual = (t_ijab - t_ijab_old).ravel()
        t_ia_residual = (t_ia - t_ia_old).ravel()

        t_ijab_vector.append(t_ijab.copy())
        t_ia_vector.append(t_ia.copy())

        DIIS_error_vector.append(np.concatenate((t_ijab_residual, t_ia_residual)))

        if step > 2 and calculation.DIIS: 

            t_ijab_DIIS, t_ia_DIIS, _ = update_DIIS((t_ijab_vector, t_ia_vector), DIIS_error_vector, calculation, silent=silent)

            t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab
            t_ia = t_ia_DIIS if t_ia_DIIS is not None else t_ia

        # Applies damping to amplitudes
        t_ijab, t_ia, _ =  apply_damping(t_ijab, t_ijab_old, calculation, t_ia=t_ia, t_ia_old=t_ia_old)


    log_spacer(calculation, silent=silent)

    log(f"\n  Singles contribution:               {E_LCCSD_singles:13.10f}", calculation, 1, silent=silent)
    log(f"  Connected doubles contribution:     {E_LCCSD_connected_doubles:13.10f}", calculation, 1, silent=silent)

    log(f"\n  LCCSD correlation energy:           {E_LCCSD:13.10f}", calculation, 1, silent=silent)


    return E_LCCSD, t_ia, t_ijab










def calculate_CCSD_energy(g, e_ia, e_ijab, t_ia, t_ijab, F, o, v, calculation, silent=False):

    """ 
    
    Calculates the coupled cluster singles and doubles energy.

    Args:
        g (array): Spin orbital ERI tensor
        e_ia (array): Singles epsilon tensor
        e_ijab (array): Doubles epsilon tensor
        t_ia (array): Guess singles amplitudes
        t_ijab (array): Guess doubles amplitudes
        F (array): Spin orbital Fock matrix
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_CCSD (float): CCSD energy
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes

    """

    E_CCSD = 0.0

    CC_max_iter = calculation.CC_max_iter

    DIIS_error_vector = []
    t_ijab_vector = []
    t_ia_vector = []

    kronecker_delta = np.eye(F.shape[1])


    log_spacer(calculation, silent=silent, start="\n")
    log("               CCSD Energy and Density ", calculation, 1, silent=silent, colour="white")


    coupled_cluster_initial_print(t_ijab, g, "CCSD", o, v, calculation, silent=silent)


    for step in range(1, CC_max_iter + 1):

        t_ijab_old = t_ijab.copy()
        t_ia_old = t_ia.copy()

        E_old = E_CCSD

        # Build tau tensors, all equations from Stanton paper on DPD coupled cluster, referenced by Crawford tutorials
        tau_tilde_ijab = t_ijab + (1 / 2) * (np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True))
        tau_ijab = t_ijab + np.einsum("ia,jb->ijab", t_ia, t_ia, optimize=True) - np.einsum("ib,ja->ijab", t_ia, t_ia, optimize=True)


        # Builds curly F intermediates
        F_ae = F[v, v] - np.einsum("ae,ae->ae", kronecker_delta[v, v], F[v, v], optimize=True) - (1 / 2) * np.einsum("me,ma->ae", F[o, v], t_ia, optimize=True) + np.einsum("mf,mafe->ae", t_ia, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnaf,mnef->ae", tau_tilde_ijab, g[o, o, v, v], optimize=True)
        F_mi = F[o, o] - np.einsum("mi,mi->mi", kronecker_delta[o, o], F[o, o], optimize=True) + (1 / 2) * np.einsum("ie,me->mi", t_ia, F[o, v], optimize=True) + np.einsum("ne,mnie->mi", t_ia, g[o, o, o, v], optimize=True) + (1 / 2) * np.einsum("inef,mnef->mi", tau_tilde_ijab, g[o, o, v, v], optimize=True)
        F_me = F[o, v] + np.einsum("nf,mnef->me", t_ia, g[o, o, v, v], optimize=True) 
        

        # Builds curly W intermediates
        W_mnij = g[o, o, o, o] + permute(np.einsum("je,mnie->mnij", t_ia, g[o, o, o, v], optimize=True), 2, 3) + (1 / 4) * np.einsum("ijef,mnef->mnij", tau_ijab, g[o, o, v, v], optimize=True)
        W_abef = g[v, v, v, v] - permute(np.einsum("mb,amef->abef", t_ia, g[v, o, v, v], optimize=True), 0, 1) + (1 / 4) * np.einsum("mnab,mnef->abef", tau_ijab, g[o, o, v, v], optimize=True)
        W_mbej = g[o, v, v, o] + np.einsum("jf,mbef->mbej", t_ia, g[o, v, v, v], optimize=True) - np.einsum("nb,mnej->mbej", t_ia, g[o, o, v, o], optimize=True) - np.einsum("jnfb,mnef->mbej", (1 / 2) * t_ijab + np.einsum("jf,nb->jnfb", t_ia, t_ia, optimize=True), g[o, o, v, v], optimize=True)


        # Builds t_ia tensor from intermediates
        t_ia_temporary = F[o, v] + np.einsum("ie,ae->ia", t_ia, F_ae, optimize=True) - np.einsum("ma,mi->ia", t_ia, F_mi, optimize=True) 
        t_ia_temporary += np.einsum("imae,me->ia", t_ijab, F_me, optimize=True) - np.einsum("nf,naif->ia", t_ia, g[o, v, o, v], optimize=True) - (1 / 2) * np.einsum("imef,maef->ia", t_ijab, g[o, v, v, v], optimize=True) - (1 / 2) * np.einsum("mnae,nmei->ia", t_ijab, g[o, o, v, o], optimize=True)

        
        # Builds t_ijab tensor from intermediates, pairs of terms from Stanton, specifying type is necessary to keep addition working

        t_ijab_temporary = g[o, o, v, v] + permute(np.einsum("ijae,be->ijab", t_ijab, F_ae - (1 / 2) * np.einsum("mb,me->be", t_ia, F_me,optimize=True), optimize=True), 2, 3) - permute(np.einsum("imab,mj->ijab", t_ijab, F_mi + (1 / 2) * np.einsum("je,me->mj", t_ia, F_me, optimize=True),optimize=True), 0, 1)
        t_ijab_temporary += (1 / 2) * np.einsum("mnab,mnij->ijab", tau_ijab, W_mnij, optimize=True) + (1 / 2) * np.einsum("ijef,abef->ijab", tau_ijab, W_abef, optimize=True)
        t_ijab_temporary += permute(permute(np.einsum("ijmabe->ijab", np.einsum("imae,mbej->ijmabe", t_ijab, W_mbej, optimize=True) - np.einsum("ie,ma,mbej->ijmabe", t_ia, t_ia, g[o, v, v, o], optimize=True), optimize=True), 2, 3), 0, 1)
        t_ijab_temporary += permute(np.einsum("ie,abej->ijab", t_ia, g[v, v, v, o], optimize=True), 0, 1) - permute(np.einsum("ma,mbij->ijab", t_ia, g[o, v, o, o], optimize=True), 2, 3)

        t_ia = e_ia * t_ia_temporary
        t_ijab = e_ijab * t_ijab_temporary

        # Finds the different components of the energy, singles should be zero for HF reference
        E_CCSD, E_CCSD_singles, E_CCSD_connected_doubles, E_CCSD_disconnected_doubles = calculate_coupled_cluster_energy(o, v, g, t_ijab, t_ia=t_ia, F=F)

        if E_CCSD > 1000 or any(not np.isfinite(x).all() for x in (t_ia, t_ijab)):

            error("Non-finite encountered in CCSD iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy
        delta_E = E_CCSD - E_old

        log(f"  {step:3.0f}           {E_CCSD:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv and np.linalg.norm(t_ia - t_ia_old) < calculation.amp_conv: break

        elif step >= CC_max_iter: error("The CCSD iterations failed to converge! Try increasing the maximum iterations?")

        # Update amplitudes with DIIS
        t_ijab_residual = (t_ijab - t_ijab_old).ravel()
        t_ia_residual = (t_ia - t_ia_old).ravel()

        t_ijab_vector.append(t_ijab.copy())
        t_ia_vector.append(t_ia.copy())

        DIIS_error_vector.append(np.concatenate((t_ijab_residual, t_ia_residual)))

        if step > 2 and calculation.DIIS: 

            t_ijab_DIIS, t_ia_DIIS, _ = update_DIIS((t_ijab_vector, t_ia_vector), DIIS_error_vector, calculation, silent=silent)

            t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab
            t_ia = t_ia_DIIS if t_ia_DIIS is not None else t_ia

        # Applies damping to amplitudes
        t_ijab, t_ia, _ =  apply_damping(t_ijab, t_ijab_old, calculation, t_ia=t_ia, t_ia_old=t_ia_old)


    log_spacer(calculation, silent=silent)

    log(f"\n  Singles contribution:               {E_CCSD_singles:13.10f}", calculation, 1, silent=silent)
    log(f"  Connected doubles contribution:     {E_CCSD_connected_doubles:13.10f}", calculation, 1, silent=silent)
    log(f"  Disconnected doubles contribution:  {E_CCSD_disconnected_doubles:13.10f}", calculation, 1, silent=silent)

    log(f"\n  CCSD correlation energy:            {E_CCSD:.10f}", calculation, 1, silent=silent)


    return E_CCSD, t_ia, t_ijab









def calculate_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, calculation, silent=False):

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

    log_spacer(calculation, silent=silent, start="\n")
    log("                   CCSD(T) Energy  ", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)


    def permute_three_indices(array_ijab, idx1, idx2, idx3):
        
        # Three-index permutation operator per Crawford

        return array_ijab - array_ijab.swapaxes(idx1, idx2) - array_ijab.swapaxes(idx1, idx3)


    log("  Forming disconnected triples amplitudes... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    # Temporary disconnected (d_ijkabc) and connected (c_ijkabc) triples tensors before permutation, from Crawford
    d_ijkabc = np.einsum("ia,jkbc->ijkabc", t_ia, g[o, o, v, v], optimize=True)
    t_ijkabc_d = np.einsum("ijkabc,ijkabc->ijkabc", e_ijkabc, permute_three_indices(permute_three_indices(d_ijkabc, 3, 4, 5), 0, 1, 2), optimize=True)

    log(f"[Done]", calculation, 1, silent=silent)

    log("  Forming connected triples amplitudes...    ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    c_ijkabc = np.einsum("jkae,eibc->ijkabc", t_ijab, g[v, o, v, v], optimize=True) - np.einsum("imbc,majk->ijkabc", t_ijab, g[o, v, o, o], optimize=True)
    t_ijkabc_c = np.einsum("ijkabc,ijkabc->ijkabc", e_ijkabc, permute_three_indices(permute_three_indices(c_ijkabc, 3, 4, 5), 0, 1, 2), optimize=True)

    log(f"[Done]", calculation, 1, silent=silent)

    log("\n  Calculating CCSD(T) correlation energy...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    # Final contraction for the CCSD(T) energy using the connected and disconnected approximate triples amplitudes
    E_CCSD_T = (1 / 36) * np.einsum("ijkabc,ijkabc->", t_ijkabc_c / e_ijkabc, t_ijkabc_c + t_ijkabc_d, optimize=True)

    log(f"[Done]\n\n  CCSD(T) correlation energy:         {E_CCSD_T:13.10f}", calculation, 1, silent=silent) 


    return E_CCSD_T










def calculate_CCSDT_energy(g, e_ia, e_ijab, e_ijkabc, t_ia, t_ijab, t_ijkabc, F, o, v, calculation, silent=False):
    
    """ 
    
    Calculates the coupled cluster singles, doubles and triples energy.

    Args:
        g (array): Spin orbital ERI tensor
        e_ia (array): Singles epsilon tensor
        e_ijab (array): Doubles epsilon tensor
        e_ijkabc (array): Triples epsilon tensor
        t_ia (array): Guess singles amplitudes
        t_ijab (array): Guess doubles amplitudes
        t_ijkabc (array): Guess triples amplitudes
        F (array): Spin orbital Fock matrix
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    
    Returns:
        E_CCSD (float): CCSD energy
        t_ia (array): Converged singles amplitudes
        t_ijab (array): Converged doubles amplitudes
        t_ijkabc (array): Converged triples amplitudes

    """

    E_CCSDT = 0.0

    CC_max_iter = calculation.CC_max_iter

    DIIS_error_vector = []
    t_ijkabc_vector = []
    t_ijab_vector = []
    t_ia_vector = []


    log_spacer(calculation, silent=silent, start="\n")
    log("              CCSDT Energy and Density ", calculation, 1, silent=silent, colour="white")


    coupled_cluster_initial_print(t_ijab, g, "CCSDT", o, v, calculation, silent=silent)


    for step in range(1, CC_max_iter + 1):

        E_old = E_CCSDT

        t_ia_old = t_ia.copy()
        t_ijab_old = t_ijab.copy()
        t_ijkabc_old = t_ijkabc.copy()


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

    
        t_ia += e_ia * t_ia_temporary 
        t_ijab += e_ijab * t_ijab_temporary 
        t_ijkabc += e_ijkabc * t_ijkabc_temporary 


        # Finds the different components of the energy, singles should be zero for HF reference
        E_CCSDT, E_CCSDT_singles, E_CCSDT_connected_doubles, E_CCSDT_disconnected_doubles = calculate_coupled_cluster_energy(o, v, g, t_ijab, t_ia=t_ia, F=F)
        
        if E_CCSDT > 1000 or any(not np.isfinite(x).all() for x in (t_ia, t_ijab, t_ijkabc)):

            error("Non-finite encountered in CCSDT iteration. Try stronger damping with the CCDAMP keyword?.")

        # Calculates the change in energy
        delta_E = E_CCSDT - E_old

        log(f"  {step:3.0f}           {E_CCSDT:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)

        # If convergence criteria has been reached, exit loop
        if abs(delta_E) < calculation.CC_conv and np.linalg.norm(t_ijab - t_ijab_old) < calculation.amp_conv and np.linalg.norm(t_ia - t_ia_old) < calculation.amp_conv: break

        elif step >= CC_max_iter: error("The CCSDT iterations failed to converge! Try increasing the maximum iterations?")

        # Update amplitudes with DIIS
        t_ijkabc_residual = (t_ijkabc - t_ijkabc_old).ravel()
        t_ijab_residual = (t_ijab - t_ijab_old).ravel()
        t_ia_residual = (t_ia - t_ia_old).ravel()

        t_ijkabc_vector.append(t_ijkabc.copy())
        t_ijab_vector.append(t_ijab.copy())
        t_ia_vector.append(t_ia.copy())

        DIIS_error_vector.append(np.concatenate((t_ijab_residual, t_ia_residual, t_ijkabc_residual)))

        if step > 2 and calculation.DIIS: 

            t_ijab_DIIS, t_ia_DIIS, t_ijkabc_DIIS = update_DIIS((t_ijab_vector, t_ia_vector, t_ijkabc_vector), DIIS_error_vector, calculation, silent=silent)

            t_ijkabc = t_ijkabc_DIIS if t_ijkabc_DIIS is not None else t_ijkabc
            t_ijab = t_ijab_DIIS if t_ijab_DIIS is not None else t_ijab
            t_ia = t_ia_DIIS if t_ia_DIIS is not None else t_ia

        # Applies damping to amplitudes
        t_ijab, t_ia, t_ijkabc =  apply_damping(t_ijab, t_ijab_old, calculation, t_ia=t_ia, t_ia_old=t_ia_old, t_ijkabc=t_ijkabc, t_ijkabc_old=t_ijkabc_old)


    log_spacer(calculation, silent=silent)

    log(f"\n  Singles contribution:               {E_CCSDT_singles:13.10f}", calculation, 1, silent=silent)
    log(f"  Connected doubles contribution:     {E_CCSDT_connected_doubles:13.10f}", calculation, 1, silent=silent)
    log(f"  Disconnected doubles contribution:  {E_CCSDT_disconnected_doubles:13.10f}", calculation, 1, silent=silent)

    log(f"\n  CCSDT correlation energy:           {E_CCSDT:.10f}", calculation, 1, silent=silent)


    return E_CCSDT, t_ia, t_ijab, t_ijkabc











def calculate_coupled_cluster(method, molecule, SCF_output, ERI_AO, X, H_core, calculation, silent=False):

    """

    Calculates the coupled cluster energy and density.

    Args:
        method (str): Electronic structure method
        molecule (Molecule): Molecule object
        SCF_output (Output): Output object
        ERI_AO (array): Atomic orbital basis ERI tensor
        X (array): Fock transformation matrix
        H_core (array): Core Hamiltonian matrix in AO basis
        calculation (Calculation): Calculation object
        silent (bool, optional): Cancel logging

    Returns:
        E_LCCD (float): LCCD energy
        E_CCD (float): CCD energy
        E_LCCSD (float): LCCSD energy
        E_CCSD (float): CCSD energy
        E_CCSD_T (float): CCSD(T) energy
        E_CCSDT (float): CCSDT energy
        P (array): Linearised coupled cluster density matrix
        P_alpha (array): Alpha linearised coupled cluster density matrix
        P_beta (array): Beta linearised coupled cluster density matrix
 

    """

    E_LCCD, E_CCD, E_LCCSD, E_CCSD, E_CCSD_T, E_CCSDT = (0, 0, 0, 0, 0, 0)

    n_occ = molecule.n_occ
    n_virt = molecule.n_virt

    # Calculates useful quantities for all coupled cluster calculations
    g, C_spin_block, epsilons_sorted, ERI_spin_block, o, v, spin_labels_sorted, spin_orbital_labels_sorted = ci.begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=silent)

    log("\n Preparing arrays for coupled cluster...     ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Spin blocks and transforms the core Hamiltonian
    H_core_spin_block = ci.spin_block_core_Hamiltonian(H_core)
    H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)

    # Combines the spin-orbital core Hamiltonian and ERIs to get the spin-orbital basis Fock matrix
    F = ci.build_spin_orbital_Fock_matrix(H_core_SO, g, slice(0, n_occ))

    # Builds the inverse epsilon tensors
    e_ia = ci.build_singles_epsilons_tensor(epsilons_sorted, o, v)
    e_ijab = ci.build_doubles_epsilons_tensor(epsilons_sorted, epsilons_sorted, o, o, v, v)
    e_ijkabc = ci.build_triples_epsilons_tensor(epsilons_sorted, o, v)

    # Defines the guess t-amplitudes
    t_ia = np.einsum("ia,ia->ia", e_ia, F[o, v], optimize=True)
    t_ijab = ci.build_MP2_t_amplitudes(g[o, o, v, v], e_ijab)
    t_ijkabc = np.zeros_like(e_ijkabc)

    log("[Done]", calculation, 1, silent=silent)


    if "LCCD" in method:

        E_LCCD, t_ijab = calculate_LCCD_energy(g, e_ijab, t_ijab, o, v, calculation, silent=silent)

    elif "CCD" in method:

        E_CCD, t_ijab = calculate_CCD_energy(g, e_ijab, t_ijab, o, v, calculation, silent=silent)

    elif "LCCSD" in method:

        E_LCCSD, t_ia, t_ijab = calculate_LCCSD_energy(g, e_ia, e_ijab, t_ia, t_ijab, F, o, v, calculation, silent=silent)

    elif "CCSD" in method and "CCSDT" not in method:

        E_CCSD, t_ia, t_ijab = calculate_CCSD_energy(g, e_ia, e_ijab, t_ia, t_ijab, F, o, v, calculation, silent=silent)
       
    elif "CCSDT" in method:

        E_CCSDT, t_ia, t_ijab, t_ijkabc = calculate_CCSDT_energy(g, e_ia, e_ijab, e_ijkabc, t_ia, t_ijab, t_ijkabc, F, o, v, calculation, silent=silent)


    # Determines and prints the T1 diagnostic and norm of the singles
    calculate_T1_diagnostic(molecule, t_ia, spin_labels_sorted, n_occ, molecule.n_alpha, molecule.n_beta, calculation, silent=silent)

    # Determines and prints the largest amplitudes
    find_largest_amplitudes(t_ijab, t_ia, spin_orbital_labels_sorted, calculation, molecule, silent=silent)

    # Calculates the unrelaxed density matrix in the AO basis
    P, P_alpha, P_beta = calculate_coupled_cluster_linearised_density(t_ia, t_ijab, molecule.n_SO, C_spin_block, n_occ, o, v, calculation, silent=silent)
    
    # If NATORBS is used, calculate and print the natural orbitals
    if calculation.natural_orbitals: mp.calculate_natural_orbitals(P, X, calculation, silent=silent)


    if "CCSD[T]" in method:

        E_CCSD_T = calculate_CCSD_T_energy(g, e_ijkabc, t_ia, t_ijab, o, v, calculation, silent=silent)


    log_spacer(calculation, silent=silent)


    return E_LCCD, E_CCD, E_LCCSD, E_CCSD, E_CCSD_T, E_CCSDT, P, P_alpha, P_beta



