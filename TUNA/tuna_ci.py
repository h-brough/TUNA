import numpy as np
from tuna_util import *



def spin_block_core_Hamiltonian(H_core):

    """
    
    Spin blocks core Hamiltonian.

    Args:  
        H_core (array): Core Hamiltonian in AO basis
    
    Returns:
        H_core_spin_block (array): Spin blocked core Hamiltonian in AO basis
    
    """

    H_core_spin_block = np.kron(np.eye(2), H_core)

    return H_core_spin_block





def build_spin_orbital_Fock_matrix(H_core_SO, g, o):

    """
    
    Builds Fock matrix in SO basis.

    Args:  
        H_core_SO (array): Core Hamiltonian in SO basis
        g (array): Antisymmetrised ERI in SO basis
        o (slice): Occupied spin orbitals slice
    
    Returns:
        F_SO (array): Fock matrix in SO basis
    
    """


    F_SO = H_core_SO + np.einsum('piqi->pq', g[:, o, :, o], optimize=True)

    return F_SO





def antisymmetrise_integrals(ERI):

    """

    Antisymmetrises two-electron integrals.

    Args:   
        ERI (array): Electron repulsion integrals

    Returns:
        ERI_ansym (array): Antisymmetrised electron repulsion integrals

    """

    ERI_ansym = ERI - ERI.transpose(0, 1, 3, 2)

    return ERI_ansym






def spin_block_molecular_orbitals(molecular_orbitals_alpha, molecular_orbitals_beta, epsilons):

    """

    Spin blocks alpha and beta molecular orbitals.

    Args:   
        molecular_orbitals_alpha (array): Alpha molecular orbitals in AO basis
        molecular_orbitals_beta (array): Beta molecular orbitals in AO basis
        epsilons (array): Orbital eigenvalues

    Returns:
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis

    """

    C_spin_block = np.block([[molecular_orbitals_alpha, np.zeros_like(molecular_orbitals_beta)], [np.zeros_like(molecular_orbitals_alpha), molecular_orbitals_beta]])
    
    C_spin_block = C_spin_block[:, epsilons.argsort()] 

    return C_spin_block





def transform_ERI_AO_to_SO(ERI_AO, C_1, C_2):

    """

    Transforms electron repulsion integrals from the AO basis to the SO basis.

    Args:   
        ERI_AO (array): Electron repulsion integrals in AO basis
        C_1 (array): Molecular orbitals in AO basis
        C_2 (array): Molecular orbitals in AO basis

    Returns:
        ERI_SO (array): Electron repulsion integrals in SO basis

    """

    ERI_SO = np.einsum("mi,nj,mkln,ka,lb->ijab", C_1, C_2, ERI_AO, C_1, C_2, optimize=True)       

    return ERI_SO








def transform_ERI_AO_to_MO(ERI_AO, C):

    """

    Transforms electron repulsion integrals from the AO basis to the SO basis.

    Args:   
        ERI_AO (array): Electron repulsion integrals in AO basis
        C (array): Molecular orbitals in AO basis

    Returns:
        ERI_MO (array): Electron repulsion integrals in MO basis

    """

    ERI_MO = np.einsum("mi,nj,mnkl,ka,lb->ijab", C, C, ERI_AO, C, C, optimize=True)       

    return ERI_MO








def build_singles_epsilons_tensor(epsilons, o, v):

    """

    Builds inverse epsilon tensor with shape ijab.

    Args:   
        epsilons (array): Orbital eigenvalues
        o (slice): Occupied slice
        v (slice): Virtual slice

    Returns:
        e_ia (array): Inverse epsilons tensor for single excitations with shape ia

    """

    n = np.newaxis

    e_ia = 1 / (epsilons[o, n] - epsilons[n, v])

    return e_ia









def build_doubles_epsilons_tensor(epsilons_1, epsilons_2, o_1, o_2, v_1, v_2):

    """

    Builds inverse epsilon tensor with shape ijab.

    Args:   
        epsilons_1 (array): Orbital eigenvalues
        epsilons_2 (array): Orbital eigenvalues
        o_1 (slice): Occupied slice
        o_2 (slice): Occupied slice
        v_1 (slice): Virtual slice
        v_2 (slice): Virtual slice

    Returns:
        e_ijab (array): Inverse epsilons tensor with shape ijab

    """

    n = np.newaxis

    e_ijab = 1 / (epsilons_1[o_1, n, n, n] + epsilons_2[n, o_2, n, n] - epsilons_1[n, n, v_1, n] - epsilons_2[n, n, n, v_2])

    return e_ijab








def build_triples_epsilons_tensor(epsilons, o, v):

    """

    Builds inverse epsilon tensor with shape ijkabc.

    Args:   
        epsilons (array): Orbital eigenvalues
        o (slice): Occupied slice
        v (slice): Virtual slice

    Returns:
        e_ijkabc (array): Inverse epsilons tensor with shape ijkabc

    """

    n = np.newaxis

    e_ijkabc = 1 / (epsilons[o, n, n, n, n, n] + epsilons[n, o, n, n, n, n] + epsilons[n, n, o, n, n, n] - epsilons[n, n, n, v, n, n] - epsilons[n, n, n, n, v, n] - epsilons[n, n, n, n, n, v])

    return e_ijkabc









def build_MP2_t_amplitudes(g, e_ijab):

    """

    Build MP2 t amplitudes with shape ijab.

    Args:   
        g (array): Electron-repulsion integrals in SO basis or spatial orbital basis OOVV sliced
        e_ijab (array): Inverse epsilons tensor shape ijab

    Returns:
        t_ijab (array): Doubles t-amplitudes shape ijab

    """

    t_ijab = g * e_ijab

    return t_ijab









def transform_matrix_AO_to_SO(M, C):

    """

    Transforms two-index tensor from AO basis to SO basis.

    Args:   
        M (array): Matrix in AO basis
        C (array): Molecular orbitals in AO basis

    Returns:
        M_SO (array): Matrix in SO basis

    """

    M_SO = np.einsum("mi,mn,na->ia", C, M, C, optimize=True)

    return M_SO








def transform_P_SO_to_AO(P_SO, C_spin_block, n_SO):

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










def begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=False):

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

        error("Not enough spin-orbitals to freeze!")

    if molecule.n_core_orbitals < 0:

        error("Cannot freeze a negative number of orbitals!")

    o = slice(minimum_orbital, n_occ)
    v = slice(n_occ, None)

    epsilons_combined = SCF_output.epsilons_combined

    log("\n Preparing transformation to spin-orbital basis...", calculation, 1, silent=silent)

    # Spin-blocks electron repulsion integrals
    ERI_spin_block = np.kron(np.eye(2), np.kron(np.eye(2), ERI_AO).T)

    # Spin-blocks molecular orbitals and transforms electron repulsion integrals
    C_spin_block = spin_block_molecular_orbitals(SCF_output.molecular_orbitals_alpha, SCF_output.molecular_orbitals_beta, epsilons_combined)

    log("\n Transforming two-electron integrals...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    ERI_SO = transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block, C_spin_block)

    log("[Done]", calculation, 1, silent=silent)

    log(" Antisymmetrising two-electron integrals...  ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    # Antisymmetrise electron repulsion integrals
    g = antisymmetrise_integrals(ERI_SO)

    log("[Done]", calculation, 1, silent=silent)

    # Sorts epsilons
    epsilons_sorted = np.sort(epsilons_combined)

    # Tracks the spin state of each epsilon
    spin_labels = ['a'] * len(SCF_output.molecular_orbitals_alpha) + ['b'] * len(SCF_output.molecular_orbitals_beta)
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

        log(f"\n The {molecule.n_core_spin_orbitals} lowest energy spin-orbitals will be frozen.", calculation, 1, silent=silent)

    else:
        
        log(f"\n All electrons will be correlated.", calculation, 1, silent=silent)


    return g, C_spin_block, epsilons_sorted, ERI_spin_block, o, v, spin_labels_sorted, spin_orbital_labels_sorted









def begin_spatial_orbital_calculation(molecule, ERI_AO, SCF_output, n_doubly_occ, calculation, silent=False):

    """
    
    Sets up useful quantities for a spatial orbital calculation.

    Args:
        molecule (Molecule): Molecule object
        ERI_AO (array): Electron-repulsion integrals in AO basis
        SCF_output (Output): SCF Output object
        n_doubly_occ (int): Number of doubly occupied orbitals
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed
    
    Returns:
        ERI_MO (array): Molecular integrals in spatial MO basis
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
    o = slice(minimum_orbital, n_doubly_occ)
    v = slice(n_doubly_occ, None)

    # Recovers SCF output objects
    molecular_orbitals = SCF_output.molecular_orbitals
    epsilons = SCF_output.epsilons


    log("\n Preparing transformation to spatial-orbital basis...", calculation, 1, silent=silent)

    log("\n Transforming two-electron integrals...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()

    g = transform_ERI_AO_to_MO(ERI_AO, molecular_orbitals)

    log("[Done]", calculation, 1, silent=silent)

    # Logs information about freezing orbitals
    if calculation.freeze_core and molecule.n_core_orbitals != 0: 

        log(f"\n The {molecule.n_core_orbitals} lowest energy spatial-orbitals will be frozen.", calculation, 1, silent=silent)

    else:
        
        log(f"\n All electrons will be correlated.", calculation, 1, silent=silent)


    return g, molecular_orbitals, epsilons, o, v









def build_CIS_Hamiltonian(n_occ, n_virt, n_SO, epsilons, g):
    
    """

    Contructs array of excitations and CIS Hamiltonian matrix in determinantal basis.

    Args:   
        n_occ (int): Number of occupied spin orbitals
        n_virt (int): Number of virtual spin orbitals
        n_SO (int): Number of spin orbitals
        epsilons (array): Fock matrix orbital eigenvalues
        g (array): Antisymmetrised electron repulsion integrals in SO basis

    Returns:
        H_CIS (array): Shifted CIS Hamiltonian in determinantal basis
        excitations (array): List of possible excitations between occupied and virtual spin orbitals

    """
    
    excitations = []
    H_CIS = np.zeros((n_occ * n_virt, n_occ * n_virt))

    # Builds excitations list of every combination from occupied to virtual spin orbitals
    for i in range(n_occ):
        for a in range(n_occ, n_SO):

            excitations.append((i, a))

    # Builds shifted CIS Hamiltonian
    for p, (i, a) in enumerate(excitations):
        for q, (j, b) in enumerate(excitations):
        
            H_CIS[p, q] = (epsilons[a] - epsilons[i]) * (i == j) * (a == b) + g[a, j, i, b]


    return H_CIS, np.array(excitations)





def calculate_weights_matrix(weights_of_interest, excitations, n_occ, n_virt):

    """

   Calculates the two index weight matrix for a state of interest.

    Args:   
        weights_of_interest (array): Weights array for state of interest
        excitations (array): List of possible excitations
        n_virt (int): Number of virtual spin orbitals

    Returns:
        b_ia (array): Two index weights matrix for state of interest

    """

    # Virtual indices shifted down to start counting the vertical indices from zero
    i_indices = excitations[:, 0]
    a_indices = excitations[:, 1] - n_occ

    b_ia = np.zeros((n_occ, n_virt))

    b_ia[i_indices, a_indices] = weights_of_interest

    return b_ia











def calculate_CIS_density_matrix(n_SO, n_occ, C_spin_block, o, v, b_ia):

    """

    Calculates the CIS density matrix for a state of interest.

    Args:   
        n_SO (int): Number of spin orbitals
        n_occ (int): Number of occupied spin orbitals
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis
        o (slice): Occupied orbitals slice
        v (slice): Virtual orbitals slice
        b_ia (array): Weights matrix for state of interest

    Returns:
        P (array): Density matrix in AO basis
        P_alpha (array): Density matrix for alpha orbitals in AO basis
        P_beta (array): Density matrix for beta orbitals in AO basis
        P_transition (array): Difference density matrix for in AO basis
        P_transition_alpha (array): Difference density matrix for alpha orbitals in AO basis
        P_transition_beta (array): Difference density matrix for beta orbitals in AO basis
        
    """
    
    # Equation from Foresman 1992 - this is the transition density, delta P_CIS
    DP_ij = -1 * np.einsum("ia,ja->ij", b_ia, b_ia, optimize=True)
    DP_ab = np.einsum("ia,ib->ab", b_ia, b_ia, optimize=True)

    # Initialises HF density matrix in SO basis
    P_transition = np.zeros((n_SO, n_SO))

    # Adds on transition density blocks to form unrelaxed density matrix in SO basis
    P_transition[v, v] = DP_ab
    P_transition[o, o] = DP_ij

    P_SO = P_transition.copy()

    P_SO[o, o] += np.identity(n_occ)

    P, P_alpha, P_beta = transform_P_SO_to_AO(P_SO, C_spin_block, n_SO)
    P_transition, P_transition_alpha, P_transition_beta = transform_P_SO_to_AO(P_transition, C_spin_block, n_SO)

    return P, P_alpha, P_beta, P_transition, P_transition_alpha, P_transition_beta






def calculate_transition_dipoles(D, weights, excitations, C_spin_block):

    """

    Calculates the transition dipoles (measure of intensity) from a CIS calculation.

    Args:   
        D (array): Dipole integrals
        weights (array): Weights of each excitation (determinant) from CIS calculation
        excitations (array): List of possible excitations
        C_spin_block (array): Spin-blocked molecular orbitals in AO basis

    Returns:
        transition_dipoles (array): Transition dipoles

    """
    
    D_spin_block = np.kron(np.eye(2), D)
    D_SO = transform_matrix_AO_to_SO(D_spin_block, C_spin_block)

    # Left column is occupied, right column is virtual
    i_indices = excitations[:, 0]
    a_indices = excitations[:, 1]

    transition_dipoles = weights.T @ D_SO[a_indices, i_indices]

    return transition_dipoles





def calculate_oscillator_strengths(transition_dipoles, excitation_energies):

    """

    Calculates the oscillator strengths of all states at once.

    Args:   
        transition_dipoles (array): Transition dipoles for each state
        excitation_energies (array): Excitation energies for each state

    Returns:
        oscillator_strengths (array): Oscillator strengths for states

    """
    
    oscillator_strengths = (2 / 3) * excitation_energies * transition_dipoles ** 2

    return oscillator_strengths






def label_spin_orbital(index, calculation):

    """

    Labels and formats the spin orbitals for logging.

    Args:   
        index (int): Index of spin orbital
        calculation (Calculation): Calculation object

    Returns:
        labelled_SO (string): Formatted spin orbital label

    """
    
    reduced_index = index // 2 + 1

    labelled_SO = str(reduced_index)

    # Adds an "a" for alpha and "b" for beta for UHF
    if calculation.reference == "UHF":
        
        labelled_SO += "a" if index % 2 == 0 else "b"
    
    return labelled_SO







def process_CIS_results(excitation_energies, weights):

    """
    Processes arrays from CIS calculation to group triplets.

    Args:   
        excitation_energies (array): Excitation energies
        percent_contributions (array): Percent contributions to each state
        transition_dipoles (array): Transition dipoles
        oscillator_strengths (array): Oscillator strengths

    Returns:
        new_excitation_energies (array): Excitation energies
        new_percent_contributions (array): Percent contributions to each state
        new_transition_dipoles (array): Transition dipoles
        new_oscillator_strengths (array): Oscillator strengths
        state_types (array): Either triplet or singlet for states

    """

    threshold = 0.001  # Adjust this value as needed

    # Initialize variables to store the new excitation energies, contributions, transition dipoles, oscillator strengths, and state types
    new_excitation_energies = []
    new_percent_contributions = []
    new_weights = []
    state_types = []  # To store 'Singlet' or 'Triplet' for each state

    n_states = len(excitation_energies)
    used_indices = set()
    i = 0

    while i < n_states:
        if i in used_indices:
            i += 1
            continue

        # Start a new group with the current excitation energy
        group_indices = [i]
        current_energy = excitation_energies[i]

        # Compare with subsequent excitation energies to find close ones
        for j in range(i + 1, n_states):
            if abs(excitation_energies[j] - current_energy) < threshold:
                group_indices.append(j)
                used_indices.add(j)
            else:
                break

        num_in_group = len(group_indices)

        if num_in_group == 3:
            # Assume this is a Triplet state
            state_types.append('Triplet')

            # Average the excitation energies
            avg_energy = np.mean(excitation_energies[group_indices])
            new_excitation_energies.append(avg_energy)

            first_weights = weights[:, group_indices[1]]
            new_weights.append(first_weights)


        elif num_in_group == 1:

            # Assume these are Singlet states
            for idx in group_indices:
                state_types.append('Singlet')
                new_excitation_energies.append(excitation_energies[idx])
                new_weights.append(weights[:, idx])


        i = group_indices[-1] + 1  # Move to the next unprocessed excitation energy

    # Convert lists to numpy arrays
    new_excitation_energies = np.array(new_excitation_energies)
    new_weights = np.array(new_weights).T

    new_percent_contributions = new_weights ** 2 * 100


    return new_excitation_energies, new_weights, new_percent_contributions, state_types







def print_excited_state_information(excitation_energies, SCF_output, contributions, excitations_formatted, calculation, state_types, weights, silent=False):
      
    """
    Prints information about each excited state.

    Args:   
        excitation_energies (array): Excitation energies
        SCF_output (Output): Output from SCF calculation
        contributions (array): Percent contributions to each state
        excitations_formatted (list): Formatted excitations
        calculation (Calculation): Calculation object
        state_types (list): List of 'Singlet' or 'Triplet' for each state
        silent (bool, optional): Should output be silenced

    Returns:
        None: Nothing is returned
    """

    for state in range(len(excitation_energies)):

        if state < calculation.n_states:
            
            # Prints excitation energy and energy of each state
            if calculation.reference == "UHF":
                state_type = ""
            else:
                state_type = state_types[state]
            log(f"\n\n  ~~~~~ State {state + 1} ~~~~~  {state_type}", calculation, 2, silent=silent)

            log(f"\n  Excitation energy:   {excitation_energies[state]:13.10f}", calculation, 2, silent=silent)
            log(f"  Energy of state:     {(excitation_energies[state] + SCF_output.energy):13.10f}\n", calculation, 2, silent=silent)
            
            # Aggregate contributions for duplicate excitations
            excitation_contributions = {}

            for i, excitation in enumerate(excitations_formatted):
                
                # Collect all contributions without threshold check
                excitation_key = (excitation[0], excitation[1])

                if excitation_key not in excitation_contributions:

                    excitation_contributions[excitation_key] = {
                        'contribution': contributions[i, state],
                        'weight': weights[i, state]
                    }

                else:

                    excitation_contributions[excitation_key]['contribution'] += contributions[i, state]
                    excitation_contributions[excitation_key]['weight'] += weights[i, state] 

            # Prints the aggregated contributions if they are above the threshold
            for excitation_key, values in excitation_contributions.items():
                
                total_contribution = values['contribution']

                if total_contribution > calculation.CIS_contribution_threshold:
                    exc_from, exc_to = excitation_key

                    if calculation.reference  == "UHF":
                        log(f"    {exc_from} -> {exc_to}  :  {total_contribution:6.2f} %    ({values['weight']:8.5f})", calculation, 2, silent=silent)
                    else:
                        log(f"    {exc_from} -> {exc_to}  :  {total_contribution:6.2f} %   ", calculation, 2, silent=silent)








def print_CIS_absorption_spectrum(molecule, excitation_energies_eV, calculation, frequencies_per_cm, wavelengths_nm, transition_dipoles, oscillator_strengths, state_types, silent=False):
    
    """

    Prints CIS absorption spectrum information.

    Args:   
        molecule (Molecule): Molecule object
        excitation_energies_eV (array): Excitation energies in eV
        calculation (Calculation): Calculation object
        frequencies_per_cm (array): Frequencies in per cm
        wavelengths_nm (array): Wavelengths in nm
        transition_dipoles (array): Transition dipoles
        oscillator_strengths (array): Oscillator strengths
        silent (bool, optional): Should output be silenced

    Returns:
        None: Nothing is returned

    """
   
    log(f"\n\n Transition dipole moment origin is the centre of mass, {bohr_to_angstrom(molecule.centre_of_mass):.4f} angstroms from the first atom.", calculation, 1, silent=silent)
    
    log_big_spacer(calculation, silent=silent, start="\n")
    log("                                          CIS Absorption Spectrum", calculation, 1, silent=silent, colour="white")
    log_big_spacer(calculation, silent=silent)
    log("   State     Energy (eV)     Freq. (per cm)     Wavelength (nm)     Osc. Strength     Transition Dipole", calculation, 1, silent=silent)
    log_big_spacer(calculation, silent=silent)

    for state in range(len(excitation_energies_eV)):

        if state < calculation.n_states:
            
            state_type = " - " + state_types[state][0] if calculation.reference == "RHF" else "  "
            gap = "" if calculation.reference == "RHF" else "  "

            log(f"  {gap}{(state + 1):2}{state_type}       {excitation_energies_eV[state]:7.4f}          {frequencies_per_cm[state]:8.1f}            {wavelengths_nm[state]:5.1f}              {oscillator_strengths[state]:.6f}          {transition_dipoles[state]:10.7f}", calculation, 1, silent=silent)

    log_big_spacer(calculation, silent=silent)








def calculate_CIS_doubles_correction(excitation_energy, epsilons, root, g, o, v, b_ia, calculation, silent=False):

    """

    Calculates CIS(D) correction to CIS excitation energy of one state.

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
        E_CIS_D (float): CIS(D) correction to CIS excitation energy

    """

    # Equations taken from Head-Gordon paper
    log_spacer(calculation, silent=silent, start="\n")
    log("          CIS(D) Perturbative Correction", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log(f"  Applying doubles correction to state {root + 1} only.", calculation, 1, silent=silent)
   
    log(f"\n  Building doubles amplitudes...           ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    # Builds and inverts inverse epsilons tensor, to upright e_ijab_inv
    e_ijab_inv = 1 / build_doubles_epsilons_tensor(epsilons, epsilons, o, o, v, v)
    e_ijab_inv_minus_w = 1 / (e_ijab_inv + excitation_energy)

    t_ijab = build_MP2_t_amplitudes(g[o, o, v, v], 1 / e_ijab_inv)

    log(f"  [Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating direct contribution...  ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    u_1 = np.einsum("abcj,ic->ijab", g[v, v, v, o], b_ia, optimize=True)
    u_2 = np.einsum("abci,jc->ijab", g[v, v, v, o], b_ia, optimize=True)
    u_3 = np.einsum("kaij,kb->ijab", g[o, v, o, o], b_ia, optimize=True)
    u_4 = np.einsum("kbij,ka->ijab", g[o, v, o, o], b_ia, optimize=True)

    u_ijab = u_1 - u_2 + u_3 - u_4

    log(f"       [Done]", calculation, 1, silent=silent)

    log(f"  Calculating indirect contribution...  ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    v_1 = (1 / 2) * np.einsum("jkbc,ib,jkca->ia", g[o, o, v, v], b_ia, t_ijab, optimize=True)
    v_2 = (1 / 2) * np.einsum("jkbc,ja,ikcb->ia", g[o, o, v, v], b_ia, t_ijab, optimize=True)
    v_3 = np.einsum("jkbc,jb,ikac->ia", g[o, o, v, v], b_ia, t_ijab, optimize=True)

    v_ia = v_1 + v_2 + v_3

    log(f"     [Done]", calculation, 1, silent=silent)

    log(f"\n  Calculating CIS(D) correction...      ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    E_CIS_D = (1 / 4) * np.einsum("ijab,ijab,ijab->", u_ijab, u_ijab, e_ijab_inv_minus_w, optimize=True) + np.einsum("ia,ia->", b_ia, v_ia, optimize=True)

    log(f"     [Done]", calculation, 1, silent=silent)
    
    # Correction energy in eV prints if P used
    log(f"\n  Excitation energy from CIS:       {excitation_energy:15.10f}", calculation, 1, silent=silent)
    log(f"  Correction energy from CIS(D):    {E_CIS_D:15.10f}", calculation, 1, silent=silent)
    log(f"  Correction energy (eV):           {(E_CIS_D * constants.eV_in_hartree):15.10f}", calculation, 3, silent=silent)
    log(f"\n  Excitation energy from CIS(D):    {(E_CIS_D + excitation_energy):15.10f}", calculation, 1, silent=silent)
    log_spacer(calculation, silent=silent)
  
    return E_CIS_D





def run_CIS(ERI_AO, n_occ, n_virt, n_SO, calculation, SCF_output, molecule, silent=False):

    """
    Begins a configuration interaction singles calculation.

    Args:   
        ERI_AO (array): Electron repulsion integrals in AO basis
        n_occ (int): Number of occupied spin orbitals
        n_virt (int): Number of virtual spin orbitals
        n_SO (int): Number of spin orbitals
        calculation (Calculation): Calculation object
        SCF_output (Output): SCF output object
        molecule (Molecule): Molecule object
        silent (bool, optional): Should output be silenced

    Returns:
        E_CIS (float): Energy of state of interest
        E_transition (float): Transition energy to state of interest
        P_CIS (array): Density matrix of state of interest in AO basis      
        P_CIS_a (array): Density matrix of alpha orbitals of state of interest in AO basis
        P_CIS_b (array): Density matrix of beta orbitals of state of interest in AO basis

    """
    
    # Converting into computer counting from human counting
    root = calculation.root - 1


    g, C_spin_block, epsilons_sorted, _, o, v, _, _ = begin_spin_orbital_calculation(molecule, ERI_AO, SCF_output, n_occ, calculation, silent=silent)


    log_spacer(calculation, silent=silent, start="\n")
    log("         Configuration Interaction Singles", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)
    
    log("\n  Building CIS Hamiltonian...                ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

    H_CIS, excitations = build_CIS_Hamiltonian(n_occ, n_virt, n_SO, epsilons_sorted, g)

    log("[Done]", calculation, 1, silent=silent)

    log("  Diagonalising CIS Hamiltonian...           ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

    excitation_energies, weights = np.linalg.eigh(H_CIS)

    excitation_energies, weights, percent_contributions, state_types = process_CIS_results(excitation_energies, weights)

    log("[Done]", calculation, 1, silent=silent)


    log("\n  Calculating transition dipoles...          ", calculation, 1, silent=silent, end=""); sys.stdout.flush()

    transition_dipoles = calculate_transition_dipoles(SCF_output.D, weights, excitations, C_spin_block)
    oscillator_strengths = calculate_oscillator_strengths(transition_dipoles, excitation_energies)

    log("[Done]", calculation, 1, silent=silent)


    log(f"  Constructing density matrix...             ", calculation, 1, silent=silent, end=""); sys.stdout.flush()
    
    try:
        weights_of_interest = weights[:, root]
        
    except: error("Specified root does not exist!")

    # Calculates weight matrix for state of interest
    b_ia = calculate_weights_matrix(weights_of_interest, excitations, n_occ, n_virt)
    P_CIS, P_CIS_a, P_CIS_b, P_transition, P_transition_alpha, P_transition_beta = calculate_CIS_density_matrix(n_SO, n_occ, C_spin_block, o, v, b_ia)

    log("[Done]", calculation, 1, silent=silent)

    log("\n  Printing excited state information...  ", calculation, 2, silent=silent)
    log(f"  Printing contributions larger than {calculation.CIS_contribution_threshold:.1f}%.  ", calculation, 2, silent=silent)

    excitations_formatted = [(label_spin_orbital(i, calculation), label_spin_orbital(a, calculation)) for i, a in excitations]


    frequencies_per_cm = excitation_energies * constants.per_cm_in_hartree
    wavelengths_nm = 1e7 / frequencies_per_cm
    excitation_energies_eV = constants.eV_in_hartree * excitation_energies

    print_excited_state_information(excitation_energies, SCF_output, percent_contributions, excitations_formatted, calculation, state_types, weights, silent=silent)

    print_CIS_absorption_spectrum(molecule, excitation_energies_eV, calculation, frequencies_per_cm, wavelengths_nm, transition_dipoles, oscillator_strengths, state_types, silent=silent)

    # Energy of transition of state of interest
    try:

        E_transition = excitation_energies[root]
    
    except: error("Specified root does not exist!")

    # Optionally applies doubles correction to transition energy for a specified state
    if "[D]" in calculation.method:
            
        E_CIS_D = calculate_CIS_doubles_correction(E_transition, epsilons_sorted, root, g, o, v, b_ia, calculation, silent=silent)

        E_transition += E_CIS_D

    E_CIS = SCF_output.energy + E_transition


    return E_CIS, E_transition, P_CIS, P_CIS_a, P_CIS_b, P_transition, P_transition_alpha, P_transition_beta
