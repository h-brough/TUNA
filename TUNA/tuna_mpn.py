import numpy as np
import sys
import scipy
import tuna_scf as scf
import tuna_ci as ci
from tuna_util import *



def calculate_MP2_energy(t_ijab, ERI_SO):

    """

    Calculates the MP2 energy.

    Args:
        t_ijab (array): Amplitude with shape ijab
        ERI_SO (array): Electron repulsion integrals (optionally antisymmetrised) in SO basis

    Returns:
        E_MP2 (float): Contribution to MP2 energy

    """

    E_MP2 = (1 / 4) * np.einsum("ijab,ijab->", t_ijab, ERI_SO, optimize=True)

    return E_MP2






def build_MP2_density_contribution(n_atomic_orbitals, t_ijab, o, v):

    """

    Calculates the MP2 energy.

    Args:
        n_atomic_orbitals (int): Number of atomic orbitals
        t_ijab (array): Amplitude with shape ijab
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice

    Returns:
        P_MP2 (array): Contribution to MP2 density matrix

    """

    P_MP2 = np.zeros((n_atomic_orbitals, n_atomic_orbitals))

    # Occupied-occupied and virtual-virtual contributions to MP2 density matrix
    P_MP2[v, v] +=  (1 / 2) * np.einsum('ijac,ijbc->ab', t_ijab, t_ijab, optimize=True)
    P_MP2[o, o] += - (1 / 2) * np.einsum('jkab,ikab->ij', t_ijab, t_ijab, optimize=True)

    return P_MP2






def spin_component_scale_MP2_energy(E_MP2_SS, E_MP2_OS, same_spin_scaling, opposite_spin_scaling, calculation, silent=False):
    
    """

    Scales the different spin components of the MP2 energy.

    Args:
        E_MP2_SS (float): Same-spin contribution to MP2 energy
        E_MP2_OS (float): Opposite-spin contribution to MP2 energy
        same_spin_scaling (float): Scaling of same-spin contribution
        opposite_spin_scaling (float): Scaling of opposite-spin contribution
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2_SS_scaled (float): Same-spin scaled contribution to MP2 energy
        E_MP2_OS_scaled (float): Opposite-spin scaled contribution to MP2 energy

    """

    # Scaling energy components
    E_MP2_SS_scaled = same_spin_scaling * E_MP2_SS 
    E_MP2_OS_scaled = opposite_spin_scaling * E_MP2_OS 

    log(f"  Same-spin scaling: {same_spin_scaling:.3f}", calculation, 1, silent=silent)
    log(f"  Opposite-spin scaling: {opposite_spin_scaling:.3f}\n", calculation, 1, silent=silent)
    

    return E_MP2_SS_scaled, E_MP2_OS_scaled
    





def calculate_natural_orbitals(P, X, calculation, silent=False):

    """

    Calculates the natural orbitals and occupancies of a density matrix.

    Args:
        P (array): Density matrix in AO basis
        X (array): Fock transformation matrix
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        natural_orbital_occupancies (array): Occupancies for natural orbitals
        natural_orbitals (array): Natural orbitals in AO basis

    """

    # Transforms density matrix to orthogonal basis
    P_orthogonal = np.linalg.inv(X) @ P @ np.linalg.inv(X)

    natural_orbital_occupancies, natural_orbitals = np.linalg.eigh(P_orthogonal)

    natural_orbital_occupancies = np.sort(natural_orbital_occupancies)[::-1]
    sum_of_occupancies = np.sum(natural_orbital_occupancies)

    natural_orbitals = natural_orbitals[:, natural_orbital_occupancies.argsort()] 

    log("\n  Natural orbital occupancies: \n", calculation, 2, silent=silent)

    # Prints out all the natural orbital occupancies, the sum and the trace of the density matrix
    for i in range(len(natural_orbital_occupancies)): 
        
        log(f"    {i + 1}.   {natural_orbital_occupancies[i]:.8f}", calculation, 2, silent=silent)

    log(f"\n  Sum of natural orbital occupancies: {sum_of_occupancies:.6f}", calculation, 2, silent=silent)
    log(f"  Trace of density matrix: {np.trace(P_orthogonal):.6f}", calculation, 2, silent=silent)


    return natural_orbital_occupancies, natural_orbitals






def calculate_MP3_energy(e_ijab, ERI_SO_ansym, o, v, calculation, silent=False):

    """

    Calculates the MP3 correlation energy.

    Args:
        e_ijab (array): Epsilions inverse tensor shape ijab
        ERI_SO_ansym (array): Antisymmetrised electron repulsion integrals in SO basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        calculation (Calculation): Calculation object
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP3 (float): MP3 correlation energy

    """

    log("  Calculating MP3 correlation energy...      ", calculation, 1, end="", silent=silent); sys.stdout.flush()
        
    E_MP3 = (1 / 8) * np.einsum('ijab,klij,abkl,ijab,klab->', ERI_SO_ansym[o, o, v, v], ERI_SO_ansym[o, o, o, o], ERI_SO_ansym[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += (1 / 8) * np.einsum('ijab,abcd,cdij,ijab,ijcd->', ERI_SO_ansym[o, o, v, v], ERI_SO_ansym[v, v, v, v], ERI_SO_ansym[v, v, o, o], e_ijab, e_ijab, optimize=True)
    E_MP3 += np.einsum('ijab,kbcj,acik,ijab,ikac->', ERI_SO_ansym[o, o, v, v], ERI_SO_ansym[o, v, v, o], ERI_SO_ansym[v, v, o, o], e_ijab, e_ijab, optimize=True)

    log(f"[Done]\n\n  MP3 correlation energy:             {E_MP3:13.10f}", calculation, 1, silent=silent)

    return E_MP3







def run_MP2(molecule, calculation, SCF_output, n_SO, ERI_spin_block, X, silent=False):

    """

    Calculates the MP2 energy and unrelaxed density.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        SCF_output (Output): Output object
        n_SO (int): Number of spin orbitals
        ERI_spin_block (array): Spin-blocked electron repulsion integrals in AO basis
        X (array): Fock transformation matrix
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): MP2 correlation energy
        P (array): MP2 unrelaxed density matrix in AO basis
        P_alpha (array): MP2 unrelaxed density matrix for alpha orbitals in AO basis
        P_beta (array): MP2 unrelaxed density matrix for beta orbitals in AO basis

    """

    molecular_orbitals_alpha = SCF_output.molecular_orbitals_alpha
    molecular_orbitals_beta = SCF_output.molecular_orbitals_beta

    epsilons_alpha = SCF_output.epsilons_alpha
    epsilons_beta = SCF_output.epsilons_beta

    n_occ_alpha = molecule.n_alpha
    n_occ_beta = molecule.n_beta

    # Defines occupied and virtual slices for alpha and beta orbitals
    o_a = slice(0, n_occ_alpha)
    o_b = slice(0, n_occ_beta)

    v_a = slice(n_occ_alpha, n_SO // 2)
    v_b = slice(n_occ_beta, n_SO // 2)

    # Initialises unscaled scaling factors
    same_spin_scale = 1
    opposite_spin_scale = 1


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("                MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)

    # Spin-blocks alpha and beta orbitals separately
    C_spin_block_alpha = ci.spin_block_molecular_orbitals(molecular_orbitals_alpha, molecular_orbitals_alpha, epsilons_alpha)
    C_spin_block_beta = ci.spin_block_molecular_orbitals(molecular_orbitals_beta, molecular_orbitals_beta, epsilons_beta)

    log("  Transforming two-electron integrals... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    # Transforms ERI for alpha, beta and alpha and beta spins
    ERI_SO_a = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_alpha, C_spin_block_alpha)
    ERI_SO_b = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_beta, C_spin_block_beta)
    ERI_SO_ab = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block_alpha, C_spin_block_beta)

    # Antisymmetrises alpha and beta spins, but not alpha-beta spins
    ERI_SO_ansym_a = ci.antisymmetrise_integrals(ERI_SO_a)
    ERI_SO_ansym_b = ci.antisymmetrise_integrals(ERI_SO_b)

    log("    [Done]", calculation, 1, silent=silent)

    log("\n  Calculating MP2 correlation energy... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    epsilons_alpha = np.sort(epsilons_alpha)
    epsilons_beta = np.sort(epsilons_beta)

    # Slicing out occupied and virtual parts of alpha-alpha, beta-beta and alpha-beta contributions to ERI
    ERI_SO_aa = ERI_SO_ansym_a[o_a, o_a, v_a, v_a]
    ERI_SO_bb = ERI_SO_ansym_b[o_b, o_b, v_b, v_b]
    ERI_SO_ab = ERI_SO_ab[o_a, o_b, v_a, v_b]

    # Epsilons tensor for alpha-alpha, beta-beta and alpha-beta spin pairs
    e_ijab_aa = ci.build_epsilons_tensor(epsilons_alpha, epsilons_alpha, o_a, o_a, v_a, v_a)
    e_ijab_bb = ci.build_epsilons_tensor(epsilons_beta, epsilons_beta, o_b, o_b, v_b, v_b)
    e_ijab_ab = ci.build_epsilons_tensor(epsilons_alpha, epsilons_beta, o_a, o_b, v_a, v_b)

    # MP2 amplitudes for alpha-alpha, beta-beta, alpha-beta and beta-alpha pairs 
    t_ijab_aa = ci.build_MP2_t_amplitudes(ERI_SO_aa, e_ijab_aa)
    t_ijab_bb = ci.build_MP2_t_amplitudes(ERI_SO_bb, e_ijab_bb)
    t_ijab_ab = ci.build_MP2_t_amplitudes(ERI_SO_ab, e_ijab_ab)
    t_ijab_ba = t_ijab_ab.transpose(1,0,3,2) 
    
    # Calculates MP2 energy for alpha-alpha, beta-beta and alpha-beta pairs
    E_aa = calculate_MP2_energy(t_ijab_aa, ERI_SO_aa)
    E_bb = calculate_MP2_energy(t_ijab_bb, ERI_SO_bb)
    E_ab = 4 * calculate_MP2_energy(t_ijab_ab, ERI_SO_ab)

    # Calculates same-spin and opposite-spin contributions
    E_MP2_SS = E_aa + E_bb
    E_MP2_OS = E_ab

    log("     [Done]\n", calculation, 1, silent=silent)

    # Optionally scales the same- and opposite-spin contributions to energy
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"]: 
        
        E_MP2_SS, E_MP2_OS = spin_component_scale_MP2_energy(E_MP2_SS, E_MP2_OS, calculation.same_spin_scaling, calculation.opposite_spin_scaling, calculation, silent=silent)


    E_MP2 = E_MP2_SS + E_MP2_OS


    log(f"  Energy from alpha-alpha pairs:      {E_aa:13.10f}", calculation, 1, silent=silent)
    log(f"  Energy from beta-beta pairs:        {E_bb:13.10f}", calculation, 1, silent=silent)
    log(f"  Energy from alpha-beta pairs:       {E_ab:13.10f}", calculation, 1, silent=silent)

    log(f"\n  Same spin contribution:             {E_MP2_SS:13.10f}", calculation, 1, silent=silent)
    log(f"  Opposite spin contribution:         {E_MP2_OS:13.10f}", calculation, 1, silent=silent)
    log(f"\n  MP2 correlation energy:             {E_MP2:13.10f}", calculation, 1, silent=silent)

    log("\n  Constructing MP2 unrelaxed density... ", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    P_MP2_a = np.zeros((n_SO // 2, n_SO // 2))
    P_MP2_b = np.zeros((n_SO // 2, n_SO // 2))

    # Fills up alpha and beta density matrices with occupied orbitals
    np.fill_diagonal(P_MP2_a[:n_occ_alpha, :n_occ_alpha], 1)
    np.fill_diagonal(P_MP2_b[:n_occ_beta, :n_occ_beta], 1)

    # Finds alpha-alpha and alpha-beta density contributions
    P_MP2_aa = build_MP2_density_contribution(n_SO // 2, t_ijab_aa, o_a, v_a)
    P_MP2_ab = build_MP2_density_contribution(n_SO // 2, t_ijab_ab, o_a, v_a) 

    # Finds beta-beta and beta-alpha density contributions
    P_MP2_bb = build_MP2_density_contribution(n_SO // 2, t_ijab_bb, o_b, v_b)
    P_MP2_ba = build_MP2_density_contribution(n_SO // 2, t_ijab_ba, o_b, v_b)

    # Optionally applies spin scaling to density
    if calculation.method in ["SCS-MP2", "USCS-MP2", "SCS-MP3", "USCS-MP3"]:

        same_spin_scale = calculation.same_spin_scaling
        opposite_spin_scale = calculation.opposite_spin_scaling

    # Builds alpha and beta MP2 density matrices
    P_MP2_a += same_spin_scale * P_MP2_aa + opposite_spin_scale * 2 * P_MP2_ab
    P_MP2_b += same_spin_scale * P_MP2_bb + opposite_spin_scale * 2 * P_MP2_ba

    # Transform MP2 density back to AO basis
    P_alpha = molecular_orbitals_alpha @ P_MP2_a @ molecular_orbitals_alpha.T 
    P_beta = molecular_orbitals_beta @ P_MP2_b @ molecular_orbitals_beta.T 

    # Total AO density matrix
    P = P_alpha + P_beta

    log("     [Done]", calculation, 1, silent=silent)
    
    # Calculates and prints natural orbital occupancies
    calculate_natural_orbitals(P, X, calculation, silent=silent)

    return E_MP2, P, P_alpha, P_beta







def run_OMP2(molecule, calculation, ERI_SO_anysm, C_spin_block, H_core, V_NN, n_SO, X, E_HF, ERI_spin_block, o, v, silent=False):

    """

    Calculates the orbital-optimised MP2 energy and relaxed density.

    Args:
        molecule (Molecule): Molecule object
        calculation (Calculation): Calculation object
        ERI_SO_anysm (array): Antisymmetrised electron repulsion integrals in SO basis
        C_spin_block (array): Spin-blocked molecular integrals in AO basis
        H_core (array): Core hamiltonian matrix in AO basis
        V_NN (float): Nuclear-nuclear repulsion energy
        n_SO (int): Number of spin orbitals
        X (array): Fock transformation matrix
        E_HF (float): Hartree-Fock total energy
        ERI_spin_block (array): Spin-blocked electron repulsion integrals in AO basis
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        silent (bool, optional): Should anything be printed

    Returns:
        E_OMP2 (float): OMP2 correlation energy
        P (array): OMP2 relaxed density matrix in AO basis
        P_alpha (array): OMP2 relaxed density matrix for alpha orbitals in AO basis
        P_beta (array): OMP2 relaxed density matrix for beta orbitals in AO basis

    """

    n_occ = molecule.n_occ
    n_virt = molecule.n_virt

    E_conv = calculation.OMP2_conv
    OMP2_max_iter = calculation.OMP2_max_iter


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("      Orbital-optimised MP2 Energy and Density ", calculation, 1, silent=silent, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)


    log(f"\n  Tolerance for energy convergence:   {E_conv:.10f}", 1, silent=silent)
    log("\n  Starting orbital-optimised MP2 iterations...\n", calculation, 1, end="", silent=silent)


    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("  Step          Correlation E               DE", calculation, 1, silent=silent)
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)

    E_OMP2_old = 0

    n = np.newaxis

    # Spin blocks and transforms core Hamiltonian to spin-orbital basis
    H_core_spin_block = np.kron(np.eye(2), H_core)
    H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)

    # Sets up matrices based on number of spin orbitals
    P_corr = np.zeros((n_SO, n_SO))
    P_ref = np.zeros((n_SO, n_SO))
    R = np.zeros((n_SO, n_SO))
    D_corr = np.zeros((n_SO, n_SO, n_SO, n_SO))

    # Fills reference one-particle density matrix with ones, up to number of occupied spin orbitals
    P_ref[o, o] = np.identity(n_occ)

    # Sets up t amplitudes based on number of virtual and occupied orbitals
    t_abij = np.zeros((n_virt, n_virt, n_occ, n_occ))


    for iteration in range(1, OMP2_max_iter + 1):

        # Build Fock matrix from core Hamiltonian and two-electron integrals, in spin-orbital basis
        F = H_core_SO + np.einsum('piqi->pq', ERI_SO_anysm[:, o, :, o], optimize=True)

        # Build off-diagonal Fock matrix, epsilons obtained from diagonal elements
        F_prime = F.copy()
        np.fill_diagonal(F_prime, 0)
        epsilons_combined = F.diagonal()

        # Full t amplitudes for MP2, with permutations 
        t_1 = ERI_SO_anysm[v, v, o, o]
        t_2 = np.einsum('ac,cbij->abij', F_prime[v, v], t_abij, optimize=True)
        t_3 = np.einsum('ki,abkj->abij', F_prime[o, o], t_abij, optimize=True)
        t_abij = t_1 + t_2 - t_2.transpose((1, 0, 2, 3)) - t_3 + t_3.transpose((0, 1, 3, 2))
        
        # Epsilons tensor built and transposed to abij shape, forms final t amplitudes by multiplication
        e_abij = ci.build_epsilons_tensor(epsilons_combined, epsilons_combined, o, o, v, v).transpose(2,3,0,1) 
        t_abij *= e_abij

        # Build one-particle reduced density matrix, using t_ijab
        P_corr = build_MP2_density_contribution(n_SO, t_abij.transpose(2,3,0,1), o, v)

        # Add to reference P, which is diagonal of ones up to number of occupied spin orbitals
        P_OMP2 = P_corr + P_ref 

        # Forms two-particle density matrix from t amplitudes
        D_corr[v, v, o, o] = t_abij
        D_corr[o, o, v, v] = t_abij.transpose(2,3,0,1)

        # Forms other contributions to two-particle density matrix and their permutations 
        D_2 = np.einsum('rp,sq->rspq', P_corr, P_ref, optimize=True)
        D_3 = np.einsum('rp,sq->rspq', P_ref, P_ref, optimize=True)

        # Forms total two-particle density matrix
        D = D_corr + D_2 - D_2.transpose(1, 0, 2, 3) - D_2.transpose(0, 1, 3, 2) + D_2.transpose(1, 0, 3, 2) + D_3 - D_3.transpose(1, 0, 2, 3)

        # Forms generalised Fock matrix
        F = np.einsum('pr,rq->pq', H_core_SO, P_OMP2, optimize=True) + (1 / 2) * np.einsum('prst,stqr->pq', ERI_SO_anysm, D, optimize=True)

        # Only consider rotations between occupied and virtual orbitals, as only these change the energy
        R[v, o] = (F - F.T)[v, o] / (epsilons_combined[n, o] - epsilons_combined[v, n])

        # Builds orbital rotation matrix by exponentiation
        U = scipy.linalg.expm(R - R.T)

        # Rotates orbitals
        C_spin_block = C_spin_block @ U

        # Uses new orbitals to form new core Hamiltonian and antisymmetrised two-electron integrals
        H_core_SO = ci.transform_matrix_AO_to_SO(H_core_spin_block, C_spin_block)
        ERI_SO = ci.transform_ERI_AO_to_SO(ERI_spin_block, C_spin_block, C_spin_block)
        ERI_SO_anysm = ci.antisymmetrise_integrals(ERI_SO)

        # Calculates total energy from one-electron and two-electron contractions with one- and two-electron Hamiltonians
        E_OMP2 = V_NN + scf.calculate_one_electron_property(P_OMP2, H_core_SO) + scf.calculate_two_electron_property(D, ERI_SO_anysm)

        # Determines change in energy, and correlation energy by removing original reference (HF) energy
        E_OMP2 = E_OMP2 - E_HF
        delta_E = E_OMP2 - E_OMP2_old

        # Formats output lines nicely
        log(f"  {iteration:3.0f}           {E_OMP2:13.10f}         {delta_E:13.10f}", calculation, 1, silent=silent)
        
        # Updates the "old" energy, to continue loop
        E_OMP2_old = E_OMP2

        # If convergence criteria has been reached, exit loop
        if (abs(delta_E)) < E_conv:

            break
        elif iteration >= OMP2_max_iter: error("Orbital-optimised MP2 failed to converge! Try increasing the maximum iterations?")


    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)

    log(f"\n  OMP2 correlation energy:           {E_OMP2:.10f}", calculation, 1, silent=silent)

    log("\n  Constructing OMP2 relaxed density...", calculation, 1, end="", silent=silent); sys.stdout.flush()
    
    P, P_alpha, P_beta = ci.transform_P_SO_to_AO(P_OMP2, C_spin_block, n_SO)

    log("      [Done]", calculation, 1, silent=silent)

    # Calculates natural orbitals from OMP2 density
    calculate_natural_orbitals(P, X, calculation, silent=silent)

    return E_OMP2, P, P_alpha, P_beta






def run_MP3(calculation, ERI_SO_anysm, epsilons_sorted, E_MP2, o, v, silent=False):

    """

    Calculates the (SCS-)MP3 energy.

    Args:
        calculation (Calculation): Calculation object
        ERI_SO_anysm (array): Antisymmetrised electron repulsion integrals in SO basis
        epsilons_sorted (array): Sorted array of Fock matrix eigenvalues
        E_MP2 (float): MP2 correlation energy
        o (slice): Occupied orbital slice
        v (slice): Virtual orbital slice
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP3 (float): (SCS-)MP3 correlation energy

    """

    log("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)
    log("                      MP3 Energy  ", calculation, 1, silent=silent, colour="white")
    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)    


    e_ijab = ci.build_epsilons_tensor(epsilons_sorted, epsilons_sorted, o, o, v, v)

    E_MP3 = calculate_MP3_energy(e_ijab, ERI_SO_anysm, o, v, calculation, silent=silent)


    # Applies Grimme's default scaling to MP3 energy if SCS-MP3 is requested, then prints this information
    if calculation.method in ["SCS-MP3", "USCS-MP3"]:

        E_MP3 *= calculation.MP3_scaling
        
        log(f"\n  Scaling for MP3: {calculation.MP3_scaling:.3f}\n", calculation, 1, silent=silent)
        log(f"  Scaled MP3 correlation energy:      {E_MP3:.10f}", calculation, 1, silent=silent)
        log(f"  SCS-MP3 correlation energy:         {(E_MP3 + E_MP2):.10f}", calculation, 1, silent=silent)


    return E_MP3






def calculate_Moller_Plesset(method, molecule, SCF_output, ERI_AO, calculation, X, H_core, V_NN, silent=False):

    """

    Calculates the requested Moller-Plesset perturbation theory energy and density.

    Args:
        method (str): Electronic structure method
        molecule (Molecule): Molecule object
        SCF_output (Output): SCF output object
        ERI_AO (array): Electron repulsion integrals in AO basis 
        calculation (Calculation): Calculation object
        X (array): Fock transformation matrix
        H_core (array): Core Hamiltonian matrix in AO basis
        V_NN (float): Nuclear-nuclear repulsion energy
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): (SCS-)MP2 correlation energy
        E_MP3 (float): (SCS-)MP3 correlation energy
        P (array): (SCS-)MP2 unrelaxed density matrix in AO basis
        P_alpha (array): (SCS-)MP2 unrelaxed alpha density matrix in AO basis
        P_beta (array): (SCS-)MP2 unrelaxed beta density matrix in AO basis

    """

    E_MP2 = 0
    E_MP3 = 0

    n_SO = molecule.n_SO

    # Calculates useful quantities for all spin orbital calculations
    ERI_SO_anysm, C_spin_block, epsilons_sorted, ERI_spin_block, o, v = ci.begin_spin_orbital_calculation(ERI_AO, SCF_output, molecule.n_occ)
    

    if method in ["OMP2", "UOMP2"]: 
        
        E_MP2, P, P_alpha, P_beta = run_OMP2(molecule, calculation, ERI_SO_anysm, C_spin_block, H_core, V_NN, n_SO, X, SCF_output.energy, ERI_spin_block, o, v, silent=silent)

    elif method in ["MP2", "SCS-MP2", "UMP2", "USCS-MP2", "MP3", "UMP3", "SCS-MP3", "USCS-MP3"]:
        
         E_MP2, P, P_alpha, P_beta = run_MP2(molecule, calculation, SCF_output, n_SO, ERI_spin_block, X, silent=silent)

         if method in ["MP3", "UMP3", "SCS-MP3", "USCS-MP3"]: 
             
             E_MP3 =  run_MP3(calculation, ERI_SO_anysm, epsilons_sorted, E_MP2, o, v, silent=silent)


    log(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", calculation, 1, silent=silent)


    return E_MP2, E_MP3, P, P_alpha, P_beta
