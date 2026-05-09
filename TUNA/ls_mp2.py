import numpy as np
import scipy
import sys
from numpy import ndarray
from tuna_calc import Calculation
from tuna_util import log, log_spacer
import tuna_scf as scf



def run_restricted_Laplace_MP2(ERI_AO: ndarray, molecular_orbitals: ndarray, F: ndarray, n_doubly_occ: int, calculation: Calculation, P: ndarray, silent: bool = False) -> float:

    """

    Calculates the restricted Laplace transform MP2 energy.

    This is an implementation of the most efficient quadrature method, Euler-Maclaurin B, with the 
    Chebyshev energy-weighted density matrix formula.
    See M. Kobayashi and H. Nakai, Chem. Phys. Lett., 2006, 420, 250-255 for details. 
    This calculates the MP2 energy as a functional of the RHF density matrix, without 
    reference to the Fock matrix eigenvalues (so suitable for implementation in a linear scaling code).

    Args:
        ERI_AO (array): Electron repulsion integrals in AO basis
        molecular_orbitals (array): Molecular orbitals
        F (array): Fock matrix in AO basis
        n_doubly_occ (int): Number of doubly occupied orbitals
        calculation (Calculation): Calculation object
        P (array): Density matrix in AO basis
        silent (bool, optional): Should anything be printed

    Returns:
        E_MP2 (float): Restricted Laplace MP2 energy
        
    """

    # Removes the factor of two from the RHF density matrix to restore idempotency

    P /= 2

    log_spacer(calculation, silent=silent, start="\n")
    log("            Laplace Transform MP2 Energy", calculation, 1, silent=silent, colour="white")
    log_spacer(calculation, silent=silent)

    log("  Constructing hole density matrix...        ", calculation, 1, end="", silent=silent)

    # This is the unoccupied orbital analogue to the HF density matrix

    Q = scf.construct_hole_density_matrix(molecular_orbitals, n_doubly_occ, 1)

    log("[Done]", calculation, 1, silent=silent)

    tau = calculation.n_MP2_grid_points

    log(f"  Building {tau} point integration grid...      ", calculation, 1, end="", silent=silent)

    k = np.arange(1, tau + 1)

    r = k / (tau + 1)

    # Performs the change of variables for the integration

    s = (r ** 3 - 0.9 * r ** 4) / (1 - r) ** 2 + r ** 2 * np.tan(np.pi * r / 2) 

    # Analytical derivative from Wolfram Alpha

    ds_dr = -r / (1 - r) ** 3 * (r * (-1.8 * r ** 2 + 4.6 * r - 3) + 2 * (r - 1) ** 3 * np.tan(np.pi * r / 2) + np.pi / 2 * r * (r - 1) ** 3 * (1 / (np.cos(np.pi * r / 2) ** 2)))

    # Precomputes antisymmetrised ERI array

    L_AO = 2 * ERI_AO - ERI_AO.swapaxes(1, 3)

    log("[Done]", calculation, 1, silent=silent)

    log("\n  Calculating energy components...           ", calculation, 1, end="", silent=silent)

    f = []

    # Construction of energy-weighted density matrices can not be easily vectorised, more efficient to calculate each e within a loop than through separate contraction

    for i in range(len(s)):

        X = scipy.linalg.expm(s[i] * P @ F) @ P
        Y = scipy.linalg.expm(-s[i] * Q @ F) @ Q

        e = np.einsum("mg,nd,kl,es,gdke,mnls->", X, Y, X, Y, ERI_AO, L_AO, optimize=True)

        f.append(e * ds_dr[i])

    log("[Done]", calculation, 1, silent=silent)

    log("\n  Integrating MP2 energy...                  ", calculation, 1, end="", silent=silent)

    # Uses the quadrature method to integrate the energy components as a functional of s(r).

    E_MP2 = -1 / (tau + 1) * np.sum(f) 

    log("[Done]", calculation, 1, silent=silent)

    log(f"\n  MP2 correlation energy:           {E_MP2:15.10f}", calculation, 1, silent=silent)


    return E_MP2