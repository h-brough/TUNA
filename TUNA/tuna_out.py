from tuna_util import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tuna_dft as dft
import matplotlib
import pickle
from matplotlib import font_manager as fm
import warnings, logging
import os





def delete_saved_plot():

    """
    
    Deletes a pickle plot, if it exists.
    
    """


    file_path = "TUNA-plot-temp.pkl"

    if os.path.exists(file_path):
        
        os.remove(file_path)
        warning(f"The file {file_path} has been deleted due to the DELPLOT keyword.\n",space=0)

    else:
        
        warning(f"Plot deletion requested but {file_path} could not be found!\n",space=0)









def scan_plot(calculation, bond_lengths, energies):

    """

    Interfaces with matplotlib to plot energy as a function of bond length.

    Args:
        calculation (Calculation): Calculation object
        bond_lengths (array): List of bond lengths  
        energies (array): List of energies at each bond length

    Returns:
        None: Nothing is returned

    """

    log("\nPlotting energy profile diagram...      ", calculation, 1, end=""); sys.stdout.flush()
    

    # Suppress warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="matplotlib.font_manager")

    _ = fm.fontManager.ttflist

    plot_font = ["Consolas", "Liberation Mono", "Courier New", "DejaVu Sans"]

    matplotlib.rcParams['font.family'] = plot_font

    # Saves temporary file if ADDPLOT used
    if calculation.add_plot:

        try:
        
            with open("TUNA-plot-temp.pkl", "rb") as f:
                fig = pickle.load(f)
                ax = fig.axes[0]
                plt.figure(fig.number)
                fig.set_size_inches(10, 6, True)
        
        except:

            fig, ax = plt.subplots(figsize=(10,6))    
    
    else: 
        
        fig, ax = plt.subplots(figsize=(10,6))   


    def mag_then_sign(n):

        if n == 1: return '+'
        if n == -1: return '-'
        
        return f"{abs(n)}{'+' if n > 0 else '-'}"


    legend_label = f"{calculation.method}/{calculation.basis}" if "CIS" not in calculation.method else f"{calculation.method}/{calculation.basis}, ROOT {calculation.root}"


    charge = "" if calculation.charge == 0 else mag_then_sign(calculation.charge)
    
    linestyle = "--" if calculation.plot_dashed_lines else ":" if calculation.plot_dotted_lines else "-"

    font_prop = fm.FontProperties(family=plot_font, size=12)

    plt.plot(bond_lengths, energies, color=calculation.scan_plot_colour,linewidth=1.75, label=legend_label, linestyle=linestyle)
    plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily=plot_font,fontsize=14)
    plt.ylabel("Energy (Hartree)",labelpad=10, fontweight="bold", fontfamily=plot_font,fontsize=14)
    plt.legend(loc="upper right", fontsize=12, frameon=False, handlelength=4, prop=font_prop)
    plt.title(f"TUNA Calculation on "f"{calculation.atomic_symbols[0].capitalize()}—"f"{calculation.atomic_symbols[1].capitalize()}"rf"$^{{{charge}}}$ Molecule",fontweight="bold",fontsize=16,fontfamily=plot_font,pad=15)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')
    
    for spine in ax.spines.values(): spine.set_linewidth(1.25)
    
    plt.minorticks_on()
    plt.tight_layout() 

    log("[Done]", calculation, 1)

    if calculation.add_plot:

        with open("TUNA-plot-temp.pkl", "wb") as f:

            pickle.dump(fig, f)

    log("Saving energy profile diagram...      ", calculation, 1, end=""); sys.stdout.flush()
    
    if calculation.save_plot:

        plt.savefig(calculation.save_plot_filepath, dpi=1200)

    log("  [Done]", calculation, 1)

    log(f"\nSaved plot as \"{calculation.save_plot_filepath}\"", calculation, 1)    
    
    # Shows the coordinate scan plot
    plt.show()









def print_trajectory(molecule, energy, coordinates, trajectory_path):

    """

    Prints trajectory from optimisation or MD simulation to file.

    Args:   
        molecule (Molecule): Molecule object
        energy (float) : Final energy
        coordinates (array): Atomic coordinates
        trajectory_path (str): Path to file

    Returns:
        None : This function does not return anything

    """

    atomic_symbols = molecule.atomic_symbols
    
    with open(trajectory_path, "a") as file:
        
        # Prints energy and atomic_symbols
        file.write(f"{len(atomic_symbols)}\n")
        file.write(f"Coordinates from TUNA calculation, E = {energy:.10f}\n")

        coordinates_angstrom = bohr_to_angstrom(coordinates)

        # Prints coordinates
        for i in range(len(atomic_symbols)):

            file.write(f"  {atomic_symbols[i]}      {coordinates_angstrom[i][0]:6f}      {coordinates_angstrom[i][1]:6f}      {coordinates_angstrom[i][2]:6f}\n")

    file.close()









def build_Cartesian_grid(bond_length, extent=3, number_of_points=500):

    """
    
    Builds the Cartesian grid for plotting.

    Args:
        bond-length (float): Bond length in bohr
        extent (float, optional): How far away from the atomic centers should the grid span
        number_of_points (int, optional): Number of points on each grid axis

    Returns:
        grid (array): Two-dimensional grid on which to plot

    """

    # If bond length is not a float, set it to zero for atoms
    if isinstance(bond_length, str):

        bond_length = 0 

    # The molecule always lies along the z axis, so this axis is extended by the bond length
    x = np.linspace(-extent, extent, number_of_points)
    z = np.linspace(-extent, extent + bond_length, number_of_points)

    # No Y axis is needed because of the symmetry of linear molecules
    X, Z = np.meshgrid(x, z, indexing="ij")
    
    grid = np.stack([X, Z], axis=0)

    return grid










def plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, P=None, molecular_orbitals=None, which_MO=None, transition=False):

    """
    
    Plots requested quantity on a two-dimensional grid and shows the image with Matplotlib.

    Args:
        calculation (Calculation): Calculation object
        basis_functions_on_grid (array): Basis functions evaluated on grid
        grid (array): Two-dimensional grid for plotting
        bond_length (float): Bond length in bohr
        P (array, optional): Density matrix
        molecular_orbitals (array): Molecular orbitals
        which_MO (int): Which molecular orbital to print
        nuclear_charges (array): Nuclear relative charges
        transition (bool): Plot transition density or orbitals
    
    """
    
    X, Z = grid 

    fig, ax = plt.subplots()
    ax.axis("off")

    # If bond length is not a float, set it to zero for atoms
    if isinstance(bond_length, str):

        bond_length = 0 

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="matplotlib.font_manager")

    _ = fm.fontManager.ttflist

    plot_font = ["Consolas", "Liberation Mono", "Courier New", "DejaVu Sans"]

    matplotlib.rcParams['font.family'] = plot_font

    def mag_then_sign(n):

        if n == 1: return '+'
        if n == -1: return '-'
        
        return f"{abs(n)}{'+' if n > 0 else '-'}"

    charge = "" if calculation.charge == 0 else mag_then_sign(calculation.charge)
    
    if P is not None:

        if len(calculation.atomic_symbols) == 2:

            plt.title(f"Density from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}—"f"{calculation.atomic_symbols[1].capitalize()}"rf"$^{{{charge}}}$ molecule",fontweight="bold",fontsize=11,fontfamily=plot_font,pad=15)

        else:

            plt.title(f"Density from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}"rf"$^{{{charge}}}$ atom",fontweight="bold",fontsize=11,fontfamily=plot_font,pad=15)

        # Builds density on grid
        density = dft.construct_density_on_grid(P, basis_functions_on_grid, clean_density=False)
        
        # Ignores the extremes of density near the nuclei
        density_cut_off = 0.98

        if transition:
            
            view = np.clip(density, np.quantile(density, 1 - density_cut_off), np.quantile(density, density_cut_off))

            # Difference densities have both positive and negative parts
            #cmap = "bwr"
            cmap = LinearSegmentedColormap.from_list("bwr_247", [(0,0,1), (247/255,)*3, (1,0,0)], 257)
            
            max_abs = np.max(np.abs(view))

            vmin, vmax = -max_abs, max_abs

        else:

            view = np.clip(density, None, np.quantile(density, density_cut_off))

            # Electron density is only positive
            cmap = LinearSegmentedColormap.from_list("wp", [(247/255, 247/255, 247/255), (1, 0, 1)]) 

            vmin, vmax = 0, np.max(view)

        # Shows the image of the plot on the two-dimensional grid
        ax.imshow(view, extent=(Z.min(), Z.max(), X.min(), X.max()), cmap=cmap, vmin=vmin, vmax=vmax)



    if molecular_orbitals is not None:

        if len(calculation.atomic_symbols) == 2:

            plt.title(f"Orbital {which_MO + 1} from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}—"f"{calculation.atomic_symbols[1].capitalize()}"rf"$^{{{charge}}}$ molecule",fontweight="bold",fontsize=11,fontfamily=plot_font,pad=15)

        else:

            plt.title(f"Orbital {which_MO + 1} from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}"rf"$^{{{charge}}}$ atom",fontweight="bold",fontsize=11,fontfamily=plot_font,pad=15)

        # Builds molecular orbitals on grid
        molecular_orbitals_on_grid = np.einsum("ikl,ij->jkl", basis_functions_on_grid, molecular_orbitals, optimize=True)

        # Pickks out a particular molecular orbital
        view = molecular_orbitals_on_grid[which_MO]
        
        # Ensures consistency in colour by setting the sign to positive on the atom centred at the origin
        view *= -1 if np.sign(view[np.unravel_index(np.argmin(X ** 2 + Z ** 2), X.shape)]) < 0 else 1
        
        # Molecular orbitals can be positive or negative in sign
        cmap = "bwr"
        cmap = LinearSegmentedColormap.from_list("bwr_247", [(0,0,1), (247/255,)*3, (1,0,0)], 257)

        max_abs = np.max(np.abs(view))
        vmin, vmax = -max_abs, max_abs
        
        # Shows the image of the plot on the two-dimensional grid
        ax.imshow(view, extent=(Z.min(), Z.max(), X.min(), X.max()), cmap=cmap, vmin=vmin, vmax=vmax)


    # Plots dots for one of both atomic centres
    ax.scatter([0.0, bond_length],[0.0, 0.0], c="black", s=8, zorder=3)


    # Shows the plot
    plt.savefig("test.pdf", transparent=True, dpi=1200)
    plt.tight_layout()
    plt.show()


    return








def show_two_dimensional_plot(calculation, basis_functions, bond_length, P, P_alpha, P_beta, n_electrons, P_difference_alpha=None, P_difference_beta=None, P_difference=None, molecular_orbitals=None, natural_orbitals=None, nuclear_charges=None):

    """
    
    Shows the requested two-dimensional plot.

    Args:
        calculation (Calculation): Calculation object]
        basis_functions (list): List of basis function objects
        bond_length (float): Bond length in bohr
        P (array): Density matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        n_electrons (int): Number of electrons
        P_difference_alpha (array, optional): Alpha difference density
        P_difference_beta (array, optional): Beta difference density
        P_difference (array, optional): Difference density
        molecular_orbitals (array): Molecular orbitals
        natural_orbitals (array): Natural orbitals
        nuclear_charges (list): Relative nuclear charges
    
    """

    if calculation.method in excited_state_methods:

        # Sets the density matrices to the difference density
        if calculation.plot_difference_density or calculation.plot_difference_spin_density: 

            P = P_difference
            P_alpha = P_difference_alpha
            P_beta = P_difference_beta       
            

    # Build grid and express basis functions on the grid
    grid = build_Cartesian_grid(bond_length)
    basis_functions_on_grid = dft.construct_basis_functions_on_grid(basis_functions, grid)

    # Plots electron density
    if calculation.plot_density: 
        
        plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, P=P)

    # Plots difference density
    if calculation.plot_difference_density:

        plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, P=P, transition=True)

    # Plots spin density
    if calculation.plot_spin_density or calculation.plot_difference_spin_density: 
        
        plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, P=P_alpha-P_beta)

    # Plots molecular orbital
    if calculation.plot_HOMO or calculation.plot_LUMO or calculation.plot_molecular_orbital:

        which_MO = calculation.molecular_orbital_to_plot - 1

        # Identifies the index of the HOMO or LUMO if requested
        if calculation.plot_HOMO: 
            
            which_MO = n_electrons - 1 if calculation.reference == "UHF" else n_electrons // 2 - 1

        elif calculation.plot_LUMO: 
            
            which_MO = n_electrons if calculation.reference == "UHF" else n_electrons // 2

        try:
            
            plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, molecular_orbitals=molecular_orbitals, which_MO=which_MO)

        except IndexError:

            error("Requested molecular orbital is out of range. Increase basis set size to see more!")

    # Plots natural orbital
    if calculation.plot_natural_orbital:

        which_MO = calculation.natural_orbital_to_plot - 1

        try:
            
            plot_on_two_dimensional_grid(calculation, basis_functions_on_grid, grid, bond_length, molecular_orbitals=natural_orbitals, which_MO=which_MO)

        except IndexError:

            error("Requested natural orbital is out of range. Increase basis set size to see more!")


    return