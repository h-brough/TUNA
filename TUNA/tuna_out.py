from tuna_util import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tuna_dft as dft
import matplotlib
import pickle
from matplotlib import font_manager as fm
import warnings, logging, os
from numpy import ndarray
from tuna_molecule import Molecule



"""

This is the TUNA module for plotting and saving output files, written first for version 0.9.0. Updated and refactored in version 0.10.0
to include plotting vibrational wavefunctions.

Temporary coordinate scan plots are saved as pickle files. These can be written to several times to overlay multiple potential energy surfaces
on one plot. Densities, orbitals and vibrational wavefunctions can be plotted. Trajectories are written as xyz output files in MD simulations and 
geometry optimisations, if requested.

This module contains:

1. Useful functions for temporary plots (save_and_show_plot, delete_saved_plot)
2. The coordinate scan plotting function (plot_coordinate_scan)
3. The function for plotting vibrational wavefunctions, which shares utilities with the function for coordinate scan plotting (plot_vibrational_wavefunctions)
4. The two functions that enable plotting of densities and orbitals (show_cube_plot, show_two_dimensional_plot)
5. A function to enable printing of trajectories to xyz files (save_trajectory_to_file)

"""




def save_and_show_plot(calculation: Calculation) -> None:

    """

    Saves a plot if requested with "SAVEPLOT", and shows it.

    Args:
        calculation (Calculation): Calculation object

    """

    # Gives plot a consistent look
    plt.tight_layout()

    # Saves the plot if requested
    if calculation.save_plot:

        log("Saving plot as \"{calculation.save_plot_filepath}\"...      ", calculation, 1, end=""); sys.stdout.flush()

        plt.savefig(calculation.save_plot_filepath, dpi=1200)
    
        log("  [Done]", calculation, 1)

    # Shows the plot
    plt.show() 

    return







def delete_saved_plot() -> None:

    """
    
    Deletes a pickle plot, if it exists.
    
    """

    # This file path doesn't need to be changed by the user
    file_path = "TUNA-plot-temp.pkl"

    if os.path.exists(file_path):
        
        os.remove(file_path)

        warning(f"The file {file_path} has been deleted due to the \"DELPLOT\" keyword.\n", space=0)

    else:
        
        warning(f"Plot deletion requested but {file_path} could not be found!\n",space=0)

    return







def suppress_plot_warnings() -> None:

    """
    
    Gets rid of annoying warnings about fonts from Matplotlib.

    """

    # Ignore everything but a severe error
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", module="matplotlib.font_manager")

    # Forces the cache to build to access font list
    _ = fm.fontManager.ttflist

    return








def build_Cartesian_grid(bond_length: float, extent: float = 3, number_of_points: int = 500) -> ndarray:

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








def plot_coordinate_scan(calculation: Calculation, bond_lengths: ndarray, energies: ndarray) -> None:

    """

    Interfaces with Matplotlib to plot energy as a function of bond length.

    Args:
        calculation (Calculation): Calculation object
        bond_lengths (array): List of bond lengths  
        energies (array): List of molecular energies at each bond length

    """

    # Path for the temporary file
    temporary_pickle_path = "TUNA-plot-temp.pkl"

    log("\nPlotting energy profile diagram...      ", calculation, 1, end=""); sys.stdout.flush()
    
    suppress_plot_warnings()

    # Saves temporary file if "ADDPLOT" used
    fig, ax = read_temporary_plot_file(temporary_pickle_path) if calculation.add_plot else plt.subplots(figsize=(10, 6))

    # For excited state calculations, also print the root
    legend_label = f"{calculation.method}/{calculation.basis}" if "CIS" not in calculation.method else f"{calculation.method}/{calculation.basis}, ROOT {calculation.root}"
    linestyle = "--" if calculation.plot_dashed_lines else ":" if calculation.plot_dotted_lines else "-"

    plt.plot(bond_lengths, energies, color=calculation.scan_plot_colour,linewidth=1.75, label=legend_label, linestyle=linestyle)

    format_coordinate_scan_plot(calculation, ax)

    log("[Done]", calculation, 1)

    # If the "ADDPLOT" keyword is used, save the plot to the pickle temporary file
    if calculation.add_plot:

        with open(temporary_pickle_path, "wb") as f:

            pickle.dump(fig, f)

    save_and_show_plot(calculation)

    return








def read_temporary_plot_file(temporary_pickle_path: str) -> tuple[any, any]:

    """

    Reads data from a temporary pickle plot file.

    Args:
        temporary_pickle_path (str): Filepath for temporary file
    
    Returns:
        fig (any): Matplotlib figure
        ax (any): Matplotlib axes

    """

    try:
        
        # Attempt to open a previous temporary file
        with open(temporary_pickle_path, "rb") as f:

            fig = pickle.load(f)
            ax = fig.axes[0]

            plt.figure(fig.number)
            fig.set_size_inches(10, 6, True)
    
    except:

        # If we can't, just open a new plot
        fig, ax = plt.subplots(figsize=(10, 6))    


    return fig, ax







def format_charge(charge: int) -> str:

    """
    
    Turns a molecular charge into a formatted string.

    Args:
        charge (float): Molecular charge

    Returns:
        formatted_charge (str): Formatted charge

    """

    # Singly positive or negative charges are just +/-, higher charges are n+/n-

    match charge:
        
        case 0: return ""
        case 1: return "+"
        case -1: return "-"

    sign = "+" if charge > 0 else "-"

    formatted_charge = f"{abs(charge)}{sign}"

    return formatted_charge








def format_coordinate_scan_plot(calculation: Calculation, ax: any) -> None:

    """

    Sets up the formatting for plotting a coordinate scan.

    Args:
        calculation (Calculation): Calculation object
        ax (any): Matplotlib axes 

    """

    def format_charge(charge: int) -> str:

        # Singly positive or negative charges are just +/-, higher charges are n+/n-

        match charge:
            
            case 0: return ""
            case 1: return "+"
            case -1: return "-"

        sign = "+" if charge > 0 else "-"

        return f"{abs(charge)}{sign}"


    # These are picked in order, if they are present
    plot_font = ["Consolas", "Liberation Mono", "Courier New", "DejaVu Sans"]

    matplotlib.rcParams["font.family"] = plot_font
    font_prop = fm.FontProperties(family=plot_font, size=12)

    # Formats the charge into a nicely readable string
    charge = format_charge(calculation.charge)
    
    plt.xlabel("Bond Length (Angstrom)", fontweight="bold", labelpad=10, fontfamily=plot_font, fontsize=14)
    plt.ylabel("Energy (Hartree)",labelpad=10, fontweight="bold", fontfamily=plot_font, fontsize=14)

    plt.legend(loc="upper right", fontsize=12, frameon=False, handlelength=4, prop=font_prop)

    plt.title(f"TUNA Calculation on "f"{calculation.atomic_symbols[0].capitalize()}—"f"{calculation.atomic_symbols[1].capitalize()}"rf"$^{{{charge}}}$ Molecule", fontweight="bold", fontsize=16, fontfamily=plot_font, pad=15)
    
    # Major and minor ticks
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.25, length=6, direction='out')
    ax.tick_params(axis='both', which='minor', labelsize=11, width=1.25, length=3, direction='out')

    # Set the linewidth of the border 
    for spine in ax.spines.values(): spine.set_linewidth(1.25)
    
    plt.minorticks_on()

    return








def plot_vibrational_wavefunctions(calculation: Calculation, bond_lengths: ndarray, energies: ndarray, vibrational_energy_levels: ndarray, vibrational_wavefunctions: ndarray) -> None:

    """

    Plots the vibrational wavefunctions from an anharmonic frequency calculation.

    Args:
        calculation (Calculation): Calculation object
        x (array): Interpolated bond length
        energies (array): Interpolated potential energies
        vibrational_energy_levels (array): Vibrational energy levels
        vibrational_wavefunctions (array): Vibrational wavefunctions

    """

    suppress_plot_warnings()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plots the vibrational wavefunctions
    for i in range(len(vibrational_energy_levels)):

        # Starts at grey, gets more red with energy
        wavefunction_colour = min(i / (len(vibrational_energy_levels) + 1) + 0.3, 1), 0.3, 0.3

        # Plots each wavefunction at its energy
        vertical_offset = vibrational_energy_levels[i] - min(energies)

        plt.plot(bond_lengths, vibrational_wavefunctions[:, i] + vertical_offset, color=wavefunction_colour)

    # Set the equilibrium bond length to be zero
    energies = energies - min(energies)

    # Makes sure the plot is sensibly scaled - without this the repulsive region is too dominant
    mask = energies < 0.25

    # For excited state calculations, also print the root
    legend_label = f"{calculation.method}/{calculation.basis}" if "CIS" not in calculation.method else f"{calculation.method}/{calculation.basis}, ROOT {calculation.root}"
    linestyle = "--" if calculation.plot_dashed_lines else ":" if calculation.plot_dotted_lines else "-"
    
    # Plots the potential energy surface
    plt.plot(bond_lengths[mask], energies[mask], color="black", linewidth=1.75, label=legend_label, linestyle=linestyle)

    format_coordinate_scan_plot(calculation, ax)

    save_and_show_plot(calculation)

    return









def show_cube_plot(calculation: Calculation, basis_functions_on_grid: ndarray, grid: ndarray, bond_length: float, P: ndarray = None, molecular_orbitals: ndarray = None, which_MO: int = None, transition: bool = False) -> None:

    """
    
    Plots requested quantity (orbitals or density) on a two-dimensional grid and shows the image with Matplotlib.

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
    bond_length = 0 if isinstance(bond_length, str) else bond_length

    suppress_plot_warnings()       
    
    # These are picked in order, if they are present
    plot_font = ["Consolas", "Liberation Mono", "Courier New", "DejaVu Sans"]

    matplotlib.rcParams["font.family"] = plot_font

    # Formats the charge into a nicely readable string
    charge = format_charge(calculation.charge)

    if P is not None:

        if len(calculation.atomic_symbols) == 2:

            plt.title(f"Density from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}—"f"{calculation.atomic_symbols[1].capitalize()}"rf"$^{{{charge}}}$ molecule", fontweight="bold", fontsize=11, fontfamily=plot_font, pad=15)

        else:

            plt.title(f"Density from {calculation.method}/{calculation.basis} calculation on "f"{calculation.atomic_symbols[0].capitalize()}"rf"$^{{{charge}}}$ atom", fontweight="bold", fontsize=11, fontfamily=plot_font, pad=15)

        # Builds density on grid
        density = dft.construct_density_on_grid(P, basis_functions_on_grid, clean_density=False)
        
        # Ignores the extremes of density near the nuclei
        density_cut_off = 0.98

        if transition:
            
            view = np.clip(density, np.quantile(density, 1 - density_cut_off), np.quantile(density, density_cut_off))

            # Difference densities have both positive and negative parts
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
    ax.scatter([0.0, bond_length], [0.0, 0.0], c="black", s=8, zorder=3)

    save_and_show_plot(calculation)

    return








def show_two_dimensional_plot(calculation: Calculation, molecule: Molecule, P: ndarray, P_alpha: ndarray, P_beta: ndarray, P_difference_alpha: ndarray, P_difference_beta: ndarray, P_difference: ndarray, molecular_orbitals: ndarray, natural_orbitals: ndarray) -> None:

    """
    
    Shows the requested two-dimensional plot.

    Args:
        calculation (Calculation): Calculation object]
        molecule (Molecule): Molecule object
        P (array): Density matrix in AO basis
        P_alpha (array): Alpha density matrix in AO basis
        P_beta (array): Beta density matrix in AO basis
        P_difference_alpha (array): Alpha difference density
        P_difference_beta (array): Beta difference density
        P_difference (array): Difference density
        molecular_orbitals (array): Molecular orbitals
        natural_orbitals (array): Natural orbitals
    
    """

    if calculation.method in excited_state_methods:

        # Sets the density matrices to the difference density

        if calculation.plot_difference_density or calculation.plot_difference_spin_density: 

            P = P_difference
            P_alpha = P_difference_alpha
            P_beta = P_difference_beta       
            
    # Build grid and express basis functions on the grid

    grid = build_Cartesian_grid(molecule.bond_length)
    basis_functions_on_grid = dft.construct_basis_functions_on_grid(molecule.basis_functions, grid)

    # Plots electron density

    if calculation.plot_density: 
        
        show_cube_plot(calculation, basis_functions_on_grid, grid, molecule.bond_length, P=P)

    # Plots difference density

    if calculation.plot_difference_density:

        show_cube_plot(calculation, basis_functions_on_grid, grid, molecule.bond_length, P=P, transition=True)

    # Plots spin density

    if calculation.plot_spin_density or calculation.plot_difference_spin_density: 
        
        show_cube_plot(calculation, basis_functions_on_grid, grid, molecule.bond_length, P=P_alpha-P_beta)

    # Plots molecular orbital

    if calculation.plot_HOMO or calculation.plot_LUMO or calculation.plot_molecular_orbital:

        which_MO = calculation.molecular_orbital_to_plot - 1

        # Identifies the index of the HOMO or LUMO if requested

        if calculation.plot_HOMO: 
            
            which_MO =  molecule.n_electrons - 1 if calculation.reference == "UHF" else  molecule.n_electrons // 2 - 1

        elif calculation.plot_LUMO: 
            
            which_MO =  molecule.n_electrons if calculation.reference == "UHF" else  molecule.n_electrons // 2

        try:
            
            show_cube_plot(calculation, basis_functions_on_grid, grid, molecule.bond_length, molecular_orbitals=molecular_orbitals, which_MO=which_MO)

        except IndexError:

            error("Requested molecular orbital is out of range. Increase basis set size to see more!")

    # Plots natural orbital
    
    if calculation.plot_natural_orbital:

        which_MO = calculation.natural_orbital_to_plot - 1

        try:
            
            show_cube_plot(calculation, basis_functions_on_grid, grid, molecule.bond_length, molecular_orbitals=natural_orbitals, which_MO=which_MO)

        except IndexError:

            error("Requested natural orbital is out of range. Increase basis set size to see more!")


    return








def save_trajectory_to_file(molecule: Molecule, energy: float, coordinates: ndarray, trajectory_path: str) -> None:

    """

    Prints trajectory from optimisation or MD simulation to a file.

    Args:   
        molecule (Molecule): Molecule object
        energy (float) : Molecular energy in hartree
        coordinates (array): Atomic coordinates in bohr
        trajectory_path (str): Path to file

    """
    
    with open(trajectory_path, "a") as file:
        
        # Prints number of atoms and energy        
        file.write(f"{len(molecule.atoms)}\n")
        file.write(f"Coordinates from TUNA calculation, E = {energy:.10f}\n")

        coordinates_angstrom = bohr_to_angstrom(coordinates)

        # Prints coordinates
        for i in range(len(molecule.atoms)):

            file.write(f"  {molecule.atomic_symbols[i]}      {coordinates_angstrom[i][0]:6f}      {coordinates_angstrom[i][1]:6f}      {coordinates_angstrom[i][2]:6f}\n")

    file.close()

    return
