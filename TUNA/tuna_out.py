from tuna_util import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tuna_dft as dft
import matplotlib
import pickle
from matplotlib import font_manager as fm
import warnings, logging




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

        plt.savefig(calculation.save_plot_filepath, dpi=1200, figtransparent=True)

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




def build_Cartesian_grid(bond_length):

    bond_length = bond_length if type(bond_length) == float else 0

    extent = 3
    number_of_points = 500

    x = np.linspace(-extent, extent, number_of_points)
    z = np.linspace(-extent, extent + bond_length, number_of_points)

    X, Z = np.meshgrid(x, z, indexing="ij")
    
    grid = np.array([X, Z])

    return grid








def plot_on_two_dimensional_grid(basis_functions_on_grid, grid, bond_length, P=None, molecular_orbitals=None, which_MO=None):

    X, Z = grid 

    fig, ax = plt.subplots()
    ax.axis("off")

    bond_length = bond_length if type(bond_length) == float else 0

    if P is not None:

        density = dft.construct_density_on_grid(P, basis_functions_on_grid)

        density_cut_off = 0.98

        view = np.clip(density, None, np.quantile(density, density_cut_off))

        cmap = LinearSegmentedColormap.from_list("wp", [(1, 1, 1), (1, 0, 1)])

        vmin = 0
        vmax = np.max(view)


    elif molecular_orbitals is not None:

        molecular_orbitals_on_grid = np.einsum("ikl,ij->jkl", basis_functions_on_grid, molecular_orbitals, optimize=True)

        view = molecular_orbitals_on_grid[which_MO]
        
        view *= -1 if np.sign(view[np.unravel_index(np.argmin(X ** 2 + Z ** 2), X.shape)]) < 0 else 1

        cmap = "bwr"

        max_abs = np.max(np.abs(view))
        vmin = -max_abs
        vmax =  max_abs


    im = ax.imshow(view, extent=(Z.min(), Z.max(), X.min(), X.max()), cmap=cmap, vmin=vmin, vmax=vmax)

    ax.scatter([0.0, bond_length],[0.0, 0.0], c="black", s=8, zorder=3)

    plt.tight_layout()
    plt.show()


    return





def calculate_nuclear_electrostatic_potential(grid, bond_length, nuclear_charges):

    X, Z = grid

    Z_A, Z_B = nuclear_charges


    V_nuclear_A = Z_A / np.sqrt(X ** 2 + Z ** 2)
    V_nuclear_B = Z_B / np.sqrt(X ** 2 + (Z - bond_length) ** 2)

    V_nuclear = V_nuclear_A + V_nuclear_B


    return V_nuclear



def calculate_electronic_electrostatic_potential():

    return





def plot_plots(calculation, basis_functions, bond_length, P, P_alpha, P_beta, molecular_orbitals, n_electrons):

    grid = build_Cartesian_grid(bond_length)
    basis_functions_on_grid = dft.construct_basis_functions_on_grid_new(basis_functions, grid)
    
    if calculation.plot_density: 
        
        plot_on_two_dimensional_grid(basis_functions_on_grid, grid, bond_length, P=P)

    if calculation.plot_spin_density: 
        
        plot_on_two_dimensional_grid(basis_functions_on_grid, grid, bond_length, P=P_alpha-P_beta)

    if calculation.plot_HOMO or calculation.plot_LUMO or calculation.plot_molecular_orbital:

        which_MO = calculation.molecular_orbital_to_plot - 1

        if calculation.plot_HOMO: 
            
            which_MO = n_electrons -1 if calculation.reference == "UHF" else n_electrons // 2 - 1

        elif calculation.plot_LUMO: 
            
            which_MO = n_electrons if calculation.reference == "UHF" else n_electrons // 2

        
    if calculation.plot_ESP:

        plot_on_two_dimensional_grid(basis_functions_on_grid, grid, bond_length, P=P)

        
        if calculation.plot_molecular_orbital: 

            which_MO = calculation.molecular_orbital_to_plot - 1

        try:
            
            plot_on_two_dimensional_grid(basis_functions_on_grid, grid, bond_length, molecular_orbitals=molecular_orbitals, which_MO=which_MO)

        except IndexError:

            error("Requested molecular orbital is out of range. Increase basis set size to see more!")


    return