import sys
import numpy as np
from termcolor import colored

class Primitive_Gaussian:

    """

    Defines a primitive Gaussian.

    """

    def __init__(self, alpha, coeff, coordinates):

        """

        Initialises primitive gaussian class and defines normalisation constant, N.

        Args:
            alpha (float): Gaussian exponent   
            coeff (float): Contraction coefficient
            coordinates (array): Atomic coordinates

        Returns:
            None: Nothing is returned

        """
            
        self.alpha = alpha
        self.coeff = coeff
        self.coordinates = coordinates
        self.N = (2.0 * alpha / np.pi) ** 0.75





def generate_atomic_orbitals(atom, basis, coordinates):
    
    """

    Generates a set of atomic atomic_orbitals for a given basis set, atom type and coordinates.

    Args:
        atom (string): Atom symbol   
        basis (string): Basis set
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """

    # Replaces hyphens and pluses with words and underscores, then looks up the appropriate function to generate atomic atomic_orbitals
    basis = basis.replace("-", "_")
    basis = basis.replace("+", "_plus")
    
    atomic_orbitals = getattr(sys.modules[__name__], f"generate_{basis.lower()}_atomic_orbitals")(atom, coordinates)

    return atomic_orbitals
    




def generate_sto_3g_atomic_orbitals(atom, coordinates):

    """

    Generates a set of STO-3G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """

    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(3.425250914, 0.1543289673, coordinates), 
        Primitive_Gaussian(0.6239137298, 0.5353281423, coordinates), 
        Primitive_Gaussian(0.1688554040, 0.4446345422, coordinates)]]
        
    elif atom == "HE" or atom == "XHE":
        
        atomic_orbitals = [[Primitive_Gaussian(0.6362421394E+01, 0.1543289673E+00, coordinates), 
        Primitive_Gaussian(0.1158922999E+01, 0.5353281423E+00, coordinates), 
        Primitive_Gaussian(0.3136497915E+00, 0.4446345422E+00, coordinates)]]

    else: print_basis_error("STO-3G", atom)

    return atomic_orbitals





def generate_sto_6g_atomic_orbitals(atom,coordinates):

    """

    Generates a set of STO-6G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(0.3552322122E+02, 0.9163596281E-02, coordinates), 
        Primitive_Gaussian(0.6513143725E+01, 0.4936149294E-01, coordinates), 
        Primitive_Gaussian(0.1822142904E+01, 0.1685383049E+00, coordinates),
        Primitive_Gaussian(0.6259552659E+00, 0.3705627997E+00, coordinates), 
        Primitive_Gaussian(0.2430767471E+00, 0.4164915298E+00, coordinates), 
        Primitive_Gaussian(0.1001124280E+00, 0.1303340841E+00, coordinates)]]
        
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.6598456824E+02, 0.9163596281E-02, coordinates), 
        Primitive_Gaussian(0.1209819836E+02, 0.4936149294E-01, coordinates), 
        Primitive_Gaussian(0.3384639924E+01, 0.1685383049E+00, coordinates),
        Primitive_Gaussian(0.1162715163E+01, 0.3705627997E+00, coordinates), 
        Primitive_Gaussian(0.4515163224E+00, 0.4164915298E+00, coordinates), 
        Primitive_Gaussian(0.1859593559E+00, 0.1303340841E+00, coordinates)]]
        
    else: print_basis_error("STO-6G", atom)

    return atomic_orbitals
    




    
def generate_6_31g_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-31G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian( 18.7311370000, 0.0334945995, coordinates), 
        Primitive_Gaussian(2.8253937000, 0.2347269467, coordinates), 
        Primitive_Gaussian(0.6401217000, 0.8137573184, coordinates)],[Primitive_Gaussian( 0.1612778000, 1.000000, coordinates)]]
    
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.3842163400E+02, 0.0401397393, coordinates), 
        Primitive_Gaussian(5.7780300000, 0.2612460970, coordinates), 
        Primitive_Gaussian(1.2417740000, 0.7931846246, coordinates)],[Primitive_Gaussian( 0.2979640000, 1.000000, coordinates)]]
    
    else: print_basis_error("6-31G", atom)

    return atomic_orbitals





def generate_6_31_plusg_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-31+G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian( 18.7311370000, 0.0334945995, coordinates), 
        Primitive_Gaussian(2.8253937000, 0.2347269467, coordinates), 
        Primitive_Gaussian(0.6401217000, 0.8137573184, coordinates)],[Primitive_Gaussian( 0.1612778000, 1.000000, coordinates)]]
    
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.3842163400E+02, 0.0401397393, coordinates), 
        Primitive_Gaussian(5.7780300000, 0.2612460970, coordinates), 
        Primitive_Gaussian(1.2417740000, 0.7931846246, coordinates)],[Primitive_Gaussian( 0.2979640000, 1.000000, coordinates)]]
    
    else: print_basis_error("6-31+G", atom)
    
    return atomic_orbitals





def generate_6_31_plus_plusg_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-31++G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(0.1873113696E+02, 0.3349460434E-01, coordinates), 
        Primitive_Gaussian(0.2825394365E+01, 0.2347269535E+00, coordinates), 
        Primitive_Gaussian(0.6401216923E+00, 0.8137573261E+00, coordinates)],[Primitive_Gaussian(0.1612777588E+00, 1.000000, coordinates)],[Primitive_Gaussian(0.3600000000E-01, 0.1000000000E+01, coordinates)]]
    
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.3842163400E+02, 0.4013973935E-01, coordinates), 
        Primitive_Gaussian(0.5778030000E+01, 0.2612460970E+00, coordinates), 
        Primitive_Gaussian(0.1241774000E+01, 0.7931846246E+00, coordinates)],[Primitive_Gaussian(0.2979640000E+00, 1.000000, coordinates)],[Primitive_Gaussian(0.8600000000E-01, 1.000000, coordinates)]]
    
    else: print_basis_error("6-31++G", atom)

    return atomic_orbitals





def generate_3_21g_atomic_orbitals(atom, coordinates):

    """

    Generates a set of 3-21G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(0.5447178000E+01, 0.1562849787E+00, coordinates), 
        Primitive_Gaussian(0.8245472400E+00, 0.9046908767E+00, coordinates)],[Primitive_Gaussian(0.1831915800E+00, 1.000000, coordinates)]]
    
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.1362670000E+02, 0.1752298718E+00, coordinates), 
        Primitive_Gaussian(0.1999350000E+01 , 0.8934823465E+00, coordinates)],[Primitive_Gaussian(0.3829930000E+00, 1.000000, coordinates)]]
    
    else: print_basis_error("3-21G", atom)

    return atomic_orbitals
    



    
def generate_4_31g_atomic_orbitals(atom, coordinates):

    """

    Generates a set of 4-31G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(0.1873113696E+02, 0.3349460434E-01, coordinates), 
        Primitive_Gaussian(0.2825394365E+01, 0.2347269535E+00, coordinates),Primitive_Gaussian(0.6401216923E+00, 0.8137573261E+00, coordinates)],
        [Primitive_Gaussian(0.1612777588E+00, 1.000000, coordinates)]]
    
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(0.3842163400E+02, 0.4013973935E-01, coordinates), 
        Primitive_Gaussian(0.5778030000E+01, 0.2612460970E+00, coordinates), 
        Primitive_Gaussian(0.1241774000E+01, 0.7931846246E+00, coordinates)],[Primitive_Gaussian(0.2979640000E+00, 1.000000, coordinates)]]
    
    else: print_basis_error("4-31G", atom)

    return atomic_orbitals
    
    



def generate_6_311g_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-311G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(33.86500, 0.0254938, coordinates), 
        Primitive_Gaussian(5.094790, 0.190373, coordinates), 
        Primitive_Gaussian(1.158790,  0.852161, coordinates)],[Primitive_Gaussian(0.325840, 1.000000, coordinates)],[Primitive_Gaussian(0.102741, 1.000000, coordinates)]]
        
        
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(98.12430, 0.0287452, coordinates), 
        Primitive_Gaussian(14.76890, 0.208061, coordinates), 
        Primitive_Gaussian(3.318830,  0.837635, coordinates)],[Primitive_Gaussian(0.874047, 1.000000, coordinates)],[Primitive_Gaussian(0.244564, 1.000000, coordinates)]]

    else: print_basis_error("6-311G", atom)

    return atomic_orbitals
    




def generate_6_311_plusg_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-311+G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """
    
    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(33.86500, 0.0254938, coordinates), 
        Primitive_Gaussian(5.094790, 0.190373, coordinates), 
        Primitive_Gaussian(1.158790,  0.852161, coordinates)],[Primitive_Gaussian(0.325840, 1.000000, coordinates)],[Primitive_Gaussian(0.102741, 1.000000, coordinates)]]
        
        
    elif atom == "HE" or atom == "XHE":
    
        atomic_orbitals = [[Primitive_Gaussian(98.12430, 0.0287452, coordinates), 
        Primitive_Gaussian(14.76890, 0.208061, coordinates), 
        Primitive_Gaussian(3.318830,  0.837635, coordinates)],[Primitive_Gaussian(0.874047, 1.000000, coordinates)],[Primitive_Gaussian(0.244564, 1.000000, coordinates)]]
        
    else: print_basis_error("6-311+G", atom)

    return atomic_orbitals
    



    
def generate_6_311_plus_plusg_atomic_orbitals(atom,coordinates):

    """

    Generates a set of 6-311++G atomic atomic_orbitals.

    Args:
        atom (string): Atom symbol   
        coordinates (array): Atomic coordinates

    Returns:
        atomic_orbitals (array): Array of primitive gaussians

    """

    if atom == "H" or atom == "XH":
    
        atomic_orbitals = [[Primitive_Gaussian(33.86500, 0.0254938, coordinates), 
        Primitive_Gaussian(5.094790, 0.190373, coordinates), 
        Primitive_Gaussian(1.158790, 0.852161, coordinates)],[Primitive_Gaussian(0.325840, 1.000000, coordinates)],[Primitive_Gaussian(0.102741, 1.000000, coordinates)],[Primitive_Gaussian(0.036, 1.000000, coordinates)]]
    
    else: print_basis_error("6-311++G", atom)
    
    return atomic_orbitals





def print_basis_error(basis, atom):

    """

    Prints an error message and exits the calculation if a basis set is not parameterised.

    Args:
        basis (string): Basis set
        atom (string): Atom symbol   

    Returns:
        None: Nothing is returned

    """

    if "X" in atom: 
        
        atom = atom.split("X")[1]
        ghost = "ghost "
    
    else: ghost = ""

    atom = atom.lower().capitalize()

    print(colored(f"\nERROR: The {basis} basis is not parameterised for {ghost}{atom}. Choose another basis set!  :(\n","light_red"))
    sys.exit()
    