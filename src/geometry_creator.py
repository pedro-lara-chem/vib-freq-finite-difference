import numpy as np
from pathlib import Path

# === Script Overview ===
# This script is THE FIRST STEP in a two-step workflow for calculating vibrational frequencies
# numerically using the finite difference method.
#
# Its primary purpose is to:
# 1. Read a molecular geometry from a reference XYZ file (e.g., "input.xyz").
# 2. Generate a series of new XYZ files. Each file represents the original geometry
#    with a small displacement (+delta or -delta) along one Cartesian coordinate.
#
# These generated XYZ files (e.g., "disp_000_p.xyz", "disp_000_m.xyz") are then
# intended to be used as inputs for external quantum chemistry calculations (e.g., using ORCA)
# to compute the gradient of the energy for each displaced geometry.
#
# The output of this script (displaced XYZ files) and the parameters used (like 'delta')
# are crucial inputs/settings for the second script in this workflow, which
# will use the gradients from the QM calculations to build the Hessian matrix and
# compute vibrational frequencies.

# === Functions for XYZ file manipulation ===
def read_xyz(filename):
    """
    Reads atom symbols and coordinates from a standard XYZ file.

    Args:
        filename (str or Path): The path to the XYZ file.

    Returns:
        tuple: A tuple containing:
            - atoms (list of str): A list of atom symbols (e.g., ['C', 'H', 'H']).
            - coords (np.ndarray): A NumPy array of atomic coordinates (N_atoms, 3) in Angstroms.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())  # First line: number of atoms
    atoms = []
    coords = []
    # Lines from index 2 up to 2 + n_atoms contain atomic data
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])  # Atom symbol
        coords.append([float(x) for x in parts[1:4]])  # X, Y, Z coordinates
    return atoms, np.array(coords)

def write_xyz(filename, atoms, coords, comment=""):
    """
    Writes atom symbols and coordinates to a standard XYZ file.

    Args:
        filename (str or Path): The name of the output XYZ file.
        atoms (list of str): List of atom symbols.
        coords (np.ndarray): NumPy array of atomic coordinates (N_atoms, 3) in Angstroms.
        comment (str, optional): A comment line to be written in the XYZ file (second line).
                                 Defaults to an empty string.
    """
    with open(filename, 'w') as f:
        f.write(f"{int(len(atoms))}\n")  # Number of atoms
        f.write(f"{comment}\n")     # Comment line
        for atom, (x,y,z) in zip(atoms, coords):
            # Write atom symbol and formatted coordinates
            f.write(f"{atom} {x:.8f} {y:.8f} {z:.8f}\n")

# === Function to generate displaced geometries ===
def generate_displacements(atoms, coords, delta_value, output_directory):
    """
    Generates pairs of displaced geometries (+delta_value and -delta_value) along each Cartesian coordinate.
    Each displaced geometry is saved as a separate XYZ file in the specified output directory.

    The naming convention for output files is 'disp_CCC_s.xyz', where:
    - CCC is a zero-padded three-digit index of the Cartesian coordinate (000 to 3N-1).
    - s is 'p' for positive displacement (+delta_value) or 'm' for negative displacement (-delta_value).
    This naming convention is expected by the subsequent frequency calculation script.

    Args:
        atoms (list of str): List of atom symbols.
        coords (np.ndarray): NumPy array of initial atomic coordinates (N_atoms, 3) in Angstroms.
        delta_value (float): The magnitude of the displacement in Angstroms. This value MUST BE
                             USED CONSISTENTLY in the subsequent frequency calculation script.
        output_directory (str or Path): The directory where displaced XYZ files will be saved.
                                        This directory structure might be relevant for the
                                        frequency calculation script.
    """
    n_atoms = len(atoms)
    n_coords = 3 * n_atoms  # Total number of Cartesian coordinates (3N)
    coords_flat = coords.flatten()  # Flatten the (N_atoms, 3) coordinate array to a 1D (3N,) array

    # Create the output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Iterate over each Cartesian coordinate (0 to 3N-1)
    for i in range(n_coords):
        # Create a base displacement vector (all zeros)
        disp_vec = np.zeros(n_coords)
        # Set the displacement for the current i-th coordinate
        disp_vec[i] = delta_value

        # Generate for positive ('p') and negative ('m') displacements
        for sign, label in zip([1, -1], ["p", "m"]):
            # Apply the signed displacement: new_coords = initial_coords +/- displacement_vector
            new_coords_flat = coords_flat + sign * disp_vec
            # Reshape the 1D (3N,) coordinate array back to (N_atoms, 3)
            new_coords_reshaped = new_coords_flat.reshape((-1, 3))

            # Define the output XYZ filename (e.g., disp_000_p.xyz)
            xyz_name = Path(output_directory) / f"disp_{i:03d}_{label}.xyz"
            # Create a comment for the XYZ file indicating the displacement
            comment = (f"Displacement along Cartesian coordinate index {i}{sign} {delta_displacement_angstroms}")
            # Write the new displaced geometry to an XYZ file
            write_xyz(xyz_name, atoms, new_coords_reshaped, comment)

    print(f"Generated {2 * n_coords} displaced geometry XYZ files in '{Path(output_directory).resolve()}/'")


# === Main workflow for Script 1 (Geometry Generation) ===
if __name__ == "__main__":
    # --- User-configurable Parameters ---

    # Name of the input XYZ file containing the reference (equilibrium) geometry.
    # This SAME file will also be read by the frequency calculation script.
    reference_geometry_file = "input.xyz"  # Make sure this file exists!

    # The magnitude of the displacement for finite differences, in Angstroms.
    # This value MUST BE IDENTICAL to the 'delta_angstrom' parameter used in the
    # subsequent frequency calculation script (e.g., frequencies_calculator_final.py).
    delta_displacement_angstroms = 0.005

    # Directory where the generated displaced XYZ files will be saved.
    # The subsequent ORCA calculations should be run on files from this directory.
    output_displaced_xyz_dir = "displaced_geometries_for_orca"

    # This is a SUGGESTED name for the directory where you should save your ORCA output files
    # (e.g., .out files containing gradients) after running calculations on the
    # XYZ files from 'output_displaced_xyz_dir'.
    # The frequency calculation script will need to read from this ORCA output directory.
    suggested_orca_output_dir = "orca_gradient_outputs"


    # --- Script Execution ---
    print(f"--- Script 1: Geometry Generation for Numerical Frequencies ---")
    print(f"Step A: Reading reference geometry from '{reference_geometry_file}'...")
    try:
        atoms, coords = read_xyz(reference_geometry_file)
        print(f"Successfully read {len(atoms)} atoms from '{reference_geometry_file}'.")
    except FileNotFoundError:
        print(f"Error: The reference geometry file '{reference_geometry_file}' was not found.")
        print("Please create this file with your molecule's coordinates in XYZ format.")
        exit()
    except Exception as e:
        print(f"Error reading '{reference_geometry_file}': {e}")
        exit()

    print(f"\nStep B: Generating displaced geometries...")
    print(f"Using displacement delta = {delta_displacement_angstroms} Angstrom.")
    print(f"Outputting displaced XYZ files to: '{output_displaced_xyz_dir}'")
    # Generate and save all displaced XYZ files
    generate_displacements(atoms,
                           coords,
                           delta_value=delta_displacement_angstroms,
                           output_directory=output_displaced_xyz_dir)

    print(f"\n--- Next Steps (Manual ORCA Calculations Required) ---")
    print(f"1. You should now have a set of XYZ files in the directory: '{Path(output_displaced_xyz_dir).resolve()}'")
    print(f"2. For EACH of these XYZ files (e.g., 'disp_000_p.xyz', 'disp_000_m.xyz', etc.):")
    print(f"   a. Run a single-point gradient calculation using ORCA (or similar QM software).")
    print(f"   b. Example ORCA input keywords for a gradient calculation: ! RKS PBE0 def2-SVP EnGrad")
    print(f"3. Save the ORCA output files (e.g., 'disp_000_p.out', 'disp_000_m.out') into a")
    print(f"   DEDICATED directory. We suggest naming this directory: '{suggested_orca_output_dir}'.")
    print(f"   The frequency calculation script (e.g., frequencies_calculator_final.py) will need")
    print(f"   to read these '.out' files from that directory.")
    print(f"4. CRITICAL: The base names of the ORCA output files MUST match the base names of the")
    print(f"   displaced XYZ files. For example, the ORCA output for 'disp_000_p.xyz' must be named 'disp_000_p.out'.")
    print(f"\nAfter completing these ORCA calculations, you can proceed to run the second script")
    print(f"(e.g., frequencies_calculator_final.py) to calculate the vibrational frequencies.")
