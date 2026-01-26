import numpy as np
from pathlib import Path
from scipy.linalg import eigh
import re # For more robust parsing

# --- Script Overview ---
"""
This script is THE SECOND STEP in a two-step workflow for calculating vibrational frequencies
numerically using the finite difference method.

IT ASSUMES YOU HAVE ALREADY:
1.  Generated displaced geometries using a companion script (e.g., 'geometry_creator_final.py').
2.  Performed single-point gradient calculations (e.g., with ORCA) for EACH of these
    displaced geometries.
3.  Saved the output files from these QM calculations (e.g., ORCA '.out' files containing
    Cartesian gradients) in a dedicated directory, following a specific naming convention
    (e.g., 'disp_000_p.out' for the gradient from 'disp_000_p.xyz').

This script performs the following main steps:
1.  Reads the SAME initial molecular geometry (atom symbols and coordinates) from the
    XYZ file that was used in the geometry generation step (e.g., "input.xyz").
2.  Parses Cartesian gradients from the series of ORCA (or similar QM software) output files.
    The directory containing these files and the displacement 'delta' used MUST MATCH
    the parameters from the geometry generation step.
3.  Constructs the Cartesian Hessian matrix using the finite difference method.
4.  Symmetrizes and mass-weights the Hessian matrix.
5.  Projects out translational and rotational modes.
6.  Diagonalizes the projected mass-weighted Hessian to obtain eigenvalues (force constants)
    and eigenvectors (normal modes).
7.  Converts eigenvalues to vibrational frequencies in cm^-1.
8.  Outputs results and can generate animations for normal modes.
"""

# --- Physical Constants ---
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903  # Conversion factor: Angstroms to Bohr
ANGSTROM_PER_BOHR = 0.529177210903        # Conversion factor: Bohr to Angstroms

# Fundamental constants for frequency conversion
HARTREE_TO_J = 4.359744722e-18   # Energy: Hartree to Joules
AMU_TO_KG = 1.66053906660e-27    # Mass: Atomic Mass Units to Kilograms
BOHR_TO_M = ANGSTROM_PER_BOHR * 1e-10 # Length: Bohr radius to meters
C_CMS = 2.99792458e10             # Speed of light in cm/s
PI = np.pi                        # Mathematical constant pi

# Conversion factor for eigenvalues of the mass-weighted Hessian (in Hartree/(amu*Bohr^2))
# to vibrational frequencies in cm^-1.
# Derived from the formula: nu = (1 / (2*pi*c)) * sqrt(k/mu), where k/mu is the eigenvalue.
# Here, eigenvalue units are Eh / (amu * a0^2).
FREQ_CONVERSION_FACTOR_STD = (1.0 / (2.0 * PI * C_CMS)) * np.sqrt(HARTREE_TO_J / (AMU_TO_KG * BOHR_TO_M**2))
# This factor is approximately 5140.48 cm^-1 when eigenvalues are in Hartree/(amu*Bohr^2).

# ORCA's preferred constant for frequency conversion.
# This constant (5140.48) is typically used when eigenvalues of the mass-weighted Hessian
# are in units of aJ/(amu*Angstrom^2) (attoJoules / (amu * Angstrom^2)).
FREQ_CONVERSION_FACTOR_ORCA_STYLE = 5140.48

# To use ORCA's style factor (5140.48), eigenvalues originally in Hartree/(amu*Bohr^2)
# must be converted to aJ/(amu*Angstrom^2). This is the conversion factor for the eigenvalues themselves.
# Conversion: Hartree/(amu*Bohr^2)  ->  (Hartree * HARTREE_TO_J * 1e18) / (amu * (Bohr * BOHR_TO_M * 1e10)^2)
# Factor = (HARTREE_TO_J * 1e18) / (BOHR_TO_M * 1e10)**2
EIGENVALUE_UNIT_CONVERSION_FOR_ORCA_FACTOR = (HARTREE_TO_J * 1e18) / ((ANGSTROM_PER_BOHR * 1e-10 * 1e10)**2)
# This factor is approximately 15.5688. It effectively converts the force constant units
# from Hartree/Bohr^2 to aJ/Angstrom^2.
# It applies to eigenvalues of the mass-weighted Hessian if mass units (amu) are consistent.


# --- Functions ---

def read_xyz(filename):
    """
    Reads atom symbols and Cartesian coordinates from a standard XYZ file.
    This should be the SAME reference geometry file used by the geometry generation script.

    Args:
        filename (str or Path): The path to the XYZ file.

    Returns:
        tuple: A tuple containing:
            - atoms (list of str): A list of atom symbols.
            - coords (np.ndarray): A NumPy array of atomic coordinates (N_atoms, 3) in Angstroms.

    Raises:
        FileNotFoundError: If the specified XYZ file does not exist.
        ValueError: If the XYZ file is malformed (e.g., incorrect number of atoms,
                    truncated data, non-numeric coordinates).
        Exception: For other unexpected errors during file reading.
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip()) # First line: number of atoms
        atoms = []
        coords = []
        # Check if the file has enough lines for header + coordinates
        if len(lines) < 2 + n_atoms:
            raise ValueError(f"XYZ file '{filename}' seems truncated or malformed. "
                             f"Expected at least {2 + n_atoms} lines, found {len(lines)}.")
        # Lines from index 2 up to 2 + n_atoms contain atomic data
        for line_num, line_content in enumerate(lines[2:2 + n_atoms], start=3):
            parts = line_content.split()
            if len(parts) < 4: # Expect atom_symbol, x, y, z
                raise ValueError(f"Malformed coordinate line {line_num} in '{filename}': "
                                 f"'{line_content.strip()}'. Expected 4 parts.")
            atoms.append(parts[0]) # Atom symbol
            try:
                coords.append([float(x) for x in parts[1:4]]) # X, Y, Z coordinates
            except ValueError:
                raise ValueError(f"Non-numeric coordinate found on line {line_num} in '{filename}': "
                                 f"'{line_content.strip()}'")
        return atoms, np.array(coords)
    except FileNotFoundError:
        print(f"Error: XYZ file '{filename}' not found.")
        raise
    except ValueError as ve: # Catch specific errors like int conversion for n_atoms or float for coords
        print(f"Error processing XYZ file '{filename}': {ve}")
        raise
    except Exception as e: # Catch-all for other unexpected issues
        print(f"An unexpected error occurred while reading XYZ file '{filename}': {e}")
        raise

def parse_orca_gradient_3N(output_file, num_atoms):
    """
    Parses the Cartesian gradient vector (dE/dx1, dE/dy1, dE/dz1, dE/dx2, ...)
    from an ORCA output file (typically a '.out' file).
    The gradient is expected to be in Hartree/Bohr.

    Args:
        output_file (str or Path): Path to the ORCA output file (e.g., "disp_000_p.out").
        num_atoms (int): The expected number of atoms, used for validation.

    Returns:
        np.ndarray: A flattened 1D NumPy array of shape (3*num_atoms,) containing
                    the Cartesian gradients in Hartree/Bohr.

    Raises:
        FileNotFoundError: If the ORCA output file is not found.
        ValueError: If the gradient block is not found, is incomplete, or parsing fails.
        Exception: For other unexpected errors.
    """
    gradients = []
    in_gradient_block = False # Flag to indicate if parser is inside the gradient section
    gradient_lines_read = 0   # Counter for gradient lines successfully read

    # Regex to identify and capture gradient lines.
    gradient_line_regex = re.compile(
        r"^\s*\d+\s+[A-Za-z0-9]+(?:\s*:)?\s+"  # Atom index, atom symbol/kind, optional colon
        r"([-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?)\s+"  # dE/dx (group 1)
        r"([-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?)\s+"  # dE/dy (group 2)
        r"([-+]?\d+\.\d+(?:[eEdD][-+]?\d+)?)"    # dE/dz (group 3)
    )

    try:
        with open(output_file, 'r') as f:
            content = f.readlines() # Read all lines for easier processing

        line_iterator = iter(content) # Use an iterator for potentially skipping lines
        for line in line_iterator:
            # Search for the start of the Cartesian gradient block
            if 'CARTESIAN GRADIENT' in line.upper():
                in_gradient_block = True
                gradients = [] # Reset list for this specific block
                gradient_lines_read = 0
                try:
                    next_line_after_header = next(line_iterator).strip()
                    if "-----------" not in next_line_after_header:
                        pass # Assume consumed, next iteration will handle if it was data
                except StopIteration:
                    break # End of file immediately after finding the gradient header.
                continue # Move to the next line to start reading actual gradient data.

            if in_gradient_block:
                match = gradient_line_regex.match(line)
                if match:
                    try:
                        grad_x = float(match.group(1).replace('D', 'e').replace('d', 'e'))
                        grad_y = float(match.group(2).replace('D', 'e').replace('d', 'e'))
                        grad_z = float(match.group(3).replace('D', 'e').replace('d', 'e'))
                        gradients.append([grad_x, grad_y, grad_z])
                        gradient_lines_read += 1
                    except ValueError as ve:
                        print(f"Warning: Could not parse floats from gradient line in {output_file}: {line.strip()} ({ve})")
                        continue # Skip this problematic line

                    if gradient_lines_read == num_atoms:
                        in_gradient_block = False # Done with this gradient block
                        break # Exit the loop over lines
                elif line.strip() == "" and gradient_lines_read > 0 and gradient_lines_read < num_atoms:
                    print(f"Warning: Empty line encountered in gradient block in {output_file} "
                          f"after reading {gradient_lines_read}/{num_atoms} atoms. Assuming end of block.")
                    in_gradient_block = False
                    break
                elif "Difference to translation invariance" in line and gradient_lines_read > 0:
                    print(f"Info: Reached 'Difference to translation invariance' in {output_file}. "
                          f"Assuming end of gradient block.")
                    in_gradient_block = False
                    break

        # Validation after attempting to parse the file
        if gradient_lines_read != num_atoms:
            error_msg_detail = (f"Expected {num_atoms} gradient lines (one per atom) in {output_file}, "
                                f"but found {gradient_lines_read}.")
            if not gradients and gradient_lines_read == 0: # No gradients found at all
                 raise ValueError(f"Failed to parse any gradients from {output_file}. "
                                  "Check for 'CARTESIAN GRADIENT' marker and ensure data follows. " + error_msg_detail)
            raise ValueError("The parsed gradient data is likely incomplete or incorrect. " + error_msg_detail)

        return np.array(gradients).flatten()  # Convert list of lists to a flat (3N,) NumPy array

    except FileNotFoundError:
        print(f"Error: ORCA output file not found: {output_file}")
        raise
    except Exception as e: # Catch-all for other unexpected issues during parsing
        print(f"An unexpected error occurred during parsing of {output_file}: {e}")
        raise


def build_hessian(displacement_files_dir, displacement_delta_angstrom, num_atoms):
    """
    Builds the Cartesian Hessian matrix using finite differences of gradients from ORCA output files.
    Files are expected to be named 'disp_XXX_p.out' and 'disp_XXX_m.out'.

    Args:
        displacement_files_dir (str or Path): Directory containing the ORCA output files
                                              (e.g., "disp_000_p.out"). This directory
                                              should be where results from QM calculations on
                                              displaced geometries were saved.
        displacement_delta_angstrom (float): The finite displacement step size (in Angstroms)
                                             that was used to generate the displaced geometries
                                             by the PREVIOUS script (e.g., geometry_creator_final.py).
                                             This value MUST BE CONSISTENT.
        num_atoms (int): The number of atoms in the molecule.

    Returns:
        np.ndarray: The symmetrized Cartesian Hessian matrix (3N x 3N) in Hartree/Bohr^2.
    """
    # Files are named e.g. disp_000_p.out, disp_000_m.out, disp_001_p.out etc.
    files = sorted(Path(displacement_files_dir).glob("disp_???_[pm].out"))
    if not files:
        raise FileNotFoundError(f"No ORCA output files found in '{displacement_files_dir}' matching the pattern 'disp_???_[pm].out'. "
                                "Please ensure ORCA output files are present, correctly named, and in the specified directory.")

    num_disp_files = len(files)
    expected_num_coords = 3 * num_atoms # Total number of Cartesian coordinates (3N)

    if num_disp_files != expected_num_coords * 2:
         print(f"Warning: Found {num_disp_files} ORCA output files in '{displacement_files_dir}', "
               f"but expected {expected_num_coords * 2} files for {num_atoms} atoms "
               f"({expected_num_coords} Cartesian coordinates, each with +/- displacements). "
               "Missing files can lead to an incorrect Hessian.")

    parsed_gradients = {} # Dictionary to store parsed gradient vectors
    first_grad_vector_len = None # Used to check consistency

    print(f"Attempting to parse gradients from {num_disp_files} ORCA output files in '{displacement_files_dir}'...")
    successful_parses = 0
    for i, file_path in enumerate(files):
        name = file_path.stem # e.g., "disp_000_p"
        try:
            grad_vector = parse_orca_gradient_3N(file_path, num_atoms)

            if grad_vector is None or len(grad_vector) != expected_num_coords:
                error_msg = (f"Gradient vector from {file_path.name} is None or has incorrect length "
                             f"({len(grad_vector) if grad_vector is not None else 'None'} vs expected {expected_num_coords}).")
                print(f"Error: {error_msg}")
                raise ValueError(f"Inconsistent gradient data from {file_path.name}. {error_msg}")

            if first_grad_vector_len is None:
                first_grad_vector_len = len(grad_vector)
            elif len(grad_vector) != first_grad_vector_len:
                critical_error_msg = (f"Inconsistent gradient vector length from {file_path.name} "
                                      f"({len(grad_vector)}) compared to previous ones ({first_grad_vector_len}). Halting.")
                print(f"Critical Error: {critical_error_msg}")
                raise ValueError("Inconsistent gradient vector lengths parsed across files. " + critical_error_msg)

            parsed_gradients[name] = grad_vector
            successful_parses += 1
        except Exception as e:
            print(f"Critical Error: Failed to parse gradients from '{file_path.name}'. Reason: {e}")
            raise # Stop execution

    if successful_parses != num_disp_files:
        raise RuntimeError(f"Only {successful_parses}/{num_disp_files} gradient files were parsed successfully. "
                           "Cannot build a reliable Hessian matrix.")

    ndim = expected_num_coords
    hessian = np.zeros((ndim, ndim)) # Initialize Hessian matrix

    delta_bohr = displacement_delta_angstrom / ANGSTROM_PER_BOHR
    if abs(delta_bohr) < 1e-9: # Check for effectively zero displacement
        raise ValueError("Displacement delta (converted to Bohr) is too close to zero. "
                         "This will cause division by zero in the finite difference formula.")
    print(f"Using displacement delta for finite difference: {displacement_delta_angstrom} Angstrom = {delta_bohr:.8f} Bohr (This value MUST match the geometry generation step)")

    for k_coord_idx in range(ndim): # k_coord_idx iterates from 0 to 3N-1
        g_plus_key = f"disp_{k_coord_idx:03d}_p"
        g_minus_key = f"disp_{k_coord_idx:03d}_m"

        if g_plus_key not in parsed_gradients:
            raise KeyError(f"Missing parsed gradient data for positive displacement: '{g_plus_key}'. "
                           "Ensure ORCA output file was generated, correctly named, and parsed.")
        if g_minus_key not in parsed_gradients:
            raise KeyError(f"Missing parsed gradient data for negative displacement: '{g_minus_key}'. "
                           "Ensure ORCA output file was generated, correctly named, and parsed.")

        g_vector_plus_displacement = parsed_gradients[g_plus_key]
        g_vector_minus_displacement = parsed_gradients[g_minus_key]

        if len(g_vector_plus_displacement) != ndim or len(g_vector_minus_displacement) != ndim:
            raise ValueError(f"Gradient vector length mismatch for displacement coordinate index {k_coord_idx}. "
                             f"g_plus length: {len(g_vector_plus_displacement)}, g_minus length: {len(g_vector_minus_displacement)}, expected: {ndim}.")

        hessian_column_k = (g_vector_plus_displacement - g_vector_minus_displacement) / (2.0 * delta_bohr)
        hessian[:, k_coord_idx] = hessian_column_k

    hessian_symmetrized = 0.5 * (hessian + hessian.T)
    print("Cartesian Hessian matrix built and symmetrized (units: Hartree/Bohr^2).")
    return hessian_symmetrized

# Standard atomic masses in AMU.
atomic_masses_amu_avg = {
    "H": 1.00782503223, "HE": 4.002602, # Using more precise H, He
    "LI": 6.941,  "BE": 9.012182, "B": 10.811,
    "C": 12.00000, "N": 14.00307400443, "O": 15.99491461957, # More precise N, O
    "F": 18.99840316273, "NE": 20.1797, # More precise F
    "NA": 22.98976928, "MG": 24.3050,
    "AL": 26.9815385, "SI": 28.0855,
    "P": 30.973761998, "S": 32.065,
    "CL": 35.453, # Average for Cl
    "AR": 39.948,
    # Add more elements as needed, ensuring keys are uppercase.
}
# Ensure all keys in the dictionary are uppercase for consistent lookup
atomic_masses_amu_avg = {k.upper(): v for k, v in atomic_masses_amu_avg.items()}


def get_mass_vector(atom_symbols):
    """
    Creates a 1D NumPy array (vector) of atomic masses (AMU), where each atom's mass
    is repeated three times (for x, y, z Cartesian components).

    Args:
        atom_symbols (list of str): A list of atom symbols (e.g., ['C', 'H', 'H']).

    Returns:
        np.ndarray: A 1D NumPy array of shape (3*num_atoms,) containing atomic masses
                    in AMU, e.g., [m_C, m_C, m_C, m_H, m_H, m_H, ...].
    """
    mass_vec = []
    for symbol in atom_symbols:
        mass = atomic_masses_amu_avg.get(symbol.upper()) # Use .upper() for case-insensitive lookup
        if mass is None:
            raise ValueError(f"Atomic mass for atom symbol '{symbol}' not found in the "
                             f"`atomic_masses_amu_avg` dictionary. Please add it. "
                             f"Available keys: {list(atomic_masses_amu_avg.keys())}")
        if mass <= 0:
            raise ValueError(f"Atomic mass for '{symbol}' must be positive, but found {mass}.")
        mass_vec.extend([mass, mass, mass])
    return np.array(mass_vec)

def mass_weight_hessian(hessian_hartree_bohr2, atom_symbols):
    """
    Mass-weights the Cartesian Hessian matrix.
    The transformation is: H_mw[i,j] = H_cartesian[i,j] / sqrt(mass_i * mass_j).
    Eigenvalues of this matrix will be in units of Hartree/(amu*Bohr^2).

    Args:
        hessian_hartree_bohr2 (np.ndarray): The Cartesian Hessian matrix (3N x 3N)
                                            in units of Hartree/Bohr^2.
        atom_symbols (list of str): A list of atom symbols.

    Returns:
        np.ndarray: The mass-weighted Hessian matrix (3N x 3N).
    """
    mass_vector_amu = get_mass_vector(atom_symbols) # Shape (3N,)
    inv_sqrt_masses = 1.0 / np.sqrt(mass_vector_amu) # Element-wise 1/sqrt(m_k)
    mw_hessian = hessian_hartree_bohr2 * np.outer(inv_sqrt_masses, inv_sqrt_masses)
    print("Mass-weighted Hessian computed. Eigenvalues will be in Hartree/(amu*Bohr^2).")
    return mw_hessian

def compute_frequencies(mass_weighted_hessian_std_units, use_orca_style_conversion=False):
    """
    Computes vibrational frequencies (in cm^-1) by diagonalizing the mass-weighted Hessian matrix.
    It also returns the eigenvectors (mass-weighted normal modes, L_mw).

    Args:
        mass_weighted_hessian_std_units (np.ndarray): The mass-weighted Hessian matrix (3N x 3N).
            Its eigenvalues are expected to be in Hartree/(amu*Bohr^2).
        use_orca_style_conversion (bool, optional): If True, converts eigenvalues to
            aJ/(amu*Angstrom^2) and uses `FREQ_CONVERSION_FACTOR_ORCA_STYLE`.
            If False (default), uses eigenvalues directly with `FREQ_CONVERSION_FACTOR_STD`.

    Returns:
        tuple: A tuple containing:
            - frequencies_cm_minus_1 (np.ndarray): Sorted array of vibrational frequencies (cm^-1).
            - eigenvectors (np.ndarray): Matrix where columns are the mass-weighted normal modes (L_mw).
    """
    eigenvalues_hartree_units, eigenvectors = eigh(mass_weighted_hessian_std_units)

    if use_orca_style_conversion:
        eigenvalues_for_conversion = eigenvalues_hartree_units * EIGENVALUE_UNIT_CONVERSION_FOR_ORCA_FACTOR
        conversion_factor_to_use = FREQ_CONVERSION_FACTOR_ORCA_STYLE
        print(f"Using ORCA-style frequency conversion: Eigenvalues converted to aJ/(amu*A^2). "
              f"Conversion factor = {conversion_factor_to_use:.4f} cm^-1.")
    else:
        eigenvalues_for_conversion = eigenvalues_hartree_units
        conversion_factor_to_use = FREQ_CONVERSION_FACTOR_STD
        print(f"Using standard frequency conversion: Eigenvalues in Hartree/(amu*Bohr^2). "
              f"Conversion factor = {conversion_factor_to_use:.4f} cm^-1.")

    frequencies_cm_minus_1 = []
    eigenvalue_threshold = 1e-7 # Threshold for eigenvalues to be considered zero

    for eigval in eigenvalues_for_conversion:
        if eigval < -eigenvalue_threshold: # Significantly negative eigenvalue
            freq = -np.sqrt(abs(eigval)) * conversion_factor_to_use
        elif abs(eigval) < eigenvalue_threshold: # Eigenvalue close to zero
            freq = 0.0
        else: # Positive eigenvalue
            freq = np.sqrt(eigval) * conversion_factor_to_use
        frequencies_cm_minus_1.append(freq)

    frequencies_cm_minus_1 = np.array(frequencies_cm_minus_1)
    print("Vibrational frequencies computed (cm^-1).")
    return frequencies_cm_minus_1, eigenvectors

def write_vibrations(filename, atoms, frequencies_cm, normal_modes_mw, initial_coords_angstrom):
    """
    Writes frequencies and normal modes to a file.

    Args:
        filename (str or Path): The name of the output file.
        atoms (list of str): List of atom symbols.
        frequencies_cm (np.ndarray): Array of vibrational frequencies (cm^-1).
        normal_modes_mw (np.ndarray): (3N, 3N) matrix of mass-weighted normal modes (L_mw).
        initial_coords_angstrom (np.ndarray): Equilibrium coordinates (N_atoms, 3) in Angstroms.
    """
    num_atoms = len(atoms)
    num_modes = len(frequencies_cm)
    mass_vector_3N = get_mass_vector(atoms)
    inv_sqrt_mass_vector_3N = 1.0 / np.sqrt(mass_vector_3N)

    # Convert all mass-weighted normal modes to normalized Cartesian displacements
    cartesian_modes = np.zeros_like(normal_modes_mw)
    for i in range(num_modes):
        mode_mw = normal_modes_mw[:, i]
        cart_disp = mode_mw * inv_sqrt_mass_vector_3N
        norm = np.linalg.norm(cart_disp)
        if norm > 1e-9:
            cartesian_modes[:, i] = cart_disp / norm
        else:
            cartesian_modes[:, i] = cart_disp

    try:
        with open(filename, 'w') as f:
            f.write("VIBRATIONAL FREQUENCIES AND NORMAL MODES\n")
            f.write("-----------------------------------------\n\n")

            # Process modes in blocks of 6 (or fewer for the last block)
            for i in range(0, num_modes, 6):
                end_index = min(i + 6, num_modes)
                mode_indices = range(i, end_index)

                # Write mode numbers
                f.write("".join(f"{idx:12d}" for idx in mode_indices) + "\n")
                
                # Write frequencies for the current block
                freq_line = "".join(f"{frequencies_cm[idx]:12.4f}" for idx in mode_indices)
                f.write(freq_line + "\n")
                f.write("-" * len(freq_line) + "\n")

                # Write Cartesian displacements for each atom for the current block of modes
                for atom_idx in range(num_atoms):
                    atom_symbol = atoms[atom_idx]
                    line = f"{atom_idx:4d} {atom_symbol:>2s}: "
                    for mode_idx in mode_indices:
                        disp_x = cartesian_modes[atom_idx * 3, mode_idx]
                        disp_y = cartesian_modes[atom_idx * 3 + 1, mode_idx]
                        disp_z = cartesian_modes[atom_idx * 3 + 2, mode_idx]
                        line += f"{disp_x:8.4f}{disp_y:8.4f}{disp_z:8.4f}  "
                    f.write(line.strip() + "\n")
                f.write("\n\n")
        print(f"Vibrational analysis saved to: {filename}")
    except IOError as e:
        print(f"Error writing output to '{filename}': {e}")



def write_xyz_trajectory(filename, atoms, frames_coords_angstrom):
    """
    Writes a series of molecular coordinate frames to an XYZ trajectory file.

    Args:
        filename (str or Path): The name of the output XYZ trajectory file.
        atoms (list of str): List of atom symbols.
        frames_coords_angstrom (list of np.ndarray): List of (N_atoms, 3) coordinate arrays for frames.
    """
    try:
        with open(filename, 'w') as f:
            num_atoms = len(atoms)
            for frame_idx, coords_angstrom_frame in enumerate(frames_coords_angstrom):
                if coords_angstrom_frame.shape != (num_atoms, 3):
                    raise ValueError(f"Frame {frame_idx} has incorrect coordinate shape: {coords_angstrom_frame.shape}. "
                                     f"Expected shape: ({num_atoms}, 3).")
                f.write(f"{num_atoms}\n") # Number of atoms for this frame
                f.write(f"Normal mode animation frame {frame_idx + 1}\n") # Comment line
                for atom_symbol, (x, y, z) in zip(atoms, coords_angstrom_frame):
                    f.write(f"{atom_symbol} {x:.8f} {y:.8f} {z:.8f}\n") # Atom symbol and coordinates
    except IOError as e:
        print(f"Error writing XYZ trajectory to '{filename}': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in write_xyz_trajectory for '{filename}': {e}")
        raise

def animate_normal_mode(atoms, initial_coords_angstrom, mode_eigenvector_mw,
                        amplitude_angstrom=0.3, num_frames=20, output_xyz_path=None):
    """
    Generates a series of coordinate frames for animating a single normal mode.
    The input `mode_eigenvector_mw` (L_mw) is mass-weighted. Cartesian displacements (X)
    are obtained by un-mass-weighting: X = M^(-1/2) * L_mw.

    Args:
        atoms (list of str): List of atom symbols.
        initial_coords_angstrom (np.ndarray): Equilibrium atomic coordinates (N_atoms, 3) in Angstroms.
        mode_eigenvector_mw (np.ndarray): A single mass-weighted normal mode eigenvector (1D array, 3N).
        amplitude_angstrom (float, optional): Maximum displacement amplitude in Angstroms.
        num_frames (int, optional): Number of frames for one full cycle of motion.
        output_xyz_path (str or Path, optional): If provided, saves the animation trajectory.

    Returns:
        list of np.ndarray: List of coordinate frames, each an (N_atoms, 3) array.
    """
    num_atoms = len(atoms)
    expected_vector_shape = (3 * num_atoms,)
    expected_coords_shape = (num_atoms, 3)

    if mode_eigenvector_mw.shape != expected_vector_shape:
        raise ValueError(f"Mode eigenvector has incorrect shape: {mode_eigenvector_mw.shape}. Expected: {expected_vector_shape}.")
    if initial_coords_angstrom.shape != expected_coords_shape:
        raise ValueError(f"Initial coordinates have incorrect shape: {initial_coords_angstrom.shape}. Expected: {expected_coords_shape}.")

    mass_vector_amu_3N = get_mass_vector(atoms)
    inv_sqrt_mass_vector_3N = 1.0 / np.sqrt(mass_vector_amu_3N)
    cartesian_displacement_vector_flat = mode_eigenvector_mw * inv_sqrt_mass_vector_3N

    norm = np.linalg.norm(cartesian_displacement_vector_flat)
    if norm < 1e-9:
        print("Warning: Normal mode eigenvector has a near-zero norm. Animation might not be meaningful.")
        normalized_cartesian_displacement_flat = cartesian_displacement_vector_flat
    else:
        normalized_cartesian_displacement_flat = cartesian_displacement_vector_flat / norm

    normalized_cartesian_displacement_reshaped = normalized_cartesian_displacement_flat.reshape((num_atoms, 3))

    animation_frames = []
    for i in range(num_frames):
        phase = 2.0 * np.pi * i / num_frames
        current_displacement_coords = amplitude_angstrom * np.sin(phase) * normalized_cartesian_displacement_reshaped
        displaced_coords_angstrom = initial_coords_angstrom + current_displacement_coords
        animation_frames.append(displaced_coords_angstrom)

    if output_xyz_path:
        try:
            write_xyz_trajectory(output_xyz_path, atoms, animation_frames)
            print(f"Normal mode animation trajectory saved to: {output_xyz_path}")
        except Exception as e:
            print(f"Failed to save animation for mode to '{output_xyz_path}': {e}")
    return animation_frames


def animate_all_modes(atoms, initial_coords_angstrom, normal_modes_matrix_mw, frequencies_cm_minus_1,
                      output_directory="animations_all", freq_threshold_cm_minus_1=50.0,
                      amplitude_angstrom=0.3, num_animation_frames=20):
    """
    Animates vibrational modes with frequencies above a threshold.
    `normal_modes_matrix_mw` is (3N, 3N) where columns are mass-weighted eigenvectors (L_mw).

    Args:
        atoms (list of str): List of atom symbols.
        initial_coords_angstrom (np.ndarray): Equilibrium coordinates (N_atoms, 3) in Angstroms.
        normal_modes_matrix_mw (np.ndarray): (3N, 3N) matrix of mass-weighted normal modes.
        frequencies_cm_minus_1 (np.ndarray): 1D array of frequencies (cm^-1).
        output_directory (str or Path, optional): Directory to save animation XYZ files.
        freq_threshold_cm_minus_1 (float, optional): Only modes with |frequency| > threshold are animated.
        amplitude_angstrom (float, optional): Amplitude for animations.
        num_animation_frames (int, optional): Number of frames per animation.
    """
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    num_total_modes = normal_modes_matrix_mw.shape[1]

    if len(frequencies_cm_minus_1) != num_total_modes:
        raise ValueError("Number of frequencies does not match the number of normal modes.")

    animated_modes_count = 0
    print(f"\nStarting animation for modes with |frequency| > {freq_threshold_cm_minus_1} cm^-1...")
    for mode_idx in range(num_total_modes):
        frequency = frequencies_cm_minus_1[mode_idx]

        if abs(frequency) < freq_threshold_cm_minus_1:
            continue

        print(f"Animating mode {mode_idx} (Frequency: {frequency:.2f} cm^-1)...")
        mode_eigenvector_mw = normal_modes_matrix_mw[:, mode_idx] # Get L_mw for current mode

        freq_label = f"{abs(frequency):.0f}"
        if frequency < -1e-6 : # Small tolerance for negative (imaginary) frequencies
             freq_label = f"i{abs(frequency):.0f}"

        output_xyz_filename = Path(output_directory) / f"mode_{mode_idx:03d}_freq_{freq_label}cm.xyz"

        try:
            animate_normal_mode(
                atoms=atoms, initial_coords_angstrom=initial_coords_angstrom,
                mode_eigenvector_mw=mode_eigenvector_mw, amplitude_angstrom=amplitude_angstrom,
                num_frames=num_animation_frames, output_xyz_path=output_xyz_filename
            )
            animated_modes_count +=1
        except Exception as e:
            print(f"Error animating mode {mode_idx} (frequency {frequency:.2f} cm^-1): {e}")

    if animated_modes_count == 0:
        print(f"No modes were animated. All mode frequencies were below the threshold of {freq_threshold_cm_minus_1} cm^-1, "
              "or no modes met the criteria.")
    else:
        print(f"Finished animating {animated_modes_count} modes. "
              f"Animation trajectory files saved in '{output_directory}'.")


# --- Helper Function to Project Out T/R Modes ---
def is_linear_molecule(coords_bohr, tol=1e-3):
    """
    Determines if a molecule is linear based on SVD of centered coordinates.

    Args:
        coords_bohr (np.ndarray): Atomic coordinates (N_atoms, 3) in Bohr.
        tol (float, optional): Tolerance for singular values to be considered zero.

    Returns:
        bool: True if the molecule is considered linear, False otherwise.
    """
    from numpy.linalg import svd  # Local import
    num_atoms = coords_bohr.shape[0]

    # Handle cases with fewer than 3 atoms first to avoid index errors.
    if num_atoms < 2:
        return False  # A single atom is not considered linear.
    if num_atoms == 2:
        return True   # A diatomic molecule is always linear.

    # For molecules with 3 or more atoms, proceed with the SVD check.
    coords_centered = coords_bohr - np.mean(coords_bohr, axis=0)
    _, s, _ = svd(coords_centered)  # s contains 3 singular values, sorted descending.
    
    # For a molecule to be linear, its shape must be a "rod", meaning two of its
    # three spatial dimensions are negligible. This corresponds to the 2nd and 3rd
    # singular values being close to zero. Since s is sorted, checking s[1] is sufficient.
    return s[1] < tol

def project_out_translations_rotations(hessian_mw, atoms, initial_coords_angstrom):
    """
    Projects translational and rotational components out of the mass-weighted Hessian matrix.

    Args:
        hessian_mw (np.ndarray): The mass-weighted Hessian matrix (3N x 3N).
        atoms (list of str): List of atom symbols.
        initial_coords_angstrom (np.ndarray): Equilibrium atomic coordinates (N_atoms, 3) in Angstroms.

    Returns:
        np.ndarray: The mass-weighted Hessian matrix with T/R components projected out.
    """
    num_atoms = len(atoms)
    ndim = 3 * num_atoms
    masses_3N = get_mass_vector(atoms) # (3N,) AMU
    sqrt_masses_3N = np.sqrt(masses_3N) # (3N,) sqrt(AMU)
    coords_bohr_N_3 = initial_coords_angstrom / ANGSTROM_PER_BOHR # (N_atoms, 3) in Bohr

    tr_vectors_basis = []
    # Construct Mass-Weighted Translation Vectors
    for axis_idx in range(3): # 0 for x, 1 for y, 2 for z
        vec = np.zeros(ndim)
        vec[axis_idx::3] = sqrt_masses_3N[axis_idx::3]
        tr_vectors_basis.append(vec)

    # Construct Mass-Weighted Rotation Vectors
    masses_N = masses_3N[::3] # (N_atoms,)
    total_mass = np.sum(masses_N)
    center_of_mass_bohr = np.sum(coords_bohr_N_3.T * masses_N, axis=1) / total_mass
    centered_coords_bohr_N_3 = coords_bohr_N_3 - center_of_mass_bohr

    if is_linear_molecule(coords_bohr_N_3):
        print("Linear molecule detected by is_linear_molecule() — projecting out 2 rotational modes.")
        # For linear molecules, typically 2 rotational axes are defined.
        # This might need adjustment for arbitrarily oriented linear molecules.
        # A robust method involves inertia tensor eigenvectors. Here, fixed axes are used for simplicity.
        rotation_axes_definitions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # Rotations about x and y
    else:
        print("Non-linear molecule detected — projecting out 3 rotational modes.")
        rotation_axes_definitions = np.eye(3) # Rotations about x, y, and z

    for rot_axis_definition_vec in rotation_axes_definitions:
        atomic_displacement_vectors_due_to_rotation = np.cross(rot_axis_definition_vec, centered_coords_bohr_N_3)
        rot_vec_mw = (atomic_displacement_vectors_due_to_rotation * sqrt_masses_3N.reshape(num_atoms, 3)).flatten()
        tr_vectors_basis.append(rot_vec_mw)

    # Perform Projection
    basis_matrix_B = np.stack(tr_vectors_basis, axis=1)
    Q_orthonormal_tr_basis, _ = np.linalg.qr(basis_matrix_B)
    projector_onto_vibrational_space = np.eye(ndim) - (Q_orthonormal_tr_basis @ Q_orthonormal_tr_basis.T)
    hessian_projected = projector_onto_vibrational_space @ hessian_mw @ projector_onto_vibrational_space
    hessian_projected_symmetrized = 0.5 * (hessian_projected + hessian_projected.T)

    print(f"Projected out {Q_orthonormal_tr_basis.shape[1]} translational and rotational modes from the mass-weighted Hessian.")
    return hessian_projected_symmetrized


# --- Main Execution Block for Script 2 (Frequency Calculation) ---
if __name__ == "__main__":
    # --- Configuration ---
    # Path to the input XYZ file containing the MOLECULAR GEOMETRY.
    # CRITICAL: This MUST be the SAME 'input.xyz' (or equivalent) file that was used as the
    # reference for the 'geometry_creator_final.py' script (or whichever script generated
    # the displaced XYZs for ORCA calculations).
    input_xyz_file = "input.xyz"

    # Finite displacement step size (in Angstroms) used for generating gradient files.
    # CRITICAL: This value MUST EXACTLY MATCH the 'delta_displacement_angstroms' (or equivalent)
    # parameter used in the 'geometry_creator_final.py' script.
    # Any mismatch will lead to incorrect Hessian and frequencies.
    delta_angstrom = 0.005 # This should match 'delta_displacement_angstroms' in Script 1

    # Directory containing the ORCA OUTPUT FILES (e.g., 'disp_000_p.out', 'disp_000_m.out', etc.).
    # These files are the result of running ORCA (or other QM software) single-point gradient
    # calculations on EACH of the displaced XYZ geometries generated by the first script.
    # The naming convention ('disp_XXX_[pm].out') is expected.
    orca_outputs_dir = "orca_outputs" # This should match 'suggested_orca_output_dir' from Script 1

    # --- Optional Settings ---
    USE_ORCA_STYLE_FREQ_CONVERSION = False # See constant definitions for details
    # Threshold for animating modes (cm^-1). Set to 0.0 to animate (almost) all non-zero.
    ANIMATION_FREQ_THRESHOLD_CM_MINUS_1 = 0.0
    ANIMATION_AMPLITUDE_ANGSTROM = 0.5 # Amplitude for normal mode animations
    ANIMATION_OUTPUT_DIR = "calculated_mode_animations" # Output directory for animation files


    # --- Workflow ---
    print(f"--- Script 2: Vibrational Frequency Calculation ---")
    print(f"Step 1: Reading reference molecular geometry from '{input_xyz_file}'...")
    try:
        atom_symbols, initial_coords_angstrom = read_xyz(input_xyz_file)
        num_atoms = len(atom_symbols)
        print(f"Successfully read geometry for {num_atoms} atoms.")
    except Exception as e:
        print(f"Fatal Error: Could not read input geometry file '{input_xyz_file}'. {e}")
        exit()

    print(f"\nStep 2: Building Cartesian Hessian from ORCA gradients in '{orca_outputs_dir}'...")
    print(f"Using displacement delta = {delta_angstrom} Angstrom (this MUST match the value used in geometry generation).")
    try:
        hessian_hartree_bohr2 = build_hessian(orca_outputs_dir, delta_angstrom, num_atoms)
    except Exception as e:
        print(f"Fatal Error: Could not build Hessian matrix. {e}")
        print("Please check if ORCA output files exist in the specified directory, are correctly named, ")
        print("and if the 'delta_angstrom' parameter matches the one used for geometry displacement.")
        exit()

    print("\nStep 3: Mass-weighting the Hessian matrix...")
    try:
        mass_weighted_hessian = mass_weight_hessian(hessian_hartree_bohr2, atom_symbols)
    except Exception as e:
        print(f"Fatal Error: Could not mass-weight Hessian. {e}")
        exit()

    print("\nStep 4: Projecting out translations and rotations from mass-weighted Hessian...")
    try:
        # Use the initial coordinates (in Angstroms) for defining T/R vectors during projection
        mass_weighted_hessian_projected = project_out_translations_rotations(
            mass_weighted_hessian, atom_symbols, initial_coords_angstrom
        )
    except Exception as e:
        print(f"Warning: Could not project out T/R modes due to an error: {e}")
        print("Proceeding with the unprojected mass-weighted Hessian. Expect T/R modes in results.")
        mass_weighted_hessian_projected = mass_weighted_hessian # Fallback to unprojected

    print("\nStep 5: Computing vibrational frequencies and normal modes...")
    try:
        # Use the (potentially projected) Hessian for frequency calculation
        frequencies_cm, normal_modes_eigenvectors_mw = compute_frequencies(
            mass_weighted_hessian_projected,
            use_orca_style_conversion=USE_ORCA_STYLE_FREQ_CONVERSION
        )
    except Exception as e:
        print(f"Fatal Error: Could not compute frequencies. {e}")
        exit()

    print("\n--- Calculated Vibrational Frequencies (cm^-1) ---")
    # Determine expected number of T/R modes for labeling
    num_expected_tr_modes = 6 # Default for non-linear
    if num_atoms == 1:
        num_expected_tr_modes = 3 # Only translations for a single atom
    elif num_atoms > 1: # For multi-atom systems, check linearity
        # Convert Angstrom coords to Bohr for is_linear_molecule function if needed by its internal logic
        coords_bohr_for_linearity_check = initial_coords_angstrom / ANGSTROM_PER_BOHR
        if is_linear_molecule(coords_bohr_for_linearity_check):
            num_expected_tr_modes = 5 # 3T + 2R for linear molecules

    for i, freq in enumerate(frequencies_cm):
        mode_type = ""
        # Heuristic labeling for T/R modes (those appearing first and having low frequency)
        if i < num_expected_tr_modes and abs(freq) < ANIMATION_FREQ_THRESHOLD_CM_MINUS_1 + 50.0: # Add a bit of margin
            mode_type = "(Potential T/R Mode)"
        print(f"Mode {i:3d}: {freq:12.6f} cm^-1 {mode_type}")

    # --- Saving Results ---
    print("\nStep 6: Saving results...")
    try:
        # Save frequencies in the same format as printed to terminal
        output_freq_file = "frequencies_calculated.txt"
        header_info = (
            f"Calculated Frequencies (cm^-1) using "
            f"{'ORCA-style' if USE_ORCA_STYLE_FREQ_CONVERSION else 'Standard'} conversion.\n"
            f"Reference XYZ: {input_xyz_file}, Delta: {delta_angstrom} Angstrom\n"
            f"----------------------------------------------------------\n"
        )
        with open(output_freq_file, 'w') as f:
            f.write(header_info)
            for i, freq in enumerate(frequencies_cm):
                mode_type = ""
                if i < num_expected_tr_modes and abs(freq) < 50.0:
                    mode_type = "(Potential T/R Mode)"
                f.write(f"Mode {i:3d}: {freq:12.6f} cm^-1 {mode_type}\n")
        print(f"Formatted frequencies saved to: {output_freq_file}")

        # Save frequencies and  cartesian normal modes 
        output_file = "vibrations.txt"
        write_vibrations(
            output_file,
            atom_symbols,
            frequencies_cm,
            normal_modes_eigenvectors_mw,
            initial_coords_angstrom
        )
        # Save frequencies and  normal modes mw
        output_freq_modes_file = "frequencies_and_normal_modes_mw.txt"
        num_modes = len(frequencies_cm)
        num_components = normal_modes_eigenvectors_mw.shape[0]

        with open(output_freq_modes_file, 'w') as f:
            # Write any initial header information you have
            f.write(header_info + "\n\n")

            # Process modes in chunks of 5
            for i in range(0, num_modes, 5):
                chunk_end = min(i + 5, num_modes)
                current_indices = range(i, chunk_end)
                current_frequencies = frequencies_cm[i:chunk_end]
                # Get the columns for the current chunk of modes
                current_modes_chunk = normal_modes_eigenvectors_mw[:, i:chunk_end]

                # --- Write headers for the current chunk ---
                # Right-align headers for each mode in the chunk
                f.write("Mode Index".ljust(15) + "".join(f"{idx:<18}" for idx in current_indices) + "\n")
                f.write("Frequency (cm⁻¹)".ljust(15) + "".join(f"{freq:<18.4f}" for freq in current_frequencies) + "\n")
                f.write("-" * (15 + 18 * len(current_indices)) + "\n")

                # --- Write normal mode vector components row by row ---
                f.write("Normal Modes (L_mw)\n")
                for j in range(num_components):
                    # Get the j-th component from each mode in the chunk
                    row_data = current_modes_chunk[j, :]
                    # Format the line with the component index and the corresponding values from each mode
                    f.write(f"Component {j:<7}".ljust(15) + "".join(f"{val:<18.6e}" for val in row_data) + "\n")
                
                # Add space between paragraphs/chunks
                f.write("\n\n")

        print(f"Frequencies and mass-weighted normal modes saved to: {output_freq_modes_file}")
        # Save mass-weighted normal modes as a numpy array for advanced use
        output_modes_file = "normal_modes_mw_calculated.npy"
        np.save(output_modes_file, normal_modes_eigenvectors_mw)
        print(f"Mass-weighted normal modes (L_mw) saved to: {output_modes_file}")

    except Exception as e:
        print(f"Error saving results: {e}")


    # --- Generate Animations ---
    print(f"\nStep 7: Generating normal mode animations (threshold for animation: {ANIMATION_FREQ_THRESHOLD_CM_MINUS_1} cm^-1)...")
    try:
        animate_all_modes(
            atoms=atom_symbols,
            initial_coords_angstrom=initial_coords_angstrom,
            normal_modes_matrix_mw=normal_modes_eigenvectors_mw, # Pass the mass-weighted eigenvectors
            frequencies_cm_minus_1=frequencies_cm,
            output_directory=ANIMATION_OUTPUT_DIR,
            freq_threshold_cm_minus_1=ANIMATION_FREQ_THRESHOLD_CM_MINUS_1,
            amplitude_angstrom=ANIMATION_AMPLITUDE_ANGSTROM
        )
    except Exception as e:
        print(f"Error during animation generation: {e}")

    print("\n--- Script Finished ---")
