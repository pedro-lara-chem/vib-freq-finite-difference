# vib-freq-finite-difference
A two-step Python workflow for calculating molecular vibrational frequencies numerically. This is useful when analytic Hessians are unavailable or too costly in your Quantum Chemistry software (e.g., ORCA, Gaussian).
The workflow calculates the **Hessian matrix** (second derivative of energy) by performing finite differences on **gradients** (first derivatives) obtained from single-point QM calculations.

## Workflow Overview

1.  **Geometry Generation (`geometry_creator.py`):**
    Reads an equilibrium geometry and generates $6N$ displaced geometries (positive and negative displacements along X, Y, Z for every atom).
2.  **QM Calculation (External):**
    You run single-point gradient calculations (e.g., using ORCA) on all generated geometries.
3.  **Frequency Analysis (`frequencies_calculator.py`):**
    Parses the gradients from the QM outputs, constructs the Hessian, mass-weights it, projects out rotations/translations, and computes frequencies/normal modes.

## Prerequisites

- Python 3.x
- `numpy`
- `scipy`

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/vib-freq-finite-difference.git](https://github.com/YOUR_USERNAME/vib-freq-finite-difference.git)
   cd vib-freq-finite-difference
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
Step 1: Generate Displaced Geometries
1. Place your optimized equilibrium geometry file (e.g., input.xyz) in the project root.

2. Open geometry_creator.py and configure the User-configurable Parameters section at the bottom:
   ```python
   reference_geometry_file = "input.xyz"
delta_displacement_angstroms = 0.005  # Step size (Crucial!)
output_displaced_xyz_dir = "displaced_geometries_for_orca"
```
3. Run the script:
```bash
python geometry_creator.py
´´´
Output: A directory containing files like disp_000_p.xyz, disp_000_m.xyz, etc.
Step 2: Run External QM Calculations
For every XYZ file generated in Step 1, run a single-point gradient calculation.

* Software: Any (ORCA is natively supported by the parser).

* Task: Gradient (e.g., ! EnGrad in ORCA).

* Naming: The output files must match the XYZ names (e.g., disp_000_p.out).

* Location: Collect all output files into a single directory (e.g., orca_outputs/).

Step 3: Calculate Frequencies
Ensure the orca_outputs/ directory contains all the .out files from Step 2.

Open frequencies_calculator.py and configure the settings at the bottom. Crucially, the delta_angstrom must match Step 1.
```python
input_xyz_file = "input.xyz"       # Must be the SAME file as Step 1
delta_angstrom = 0.005             # Must match Step 1 exactly
orca_outputs_dir = "orca_outputs"  # Directory containing your .out files
```
3. Run the script:
   ```bash
   python frequencies_calculator.py
   ```
## Configuration
Both scripts are configured by editing the variables in the if __name__ == "__main__": block at the bottom of the files.

## Outputs
The analysis script generates the following files:

* frequencies_calculated.txt: Simple list of frequencies.

* vibrations.txt: Human-readable format containing frequencies and normal mode displacements.

* calculated_mode_animations/*.xyz: Trajectory files for each normal mode. These can be visualized in VMD, Chimera, or Jmol to see the vibration.

* normal_modes_mw_calculated.npy: NumPy binary file containing the mass-weighted normal modes for further programmatic analysis.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
