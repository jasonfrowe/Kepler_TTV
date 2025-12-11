# Kepler TTV Analysis

A Python toolkit for analyzing Transit Timing Variations (TTVs) in Kepler exoplanet systems. This project provides tools for measuring, analyzing, and visualizing transit timing variations using Kepler photometry data.

## Features

- **Transit Timing Analysis**: Measure and remeasure transit timing variations with improved precision
- **Fourier Analysis**: Decompose light curves using Lomb-Scargle periodograms and pre-whitening
- **BLS Transit Search**: Search for additional transiting planets in the residuals
- **Data Export**: Save timing measurements in standard `.tt` format for further analysis
- **Visualization**: Generate comparison plots of TTV measurements and save as PDF

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Setting Up a Virtual Environment

#### Using VS Code

1. Open the Command Palette: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Search: Type `Python: Create Environment` and select it
3. Choose Type: Select `Venv`
4. Select Interpreter: Choose your Python version (e.g., Python 3.10, 3.11)
5. Wait: VS Code will create a `.venv` folder in your project

#### Using Command Line

```bash
python -m venv bls_env         # Create virtual environment named 'bls_env'
source ./bls_env/bin/activate   # Activate the environment (Linux/Mac)
# or on Windows: bls_env\Scripts\activate
pip install ipykernel           # Install iPython kernel
python -m ipykernel install --user --name=bls_env  # Add Kernel to Jupyter
```

### Installing Dependencies

#### Install bls_cuda Package

Clone and install in editable mode (recommended for development):

```bash
git clone https://github.com/jasonfrowe/bls_cuda.git
cd bls_cuda
pip install -e .
```

#### Install Required Packages

```bash
pip install numpy matplotlib tqdm numba scipy astroquery pandas
```

#### Optional: Jupyter Widgets

For enhanced notebook functionality:

```bash
pip install ipywidgets
```

## Required Packages

- `pytfit5` - Transit fitting and modeling routines
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `numba` - JIT compilation for performance
- `scipy` - Scientific computing
- `astroquery` - Query astronomical databases
- `pandas` - Data manipulation

## Usage

### Quick Start

```python
import numpy as np
import pytfit5.transitPy5 as tpy5
import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep
import pytfit5.transitfit as transitf
import Kepler_TTV as kttv
import pandas as pd

# Load Kepler catalogue
url = "https://kona.ubishops.ca/architecture/KeplerARCH_20231117.csv"
Kepler_cat = pd.read_csv(url)

# Specify target KOI system
KOI = 1599

# Get data for the KOI
koi_df = tpy5.find_koi_rows(KOI, Kepler_cat)
phot, tpy5_inputs = tpy5.get_photometry(koi_df, raw=0)
sol = tpy5.populate_transit_model(Kepler_cat, KOI)
ntt, tobs, omc, omc_err = tpy5.get_timing_data(Kepler_cat, KOI)

# Mark in-transit data
kep.mark_intransit_data(phot, sol, tdurcut=2.0)
```

### Fitting a Transit Model

```python
# Define parameters to fit
params_to_fit = ["rho", "zpt", "t0", "per", "bb", "rdr"]

# Fit the transit model with TTVs
sol_fit = transitf.fitTransitModel(sol, params_to_fit, phot, 
                                   ntt=ntt, tobs=tobs, omc=omc)

# Generate model
tmodel = transitm.transitModel(sol_fit, phot.time, itime=phot.itime, 
                               ntt=ntt, tobs=tobs, omc=omc)
```

### Fourier Analysis

```python
# Lomb-Scargle periodogram
kttv.phot_lombscargle(phot, tmodel)

# Pre-whitening to remove periodic signals
full_model, found = kttv.prewhiten(phot, tmodel, n_iterations=20, max_frequency=48)

# Visualize the full model
kttv.plot_full_model_overlay(phot, tmodel, full_model)
```

### BLS Transit Search

```python
import pytfit5.bls_cpu as gbls

# Set up BLS inputs
gbls_inputs = gbls.gbls_inputs_class()
gbls_inputs.zerotime = 0.0
gbls_inputs.Rstar = tpy5_inputs.mstar
gbls_inputs.Mstar = tpy5_inputs.rstar
gbls_inputs.plots = 1
gbls_inputs.freq1 = 2/(np.max(phot.time) - np.min(phot.time))
gbls_inputs.freq2 = -1  # default 0.5 day period
gbls_inputs.normalize = "iterative_baseline"

# Run BLS on residuals
gbls_ans = gbls.bls(gbls_inputs, 
                    phot.time[(phot.icut == 0)], 
                    phot.flux_f[(phot.icut == 0)] - tmodel[(phot.icut == 0)] - full_model[(phot.icut == 0)])
```

### Measuring TTVs

```python
# Remeasure TTVs with new photometry
pflag = 0   # 0 = predictive, 1 = use previous O-C
pstart = 0  # initial O-C guess

# Subtract stellar variability model
phot.flux_f -= full_model

# Fit TTVs
ntt_new, tobs_new, omc_new, omc_err_new = kttv.fit_ttvs(
    phot, sol_fit, ntt, tobs, omc, pflag=pflag, pstart=pstart
)
```

### Visualization and Data Export

```python
# Plot comparison of old vs new TTVs (with optional PDF save)
kttv.plot_ttv_comparison(ntt, tobs, omc, omc_err, 
                         ntt_new, tobs_new, omc_new, omc_err_new, 
                         koi_df, KOI, savefig=True)

# Save timing data to .tt files
saved_files = kttv.save_timing_data(ntt_new, tobs_new, omc_new, omc_err_new, koi_df)
```

## Output Files

The toolkit generates the following output files in the `TTVs/` directory:

### Timing Data Files (`.tt`)

Format: `koi{KOI:07.2f}.tt` (e.g., `koi0001599.01.tt`)

Three space-delimited columns:
1. Calculated transit time (BJD - 2454900)
2. Observed transit time (BJD - 2454900)
3. Error on observed transit time (days)

Example:
```
120.4567891234 120.4571234567 0.0012345678
125.6789012345 125.6792345678 0.0011234567
```

### Comparison Plots (`.pdf`)

Format: `koi{KOI}_comp.pdf` (e.g., `koi1599_comp.pdf`)

High-resolution (300 dpi) comparison plots showing:
- Original TTV measurements (blue)
- New TTV measurements (red)
- Error bars on all measurements
- Separate panels for each planet in the system

## Main Functions

### `fit_ttvs()`
Measure transit timing variations by fitting individual transit times.

**Parameters:**
- `phot`: Photometry data object
- `sol_fit`: Transit model solution
- `ntt`: Number of transits per planet
- `tobs`: Observed transit times
- `omc`: O-C values
- `pflag`: Flag for predictive vs iterative fitting (0 or 1)
- `pstart`: Starting O-C value

**Returns:** `ntt_new, tobs_new, omc_new, omc_err_new`

### `save_timing_data()`
Save transit timing data to `.tt` files.

**Parameters:**
- `ntt`: Number of transit times
- `tobs`: Calculated transit times
- `omc`: O-C values
- `omc_err`: O-C errors
- `koi_df`: KOI dataframe
- `output_dir`: Output directory (default: `'TTVs'`)

**Returns:** List of saved filenames

### `plot_ttv_comparison()`
Generate comparison plot of TTV measurements.

**Parameters:**
- `ntt, tobs, omc, omc_err`: Original measurements
- `ntt_new, tobs_new, omc_new, omc_err_new`: New measurements
- `koi_df`: KOI dataframe
- `KOI`: KOI system number
- `savefig`: Save plot as PDF (default: `False`)
- `output_dir`: Output directory (default: `'TTVs'`)

### `phot_lombscargle()`
Analyze photometry residuals using Lomb-Scargle periodogram.

### `prewhiten()`
Iteratively remove periodic signals from light curve.

**Parameters:**
- `phot`: Photometry data
- `tmodel`: Transit model
- `n_iterations`: Number of frequencies to remove (default: 20)
- `max_frequency`: Maximum frequency in c/d (default: 48)

**Returns:** `full_model, found_frequencies`

## Workflow Example

See `Kepler_TTV.ipynb` for a complete workflow example that:

1. Loads Kepler catalogue data
2. Retrieves photometry for a target system
3. Fits a transit model including TTVs
4. Performs Fourier analysis to detect stellar variability
5. Searches for additional planets using BLS
6. Remeasures TTVs with improved precision
7. Exports results and generates comparison plots

## Data Sources

- **Kepler Catalogue**: https://kona.ubishops.ca/architecture/KeplerARCH_20231117.csv
- **Photometry**: Processed Kepler light curves from the Kepler Architecture catalogue
- **Timing Data**: Pre-measured transit times from Data Release 25

## Notes

- Time values are in BJD - 2454900 (Barycentric Julian Date minus offset)
- O-C values are typically displayed in minutes for visualization
- The BLS search can take 1-2 minutes depending on frequency range and CPU cores
- Raw photometry (`raw=1`) requires additional detrending before analysis

## Citation

If you use this code in your research, please cite:

- The pytfit5/bls_cuda package: https://github.com/jasonfrowe/bls_cuda
- The Kepler Architecture catalogue: https://kona.ubishops.ca/architecture/

## License

See the bls_cuda repository for license information.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the GitHub repository.
