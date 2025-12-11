import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep
import pytfit5.transitfit as transitf
from tqdm.auto import trange as _tqdm_trange  # may warn if widgets missing
from pytfit5.transitPy5 import pad_list_of_arrays 

# Robust local alias that falls back gracefully if tqdm is unavailable
try:
    _trange = _tqdm_trange
except Exception:  # pragma: no cover
    try:
        from tqdm import trange as _trange  # console fallback
    except Exception:
        def _trange(n):
            return range(n)

def fit_ttvs(phot, sol_fit, ntt=-1, tobs=-1, omc=-1, pflag = 0, pstart = 0):

    if isinstance(pflag, int):
        pflag = np.ones((sol_fit.npl),dtype=int)*pflag

    if isinstance(pstart, int):
        pstart = np.ones((sol_fit.npl),dtype=int)*pstart
    
    flux_f_copy = np.copy(phot.flux_f)
    
    Tmin = np.min(phot.time[(phot.icut == 0)])
    Tmax = np.max(phot.time[(phot.icut == 0)])
    
    tt_list      = []
    for nplanet in _trange(sol_fit.npl):
    
        #Zero out the current planet 
        rdr_copy = sol_fit.rdr[nplanet]
        sol_fit.rdr[nplanet] = 0
        #model with only other planets
        tmodel = transitm.transitModel(sol_fit, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
        sol_fit.rdr[nplanet] = rdr_copy
    
        #Make model for single planet 
        sol_c     = transitm.transit_model_class()
        # Parameters that define the star/scene for the transit model
        sol_c.rho = sol_fit.rho    
        sol_c.nl1 = sol_fit.nl1   
        sol_c.nl2 = sol_fit.nl2
        sol_c.nl3 = sol_fit.nl3   
        sol_c.nl4 = sol_fit.nl4  
        sol_c.t0  = [sol_fit.t0[nplanet]]             # Center of transit time (days)
        sol_c.per = [sol_fit.per[nplanet]]            # Orbital Period (days)
        sol_c.bb  = [sol_fit.bb[nplanet]]                      # Impact parameter
        sol_c.rdr = [sol_fit.rdr[nplanet]]  # Rp/R*
        sol_c.ecw = [0.0]                      # sqrt(e)cos(w)
        sol_c.esw = [0.0]                      # sqrt(e)sin(w)
        sol_c.krv = [0.0]                      # RV amplitude (m/s)
        sol_c.ted = [0.0]                     # thermal eclipse depth (ppm)
        sol_c.ell = [0.0]                      # Ellipsodial variations (ppm)
        sol_c.alb = [0.0]                      # Albedo amplitude (ppm)
        sol_c.npl = 1
        
        #Get duration of the current planet
        tdur = kep.transitDuration(sol_fit, nplanet)
        
        phot.flux_f = flux_f_copy - tmodel + 1
    
        # T0=sol(9)                +int((     Tmin-sol(9)                 )/sol(10)               +0.0d0)*sol(10)
        T0 = sol_fit.t0[nplanet]+np.floor((Tmin-sol_fit.t0[nplanet])/sol_fit.per[nplanet]+0.0  )*sol_fit.per[nplanet]

        cal_omc     = pstart[nplanet]
        cal_omc_old = 0.0
        d_cal_omc   = 0.0
    
        tt      = []
        # print(pflag[nplanet])
        while(T0 < Tmax):

            Ts = T0 - 2.0*tdur + cal_omc         + d_cal_omc
            Te = T0 + 2.0*tdur + cal_omc         + d_cal_omc
            Ts2= T0 - 0.5*tdur + cal_omc - 0.021 + d_cal_omc  # add 30-mins
            Te2= T0 + 0.5*tdur + cal_omc + 0.021 + d_cal_omc  
            sol_c.t0[0] = T0   + cal_omc         + d_cal_omc
    
            params_to_fit = ["t0"]
            phot.tflag = np.zeros((phot.time.shape[0]))
            phot.tflag[(phot.time >= Ts) & (phot.time <= Te)] = 1
            k =  len(phot.time[(phot.time >= Ts2) & (phot.time <= Te2)])
            # print(T0, k)
            if k > 3:
                
                cal_omc_old = cal_omc 
                
                sol_c_fit = transitf.fitTransitModel(sol_c, params_to_fit, phot)

                cal_omc = sol_c_fit.t0[0] - T0
                d_cal_omc = cal_omc - cal_omc_old
    
                # Compute reduced chi-square to scale error for stellar noise
                tmodel_single = transitm.transitModel(sol_c_fit, phot.time, itime=phot.itime)
                in_transit = (phot.time >= Ts2) & (phot.time <= Te2) & (phot.icut == 0)
                residuals = phot.flux_f[in_transit] - tmodel_single[in_transit]
                chi2 = np.sum((residuals / phot.ferr[in_transit])**2)
                dof = np.sum(in_transit) - 1  # 1 fitted parameter (t0)
                reduced_chi2 = chi2 / dof if dof > 0 else 1.0
                
                # Scale error by sqrt(reduced_chi2) to account for underestimated photometric errors
                scaled_error = sol_c_fit.dt0[0] * np.sqrt(max(1.0, reduced_chi2))
                
                tt.append([T0, cal_omc, scaled_error])
    
            else:
                cal_omc += d_cal_omc
    
            T0 = T0 + sol_fit.per[nplanet]
    
            if pflag[nplanet] == 0:
                cal_omc   = 0.0  # check if we are using predictive
                d_cal_omc = 0.0

            # print(cal_omc, cal_omc_old, d_cal_omc)
            # input()
    
        tt_list.append(np.array(tt))
    
    #Restore photometry
    phot.flux_f = np.copy(flux_f_copy)
    
    tobs_list = []
    omc_list = []
    omc_err_list = []
    ntt_list = []
    for tt1 in tt_list:
        tobs_list.append(tt1[:,0])
        omc_list.append(tt1[:,1])
        omc_err_list.append(tt1[:,2])
        ntt_list.append(len(tt1[:,0]))
    
    ntt_new     = np.array(ntt_list)
    tobs_new    = pad_list_of_arrays(tobs_list)
    omc_new     = pad_list_of_arrays(omc_list)
    omc_err_new = pad_list_of_arrays(omc_err_list)

    return ntt_new, tobs_new, omc_new, omc_err_new


def phot_lombscargle(phot, tmodel):
    """Analyze Residual Photometry using Lomb-Scargle periodogram.
    """
    ls = LombScargle(phot.time, phot.flux-tmodel, phot.ferr)
    frequency, power = ls.autopower(maximum_frequency=48.0)

    # Plotting the periodogram
    plt.figure(figsize=(10, 6))
    plt.plot(86400/frequency, power)
    plt.title('Lomb-Scargle Periodogram')
    plt.xlabel('Period (s)')
    plt.ylabel('Power')
    plt.xscale('log')
    # plt.xlim(0.1, 1)
    plt.grid(True)
    plt.show()

    # Find the frequency with the highest power
    best_frequency = frequency[np.argmax(power)]
    best_period = 1 / best_frequency
    print(f"The most significant frequency is: {best_frequency:.4f} c/d")
    print(f"Corresponding period: {best_period:.4f} day")

def prewhiten(phot, tmodel, n_iterations=20, max_frequency=48):
    """Iterative pre-whitening using Lomb-Scargle periodogram.

    Parameters
    ----------
    phot : object
        Object with attributes `time`, `flux`, and `ferr`.
    tmodel : array-like
        Current transit model evaluated at `phot.time` (same length).
    n_iterations : int, optional
        Number of frequencies to iteratively identify and remove.
    max_frequency : float, optional
        Maximum frequency for Lomb-Scargle `autopower` (cycles per day).

    Returns
    -------
    full_model : np.ndarray
        Sum of all single-frequency models found across iterations.
    found_frequencies : list[dict]
        Each dict contains `frequency`, `period`, and `amplitude`.
    """

    # Working residuals (photometry minus transit model)
    residual_flux = np.asarray(phot.flux) - np.asarray(tmodel)

    found_frequencies = []
    full_model = np.zeros_like(residual_flux)

    for i in range(n_iterations):
        # Lomb-Scargle on current residuals
        ls = LombScargle(phot.time, residual_flux, phot.ferr)
        frequency, power = ls.autopower(maximum_frequency=max_frequency)

        # Pick peak frequency
        best_idx = np.argmax(power)
        best_frequency = frequency[best_idx]
        best_period = 1.0 / best_frequency

        # Best-fit sinusoid at that frequency
        single_freq_model = ls.model(phot.time, best_frequency)

        # Simple amplitude estimate from the model range
        amplitude = (single_freq_model.max() - single_freq_model.min()) / 2.0
        found_frequencies.append({
            "frequency": float(best_frequency),
            "period": float(best_period),
            "amplitude": float(amplitude),
        })

        # Accumulate model and update residuals
        full_model += single_freq_model
        residual_flux -= single_freq_model

    return full_model, found_frequencies

def plot_full_model_overlay(phot, tmodel, full_model):
    """Overlay the cumulative multi-frequency model on the photometry.

    Plots a scatter of `phot.time` vs `phot.flux` and overlays a line for
    `tmodel + full_model` sampled at `phot.time`.

    Parameters
    ----------
    phot : object
        Object with attributes `time`, `flux`, and `ferr`.
    tmodel : array-like
        Transit model evaluated at `phot.time`.
    full_model : array-like
        Cumulative multi-frequency model returned by `prewhiten`.
    """
    time = np.asarray(phot.time)
    flux = np.asarray(phot.flux)
    model = np.asarray(tmodel) + np.asarray(full_model)

    # Sort by time for a clean line overlay
    order = np.argsort(time)
    ts = time[order]
    ms = model[order]

    plt.figure(figsize=(12, 5))
    plt.scatter(time, flux, s=5, alpha=0.6, label="Photometry")
    plt.plot(ts, ms, color="k", lw=1.5, label="Transit + Full Model")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Photometry with Full Model Overlay")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_ttv_comparison(ntt, tobs, omc, omc_err, ntt_new, tobs_new, omc_new, omc_err_new, koi_df, KOI, savefig=False, output_dir='TTVs'):
    """Plot comparison of old vs new TTV measurements.

    Parameters
    ----------
    ntt : array-like
        Number of transit times for each planet (old).
    tobs : array-like
        Observed transit times (old).
    omc : array-like
        O-C values (old).
    omc_err : array-like
        O-C errors (old).
    ntt_new : array-like
        Number of transit times for each planet (new).
    tobs_new : array-like
        Observed transit times (new).
    omc_new : array-like
        O-C values (new).
    omc_err_new : array-like
        O-C errors (new).
    koi_df : DataFrame
        KOI data frame with planet information.
    KOI : int
        KOI number for the system.
    savefig : bool, optional
        If True, save the figure as a PDF file (default: False).
    output_dir : str, optional
        Directory to save the PDF file (default: 'TTVs').
    """
    import os
    
    num_planets = len(ntt)

    # Create a figure and a set of subplots.
    # sharex=True is the key to linking the x-axes.
    # We make the figure taller based on the number of planets.
    fig, axes = plt.subplots(
        nrows=num_planets,
        ncols=1,
        figsize=(12, 2.5 * num_planets),
        sharex=True
    )
    plt.rcParams.update({'font.size': 12})

    # If there's only one planet, axes is not a list, so we make it one
    if num_planets == 1:
        axes = [axes]

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_planets))

    # Loop through each planet and plot on its dedicated axis
    for i, ntt_1 in enumerate(ntt):
        # Select the correct axis for this planet
        ax = axes[i]
        
        # Get the data slice for the current planet (old)
        time_data = tobs[i, 0:ntt_1]
        omc_data = omc[i, 0:ntt_1] * 24 * 60  # Convert to minutes
        omc_error_data = omc_err[i, 0:ntt_1] * 24 * 60 # Convert to minutes

        ax.errorbar(
            time_data[omc_error_data > 0],
            omc_data[omc_error_data > 0],
            yerr=omc_error_data[omc_error_data > 0],
            fmt='o',
            linestyle='none',
            capsize=4.0,
            label=str(koi_df["KOI"].values[i]),
            color="blue"
        )

        # Get the data slice for the current planet (new)
        time_data = tobs_new[i, 0:ntt_new[i]]
        omc_data = omc_new[i, 0:ntt_new[i]] * 24 * 60  # Convert to minutes
        omc_error_data = omc_err_new[i, 0:ntt_new[i]] * 24 * 60 # Convert to minutes

        ax.errorbar(
            time_data[omc_error_data > 0],
            omc_data[omc_error_data > 0],
            yerr=omc_error_data[omc_error_data > 0],
            fmt='o',
            linestyle='none',
            capsize=4.0,
            label="New",
            color="red"
        )
        
        # Set the y-label for each subplot
        ax.set_ylabel('O-C (mins)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    # The x-axis label only needs to be set for the bottom-most plot
    axes[-1].set_xlabel('Time (BJD-2454900)')

    # Add a title for the entire figure
    fig.suptitle(f'Transit Timing Variations for KOI-{str(KOI)} System', fontsize=16, y=0.95)

    fig.subplots_adjust(hspace=0)

    # Save figure if requested
    if savefig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.join(output_dir, f"koi{KOI}_comp.pdf")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {filename}")

    plt.show()

def save_timing_data(ntt, tobs, omc, omc_err, koi_df, output_dir='TTVs'):
    """Save transit timing data to .tt files for each planet.
    
    This function saves timing data in the format expected by get_timing_data():
    - Column 1: Calculated transit time (tobs from fit_ttvs)
    - Column 2: Observed transit time (tobs + omc)
    - Column 3: Error on observed transit time (omc_err)
    
    Files are saved as koi{KOI:07.2f}.tt in the output directory.
    
    Parameters
    ----------
    ntt : array-like
        Number of transit times for each planet.
    tobs : array-like
        Calculated transit times (2D array: planets x transits).
    omc : array-like
        O-C values (2D array: planets x transits).
    omc_err : array-like
        O-C errors (2D array: planets x transits).
    koi_df : DataFrame
        KOI data frame with planet information including 'KOI' column.
    output_dir : str, optional
        Directory to save the .tt files (default: 'TTVs').
        
    Returns
    -------
    saved_files : list
        List of filenames that were saved.
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    num_planets = len(ntt)
    
    for i in range(num_planets):
        # Get KOI number for this planet
        koi_value = koi_df["KOI"].values[i]
        koi_full_str = f"{koi_value:07.2f}"
        
        # Create filename
        filename = os.path.join(output_dir, f"koi{koi_full_str}.tt")
        
        # Get the valid data for this planet (ntt[i] points)
        n_transits = ntt[i]
        
        if n_transits > 0:
            # Extract data for this planet
            calc_times = tobs[i, :n_transits]
            omc_values = omc[i, :n_transits]
            omc_errors = omc_err[i, :n_transits]
            
            # Calculate observed times: observed = calculated + omc
            obs_times = calc_times + omc_values
            
            # Write to file
            with open(filename, 'w') as f:
                for j in range(n_transits):
                    # Format: calc_time obs_time error
                    f.write(f"{calc_times[j]:.10f} {obs_times[j]:.10f} {omc_errors[j]:.10f}\n")
            
            print(f"Saved {n_transits} timing points to {filename}")
            saved_files.append(filename)
        else:
            print(f"No timing points for KOI {koi_full_str}, skipping file creation.")
    
    return saved_files

def fourier_decompose(time, flux, max_frequency=None, n_terms=10):
    """Perform Fourier decomposition of TTV data using Lomb-Scargle periodogram.

    Parameters:
    time : array-like
        Time data points.
    ttv : array-like
        Transit Timing Variations data points.
    max_frequency : float, optional
        Maximum frequency to consider in the decomposition.
    n_terms : int, optional
        Number of Fourier terms to include in the model.

    Returns:
    model_func : function
        A function that takes time as input and returns the Fourier model values.
    coefficients : dict
        Dictionary containing the Fourier coefficients.
    """
    # Compute Lomb-Scargle periodogram
    frequency, power = LombScargle(time, flux).autopower(maximum_frequency=max_frequency)
    
    # Select top n_terms frequencies
    top_indices = power.argsort()[-n_terms:][::-1]
    top_frequencies = frequency[top_indices]
    
    # Fit sine and cosine terms for each frequency
    coefficients = {}
    for i, freq in enumerate(top_frequencies):
        omega = 2 * np.pi * freq
        A = np.sum(flux * np.sin(omega * time)) / np.sum(np.sin(omega * time)**2)
        B = np.sum(flux * np.cos(omega * time)) / np.sum(np.cos(omega * time)**2)
        coefficients[f'freq_{i}'] = (A, B, freq)
    
    def model_func(t):
        model = np.zeros_like(t)
        for A, B, freq in coefficients.values():
            omega = 2 * np.pi * freq
            model += A * np.sin(omega * t) + B * np.cos(omega * t)
        return model
    
    return model_func, coefficients