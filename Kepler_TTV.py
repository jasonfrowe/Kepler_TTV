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

def pad_list_of_arrays(list_of_arrays):
    max_len = max([len(arr) for arr in list_of_arrays])
    padded_array = np.zeros((len(list_of_arrays), max_len))
    for i, arr in enumerate(list_of_arrays):
        padded_array[i, :len(arr)] = arr
    return padded_array

def fit_ttvs(phot, sol_fit, ntt=-1, tobs=-1, omc=-1, pflag = 0, pstart = 0, return_diagnostics=False):

    if isinstance(pflag, int):
        pflag = np.ones((sol_fit.npl),dtype=int)*pflag

    if isinstance(pstart, int):
        pstart = np.ones((sol_fit.npl),dtype=int)*pstart
    
    flux_f_copy = np.copy(phot.flux_f)
    
    Tmin = np.min(phot.time[(phot.icut == 0)])
    Tmax = np.max(phot.time[(phot.icut == 0)])
    
    tt_list      = []
    
    # Lists for diagnostics
    n_points_list = []
    rchi2_list = []
    raw_err_list = []
    scatter_list = []
    mean_res_list = []
    
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
        diag_n  = []
        diag_rchi2 = []
        diag_raw_err = []
        diag_scatter = []
        diag_mean_res = []
        
        # print(pflag[nplanet])
        while(T0 < Tmax):
            
            # # Only for KOI-1599
            # if T0 > 93.0 and T0 < 115.0 and nplanet == 0:
            #     cal_omc = -0.22 
            #     print("Applied manual correction to T0 for transit around 100.00 days")
            # else:
            #     cal_omc = 0.0
            

            Ts = T0 - 2.0*tdur + cal_omc         + d_cal_omc
            Te = T0 + 2.0*tdur + cal_omc         + d_cal_omc
            Ts2= T0 - 0.5*tdur + cal_omc - 0.021 + d_cal_omc  # add 30-mins
            Te2= T0 + 0.5*tdur + cal_omc + 0.021 + d_cal_omc  
            sol_c.t0[0] = T0   + cal_omc         + d_cal_omc
    
            params_to_fit = ["t0", "zpt"]
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
                
                # Store diagnostics
                diag_n.append(k)
                diag_rchi2.append(reduced_chi2)
                diag_raw_err.append(sol_c_fit.dt0[0])
                diag_scatter.append(np.std(residuals))
                diag_mean_res.append(np.mean(residuals))
    
            else:
                cal_omc += d_cal_omc
    
            T0 = T0 + sol_fit.per[nplanet]
    
            if pflag[nplanet] == 0:
                cal_omc   = 0.0  # check if we are using predictive
                d_cal_omc = 0.0

            # print(cal_omc, cal_omc_old, d_cal_omc)
            # input()
    
        tt_list.append(np.array(tt))
        n_points_list.append(np.array(diag_n))
        rchi2_list.append(np.array(diag_rchi2))
        raw_err_list.append(np.array(diag_raw_err))
        scatter_list.append(np.array(diag_scatter))
        mean_res_list.append(np.array(diag_mean_res))
    
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

    if return_diagnostics:
        n_points_new = pad_list_of_arrays(n_points_list)
        rchi2_new = pad_list_of_arrays(rchi2_list)
        raw_err_new = pad_list_of_arrays(raw_err_list)
        scatter_new = pad_list_of_arrays(scatter_list)
        mean_res_new = pad_list_of_arrays(mean_res_list)
        return ntt_new, tobs_new, omc_new, omc_err_new, n_points_new, rchi2_new, raw_err_new, scatter_new, mean_res_new

    return ntt_new, tobs_new, omc_new, omc_err_new


def plot_ttv_diagnostics(ntt, tobs, omc, omc_err, n_points, reduced_chi2, raw_err, scatter, mean_res, koi_df, KOI, savefig=False, output_dir='TTVs'):
    """Plot diagnostic information for TTV measurements.
    
    Generates a multi-panel plot for each planet showing:
    1. O-C vs Time
    2. Reduced Chi^2 vs Time
    3. Residual Scatter vs Mean Residual (Correlation Check!)
    4. O-C Error vs Reduced Chi^2
    5. Reduced Chi^2 vs |Mean Residual| (Systematic Bias Check)
    6. Scatter vs N points

    Parameters
    ----------
    ntt : array-like
        Number of transit times for each planet.
    tobs : array-like
        Observed transit times.
    omc : array-like
        O-C values.
    omc_err : array-like
        O-C errors.
    n_points : array-like
        Number of in-transit data points per transit.
    reduced_chi2 : array-like
        Reduced chi-square values per transit.
    raw_err : array-like
        Raw (unscaled) O-C errors.
    scatter : array-like
        Standard deviation of flux residuals per transit.
    mean_res : array-like
        Mean of flux residuals per transit.
    koi_df : DataFrame
        KOI data frame.
    KOI : int
        KOI number.
    savefig : bool, optional
        If True, save the figure as a PDF file (default: False).
    output_dir : str, optional
        Directory to save the PDF file (default: 'TTVs').
    """
    import os
    from matplotlib.colors import LogNorm
    
    num_planets = len(ntt)
    
    for i in range(num_planets):
        nt = ntt[i]
        t_data = tobs[i, :nt]
        omc_data = omc[i, :nt] * 24 * 60  # minutes
        err_data = omc_err[i, :nt] * 24 * 60 # minutes
        n_data = n_points[i, :nt]
        chi2_data = reduced_chi2[i, :nt]
        scatter_data = scatter[i, :nt]
        mean_res_data = mean_res[i, :nt]
        
        # Create a 3x2 grid of subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2)
        
        # 1. O-C vs Time (Top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(t_data, omc_data, yerr=err_data, fmt='o', alpha=0.7)
        ax1.set_ylabel('O-C (min)')
        ax1.set_title(f'KOI-{koi_df["KOI"].values[i]}: O-C Diagram')
        ax1.grid(True, alpha=0.3)
        
        # 2. Reduced Chi^2 vs Time (Top-right)
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
        ax2.plot(t_data, chi2_data, 'ro', alpha=0.6)
        ax2.axhline(1.0, color='k', linestyle='--')
        ax2.set_ylabel(r'Reduced $\chi^2$')
        ax2.set_title('Fit Quality vs Time')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter vs Mean Residual (Middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        sc3 = ax3.scatter(mean_res_data, scatter_data, c=chi2_data, cmap='viridis', alpha=0.7, norm=LogNorm())
        ax3.set_xlabel('Mean Residual (Bias)')
        ax3.set_ylabel('Residual Scatter (Noise)')
        ax3.set_title('Scatter vs Mean Residual (Color=Chi2)')
        plt.colorbar(sc3, ax=ax3, label=r'Reduced $\chi^2$')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error vs Chi^2 (Middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        sc4 = ax4.scatter(chi2_data, err_data, c=n_data, cmap='viridis', alpha=0.7)
        ax4.set_xlabel(r'Reduced $\chi^2$')
        ax4.set_ylabel('O-C Error (min)')
        ax4.set_title('Error vs Fit Quality (Color=N points)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        plt.colorbar(sc4, ax=ax4, label='N points')
        ax4.grid(True, alpha=0.3)
        
        # 5. Chi^2 vs |Mean Residual| (Bottom-left) - SYSTEMATIC CHECK
        ax5 = fig.add_subplot(gs[2, 0])
        sc5 = ax5.scatter(np.abs(mean_res_data), chi2_data, c=n_data, cmap='inferno', alpha=0.7)
        ax5.set_xlabel('|Mean Residual| (Depth Mismatch?)')
        ax5.set_ylabel(r'Reduced $\chi^2$')
        ax5.set_title(r'$\chi^2_\nu$ vs Systematic Offset')
        ax5.set_yscale('log')
        # ax5.set_xscale('log')
        plt.colorbar(sc5, ax=ax5, label='N points')
        ax5.grid(True, alpha=0.3)
        
        # 6. Scatter vs Chi2 (Bottom-right) - REVISITED
        ax6 = fig.add_subplot(gs[2, 1])
        sc6 = ax6.scatter(scatter_data, chi2_data, c=np.abs(mean_res_data), cmap='coolwarm', alpha=0.7)
        ax6.set_xlabel('Residual Scatter')
        ax6.set_ylabel(r'Reduced $\chi^2$')
        ax6.set_title(r'$\chi^2$ vs Scatter (Color=|Mean Res|)')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        plt.colorbar(sc6, ax=ax6, label='|Mean Residual|')
        ax6.grid(True, alpha=0.3)

        fig.tight_layout()
        
        if savefig:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = os.path.join(output_dir, f"koi{koi_df['KOI'].values[i]}_diagnostics.pdf")
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved diagnostic figure to {filename}")
        
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

def find_overlapping_transits(ntt, tobs, omc, sol, duration_factor=1.0):
    """Find transits that overlap between different planets.
    
    For each transit of each planet, finds the closest transit from every other
    planet and checks if they overlap based on transit durations.
    
    Uses actual observed transit times (tobs + omc) for overlap detection.
    
    Parameters
    ----------
    ntt : array-like
        Number of transit times for each planet.
    tobs : array-like
        2D array of calculated transit times [planet, transit].
    omc : array-like
        2D array of O-C (observed minus calculated) values [planet, transit].
        The actual observed time is tobs + omc.
    sol : transit_model_class
        Solution object containing orbital parameters.
    duration_factor : float
        Factor to multiply transit duration by when checking overlaps.
        Default is 1.0 (full duration). Use 0.5 for half-duration overlap.
    
    Returns
    -------
    overlaps : dict
        Dictionary with planet index as key and list of overlapping 
        transit indices as value.
    overlap_pairs : list
        List of tuples ((planet_i, transit_i), (planet_j, transit_j))
        representing pairs of overlapping transits.
    """
    num_planets = len(ntt)
    
    # Calculate transit durations for each planet
    transit_durations = []
    for nplanet in range(num_planets):
        tdur = kep.transitDuration(sol, nplanet)
        transit_durations.append(tdur)
    
    # Initialize output structures
    overlaps = {i: [] for i in range(num_planets)}
    overlap_pairs = []
    
    # Compare each planet's transits with all other planets
    for i in range(num_planets):
        tdur_i = transit_durations[i] * duration_factor
        # Use actual observed times: calculated time + O-C correction
        times_i = tobs[i, :ntt[i]] + omc[i, :ntt[i]]
        
        for j in range(num_planets):
            if i == j:
                continue
                
            tdur_j = transit_durations[j] * duration_factor
            # Use actual observed times: calculated time + O-C correction
            times_j = tobs[j, :ntt[j]] + omc[j, :ntt[j]]
            
            # Combined overlap threshold: half of each transit duration
            overlap_threshold = (tdur_i + tdur_j) / 2.0
            
            # For efficiency, sort times_j and use searchsorted to find closest
            sorted_indices_j = np.argsort(times_j)
            sorted_times_j = times_j[sorted_indices_j]
            
            # Check each transit of planet i for overlap with closest transit of planet j
            for ti_idx, t_i in enumerate(times_i):
                # Find insertion point for t_i in sorted_times_j
                insert_pos = np.searchsorted(sorted_times_j, t_i)
                
                # Check candidates: the transit just before and just after insertion point
                candidates = []
                if insert_pos > 0:
                    candidates.append(insert_pos - 1)
                if insert_pos < len(sorted_times_j):
                    candidates.append(insert_pos)
                
                # Find the closest transit time
                for cand_pos in candidates:
                    t_j = sorted_times_j[cand_pos]
                    tj_idx = sorted_indices_j[cand_pos]  # Original index
                    
                    # Check if transits overlap
                    if abs(t_i - t_j) < overlap_threshold:
                        # Record overlap for planet i
                        if ti_idx not in overlaps[i]:
                            overlaps[i].append(ti_idx)
                        # Record the pair (only if i < j to avoid duplicates)
                        if i < j:
                            pair = ((i, ti_idx), (j, tj_idx))
                            if pair not in overlap_pairs:
                                overlap_pairs.append(pair)
    
    # Sort the overlap indices for each planet
    for planet in overlaps:
        overlaps[planet] = sorted(overlaps[planet])
    
    return overlaps, overlap_pairs



def plot_ttv_comparison(ntt, tobs, omc, omc_err, ntt_new, tobs_new, omc_new, omc_err_new, koi_df, KOI, 
                        savefig=False, output_dir='TTVs', 
                        overlap_indices=None, highlight_overlaps=True,
                        reduced_chi2=None, mean_res=None, 
                        chi2_threshold=None, mean_res_threshold=None,
                        planets_to_plot=None):
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
    overlap_indices : dict, optional
        Dictionary with planet index as key and list of overlapping 
        transit indices as value (from find_overlapping_transits).
    highlight_overlaps : bool, optional
        If True and overlap_indices is provided, highlight overlapping 
        transits with distinct markers (default: True).
    reduced_chi2 : array-like, optional
        Reduced chi-square values for the NEW measurements.
    mean_res : array-like, optional
        Mean residual values for the NEW measurements.
    chi2_threshold : float, optional
        Threshold for reduced chi-square. Points above this will be highlighted.
    mean_res_threshold : float, optional
        Threshold for |mean_residual|. Points above this will be highlighted.
    planets_to_plot : list, optional
        List of planet indices (0-based) to plot. Example: [0, 1] for first two planets.
        If None, plots all planets (default).
    """
    import os
    
    # Determine which planets to plot
    if planets_to_plot is None:
        planets_to_plot = list(range(len(ntt)))
    
    num_plots = len(planets_to_plot)

    # Create a figure and a set of subplots.
    # sharex=True is the key to linking the x-axes.
    # We make the figure taller based on the number of planets.
    fig, axes = plt.subplots(
        nrows=num_plots,
        ncols=1,
        figsize=(12, 5.0 * num_plots),
        sharex=True
    )
    plt.rcParams.update({'axes.labelsize': 24, 'xtick.labelsize': 20, 'ytick.labelsize': 20})

    # If there's only one planet to plot, axes is not a list, so we make it one
    if num_plots == 1:
        axes = [axes]

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(ntt))) # Use full color set for consistency

    # Loop through requested planets and plot on dedicated axes
    for plot_idx, i in enumerate(planets_to_plot):
        if i >= len(ntt):
            print(f"Warning: Planet index {i} out of range (max {len(ntt)-1}). Skipping.")
            continue
            
        # Select the correct axis for this plot
        ax = axes[plot_idx]
        
        # Get the data slice for the current planet (old)
        ntt_1 = ntt[i]
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
            color="blue", # Keep explicit colors for Old/New comparison logic
            alpha=0.5
        )

        # Get the data slice for the current planet (new)
        time_data_new = tobs_new[i, 0:ntt_new[i]]
        omc_data_new = omc_new[i, 0:ntt_new[i]] * 24 * 60  # Convert to minutes
        omc_error_data_new = omc_err_new[i, 0:ntt_new[i]] * 24 * 60 # Convert to minutes

        # Keep finite points and only draw NEW measurements with valid error bars.
        valid_new = np.isfinite(time_data_new) & np.isfinite(omc_data_new)
        valid_err_new = valid_new & np.isfinite(omc_error_data_new) & (omc_error_data_new > 0)

        ax.errorbar(
            time_data_new[valid_err_new],
            omc_data_new[valid_err_new],
            yerr=omc_error_data_new[valid_err_new],
            fmt='o',
            linestyle='none',
            capsize=4.0,
            label="New",
            color="red"
        )
        
        # Highlight overlapping transits if provided
        if overlap_indices is not None and highlight_overlaps and i in overlap_indices:
            overlap_idx = overlap_indices[i]
            if len(overlap_idx) > 0:
                # Create a mask for valid indices within the new data range
                valid_overlap_idx = [idx for idx in overlap_idx if idx < ntt_new[i]]
                if len(valid_overlap_idx) > 0:
                    overlap_times = tobs_new[i, valid_overlap_idx]
                    overlap_omc = omc_new[i, valid_overlap_idx] * 24 * 60
                    
                    # Plot X markers for overlapping transits
                    ax.scatter(
                        overlap_times,
                        overlap_omc,
                        marker='x',
                        s=150,
                        linewidths=2.5,
                        color='darkgreen',
                        zorder=10,
                        label=f"Overlap ({len(valid_overlap_idx)})"
                    )

        # Highlight High Reduced Chi-Squared
        if reduced_chi2 is not None and chi2_threshold is not None:
            chi2_data = reduced_chi2[i, 0:ntt_new[i]]
            bad_chi2_idx = np.where(chi2_data > chi2_threshold)[0]
            
            if len(bad_chi2_idx) > 0:
                 bad_times = time_data_new[bad_chi2_idx]
                 bad_omc = omc_data_new[bad_chi2_idx]
                 
                 ax.scatter(
                    bad_times,
                    bad_omc,
                    s=150,
                    facecolors='none',
                    edgecolors='magenta',
                    linewidths=2,
                    marker='s',
                    zorder=11,
                    label=f"Chi2 > {chi2_threshold}"
                 )

        # Highlight Mean Residual Outliers
        if mean_res is not None and mean_res_threshold is not None:
             mean_res_data = mean_res[i, 0:ntt_new[i]]
             # Use absolute value for threshold comparison
             bad_res_idx = np.where(np.abs(mean_res_data) > mean_res_threshold)[0]
             
             if len(bad_res_idx) > 0:
                 bad_times = time_data_new[bad_res_idx]
                 bad_omc = omc_data_new[bad_res_idx]
                 
                 ax.scatter(
                    bad_times,
                    bad_omc,
                    s=200,
                    facecolors='none',
                    edgecolors='orange',
                    linewidths=2,
                    marker='D',
                    zorder=12,
                    label=f"|MeanRes| > {mean_res_threshold}"
                 )
        
        # Set the y-label for each subplot
        ax.set_ylabel('O-C (mins)')
        ax.legend(loc='lower right', fontsize=14, framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)

    # The x-axis label only needs to be set for the bottom-most plot
    if num_plots > 0:
        axes[-1].set_xlabel('Time (BJD-2454900)')

    # Add a title for the entire figure
    # fig.suptitle(f'Transit Timing Variations for KOI-{str(KOI)} System', fontsize=16, y=0.95)

    fig.subplots_adjust(hspace=0)

    # Save figure if requested
    if savefig:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.join(output_dir, f"koi{KOI}_comp.pdf")
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {filename}")

    plt.show()
