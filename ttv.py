# %% # Load the autoreload extension for development 
# %load_ext autoreload
# %autoreload 2
# %%
# import os, sys

import numpy as np
import copy

import pytfit5.bls_cpu as gbls     # BLS routines that run on the CPU
import pytfit5.transitPy5 as tpy5  # Transit processing modules 

import matplotlib.pyplot as plt  #MatPlotLib for some simple plots 

# for BLS routine
import pytfit5.transitPy5 as tpy5        # routines to handle transit photometry
import pytfit5.bls_cpu as gbls           # BLS CPU implementation 

# Transit modelling module
import pytfit5.transitmodel as transitm
import pytfit5.keplerian as kep
import pytfit5.transitfit as transitf
import pytfit5.transitplot as transitp

# Reading in CSV file 
import pandas as pd


import Kepler_TTV as kttv
# %%
# The URL of the CSV file (please do not abuse this link)
url = "https://kona.ubishops.ca/architecture/KeplerARCH_20231117.csv"

# Read the CSV file into a pandas DataFrame
Kepler_cat = pd.read_csv(url)
# %%
KOI = 2261  #The system we want 

# Get the data frame for the requested KOI
koi_df = tpy5.find_koi_rows(KOI, Kepler_cat) 

# Fetch data products and set up necessary classes.  Set raw = 1 to get raw PDC light-curves
phot, tpy5_inputs = tpy5.get_photometry(koi_df, raw = 0) 

# Get the best-fit transit model (this is different than the Arch Catalogue) 
sol = tpy5.populate_transit_model(Kepler_cat, KOI)

# Get transit-timing measurements.
ntt, tobs, omc, omc_err = tpy5.get_timing_data(Kepler_cat, KOI)

# Update flags to mark in-transit data (note: this routine does not yet account for TTVs)
kep.mark_intransit_data(phot, sol, tdurcut = 2.0)  # tdurcut is the amount of +/- time to protect centred on the transit 
# %%
# Quick plot of the data and transit model

# Processed Light curve 
fig=plt.figure(figsize=(12, 4))
plt.rcParams.update({'font.size': 12})
plt.scatter(phot.time,phot.flux, s=5)
plt.xlabel('Time (BJD-2454900)')
plt.ylabel('Relative Flux')
plt.show()
# %%
for pl_plot in range(sol.npl):
    transitp.plotTransit(phot, sol, pl_to_plot=pl_plot+1)
# %%
tpy5.plotTTVs(ntt, tobs, omc, omc_err, KOI, koi_df)
# %%
# # Run the detrending and outlier detection (if needed) 
# tpy5_inputs.boxbin = 2.0 # set detrending length
# tpy5_inputs.nfitp  = 3   # Set polynomial order
# tpy5.run_polyfilter(phot, tpy5_inputs)  # flux_f will contain the filtered photometry
# tpy5.run_cutoutliers(phot, tpy5_inputs) # data clipping, phot.icut will flag bad data

# # Processed Light curve 
# fig=plt.figure(figsize=(12, 4))
# plt.rcParams.update({'font.size': 12})
# plt.scatter(phot.time[phot.icut==0],phot.flux[phot.icut==0], s=5)
# plt.xlabel('Time (BJD-2454900)')
# plt.ylabel('Relative Flux')
# plt.show()
# %%
# Define the parameters to fit
params_to_fit = ["rho", "zpt", "t0", "per", "bb", "rdr"]

# Fit the data - phot.flux_f is used for the fit.
# We are including the TTVs with this model. 
sol_fit = transitf.fitTransitModel(sol, params_to_fit, phot, ntt=ntt, tobs=tobs, omc=omc)

# See the fitted parameters (Note the error is a simple co-variance matrix at this stage)
transitp.printParams(sol_fit)
# %%
# Note, we are using the re-fitted model.  Change back to 'sol' to use the original.
sol_fft = copy.deepcopy(sol_fit)
# sol_fft.rdr[0] = 0.0 # disable first known planet for residual/BLS search only
tmodel = transitm.transitModel(sol_fft, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)
# %%
kttv.phot_lombscargle(phot, tmodel)
# %%
full_model, found = kttv.prewhiten(phot, tmodel, n_iterations=20, max_frequency=48)
kttv.plot_full_model_overlay(phot, tmodel, full_model)
tpy5.run_cutoutliers(phot, tpy5_inputs) # data clipping, phot.icut will flag bad data# %%

# %%
# Set up the BLS inputs and run
gbls_inputs          = gbls.gbls_inputs_class() # Initialize needed inputs
gbls_inputs.zerotime = 0.0
gbls_inputs.Rstar    = tpy5_inputs.mstar
gbls_inputs.Mstar    = tpy5_inputs.rstar
gbls_inputs.plots    = 1 # 1=X11, 2=both, 0=none
gbls_inputs.freq1    = 2/(np.max(phot.time) - np.min(phot.time))
gbls_inputs.freq2    = -1 # default value of 0.5 day period
gbls_inputs.filename = tpy5_inputs.photfile  # Used for naming the file, extension replaced with PNG.
gbls_inputs.normalize = "iterative_baseline"

## Work here to recover any individual planets.
sol_bls = copy.deepcopy(sol_fit)
## sol_bls.rdr[1] = 0.0 # disable second known planet for residual/BLS search only
tmodel = transitm.transitModel(sol_bls, phot.time, itime=phot.itime, ntt=ntt, tobs=tobs, omc=omc)

# Run BLS, applying data cuts and removing the current best-fit model (so we are searching the residuals)
gbls_ans = gbls.bls(gbls_inputs, phot.time[(phot.icut == 0)], phot.flux_f[(phot.icut == 0)] - tmodel[(phot.icut == 0)] - full_model[(phot.icut == 0)])
# %%
## Measure TTVs

pflag  = 0 # 1 = use the previous O-C measurements to initialize the next fit 
pstart = 0 # set the guess for the first O-C measurement 
# pflag  = [1, 1, 0] 
# pstart = [omc[0,0], omc[1,0], 0] # Example that uses the exisiting O-C measurements to initialize the model
phot.flux_f = phot.flux - full_model 
ntt_new, tobs_new, omc_new, omc_err_new, npts, rchi2, raw_err, scatter, mean_res = kttv.fit_ttvs(phot, sol_fit, ntt, tobs, omc, \
    pflag = pflag, pstart = pstart, return_diagnostics=True)

kttv.plot_ttv_comparison(ntt, tobs, omc, omc_err, ntt_new, tobs_new, omc_new, omc_err_new, koi_df, KOI)

# %%
print("Median Error Old:", np.median(omc_err[0])*24*60)
print("Median Error New:", np.median(omc_err_new[0])*24*60)
print("Percentage change: ", (np.median(omc_err_new[0]) - np.median(omc_err[0])) / np.median(omc_err[0]))
# %%
kttv.plot_ttv_diagnostics(
    ntt_new, tobs_new, omc_new, omc_err_new, 
    npts, rchi2, raw_err, scatter, mean_res,
    koi_df, KOI
)
# %%
# After running fit_ttvs to get new TTV measurements:
overlaps, overlap_pairs = kttv.find_overlapping_transits(ntt_new, tobs_new, omc_new, sol, duration_factor=2.2)
print(f"Found {len(overlap_pairs)} overlapping transit pairs")
# %%
# 2. Plot Comparison with Highlights
kttv.plot_ttv_comparison(
    ntt, tobs, omc, omc_err,          # Old Data
    ntt_new, tobs_new, omc_new, omc_err_new,          # New Data
    koi_df, KOI,
    reduced_chi2=rchi2,               # New Arg: Pass Chi2 array
    mean_res=mean_res,                # New Arg: Pass Mean Res array
    chi2_threshold=20.0,               # Threshold for Chi2
    mean_res_threshold=3.0e-4,        # Threshold for |Mean Res|
    overlap_indices=overlaps,          # (Optional) Overlaps
    planets_to_plot=[0, 1]
)
# %%
kttv.plot_ttv_comparison(ntt, tobs, omc, omc_err, ntt_new, tobs_new, omc_new, omc_err_new, koi_df, KOI, savefig=True)
# %%
# Save the results
saved_files = kttv.save_timing_data(ntt_new, tobs_new, omc_new, omc_err_new, koi_df)
# %%
print("hello")
# %%
