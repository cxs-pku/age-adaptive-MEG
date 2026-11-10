ECD and finite robustness validation (v2)
========================================

Run from this `codes` directory with Python:

    python ecd_robustness_v2.py

The script imports geometry and field constants from `model.py`. It writes only to `../outputs/ecd_robustness_v2/` and records the master seed, model parameters, trial counts, and elapsed time in `parameters.json`.

Model and fitting
-----------------
Truth locations are sampled continuously and uniformly by area in the Lambert equal-area disk `rho <= 1.52`, then mapped to the age-specific spherical source shell. Each trial draws a uniform tangential orientation and a 10 nA m dipole. Each measurement adds fresh Gaussian noise with a 250 fT per-channel marginal RMS. The inverse uses a separate 921-location cap grid (center, 20 radial rings with increasing azimuth counts, and a boundary ring), rather than the 169-point dictionary. At every candidate location, the two local tangential lead-field components are fit by generalized least squares; the location with the smallest profiled residual is selected. The reported primary error is the Euclidean chord distance on the source shell; geodesic error is also saved.

The same source direction, orientation, and standard-normal noise streams are paired across matching ages/trials for layouts, channel counts, and robustness conditions. Streams are reproducible from `SeedSequence([20261024, age_months, trial, stream_id])` with stream IDs 0–3 for truth, noise, position, and orientation. There are 100 trials per configuration.

Robustness cases
----------------
- Correlated noise uses `C_ij = 250^2 * [(1-rho) I_ij + rho exp(-d_ij^2/(2 ell^2))]`, with `rho=0.35`, `ell=35 mm`, and nominal sensor-center distances. Noise is sampled from this covariance and the ECD fit uses the known inverse-Cholesky whitening transform.
- Sensor registration cases perturb true sensor centers with independent isotropic Gaussian offsets having 3 mm RMS vector magnitude and perturb true sensor normals with independent tangent-plane rotations having 3 degree RMS angular magnitude. Position-only, orientation-only, and combined cases are evaluated. The inverse always uses the nominal sensor geometry.
- The nested golden-angle family uses the same sensor cap (`rho<=1.28`) and 10 mm active aperture. It consists of a center sensor followed by a deterministic van der Corput radial / golden-angle azimuth sequence; the 25-, 41-, and 61-channel arrays are prefixes. The table reports the matched-field metrics on the 169-point source dictionary for comparison with the non-nested D4 ring family.

Outputs
-------
- `ecd_trials.csv`, `ecd_summary.csv`: independent off-grid ECD Monte Carlo runs for ages 12/48/120/216 months, K=25/41/61, and both layouts.
- `robustness_trials.csv`, `robustness_summary.csv`: correlated-noise and position/orientation-mismatch ECD trials for ages 12/120/216 months, K=41/61, and both layouts.
- `nested_golden_angle_matched_field.csv`: matched-field metrics for the nested array family at ages 12/120/216 months, K=25/41/61, and both layouts.
- `parameters.json`: full run configuration and elapsed time.

Fixed-orientation sensitivity
-----------------------------
The supplementary `orientation_sensitivity.py` script runs a paired 41-channel ECD check at ages 12, 120, and 216 months for four fixed local tangential orientations (0, 45, 90, and 135 degrees), both layouts, and 50 off-grid source/noise pairs per condition. Within each pair, source location and sensor-noise vector are held fixed across orientations. The script writes `orientation_sensitivity_trials.csv` and `orientation_sensitivity_summary.csv` to this output directory. It uses the same independent 921-point fitting grid and forward model as the primary ECD test. This checks sensitivity to several in-plane dipole directions; it does not vary source-space extent, depth distribution, or spatial extent.

Scope
-----
These are controlled physical-model simulations using the quasi-static Biot-Savart kernel of the main model. Off-grid truth and a denser independent location grid avoid reusing the 169-point source dictionary, but do not substitute for MRI-derived anatomy, a Sarvas/BEM/FEM model, or empirical OPM-MEG data. The localization errors are conditional on this model and noise definition and should not be presented as clinical accuracy estimates.

The initial verification run used Python 3.9.0 and NumPy 2.0.2. The script has no dependency beyond NumPy and the adjacent `model.py` module.

To regenerate vector Figures 9 and 10 after running the simulations, execute:

    python plot_validation_v2.py

This plotting script additionally requires pandas and Matplotlib. It reads only the CSV outputs above and writes the two PDF figures into `manuscript-tex-v2/`.
