# rank-k-dstts

## hodstts
This is the code from the "Improved Directional State Transition Tensors for Aerocapture Performance Analysis Paper.'' Note that this code uses EOMs which use a natural log of density state rather than a KLE for atmospheric density. I recommend we use the KLE dynamics in `updatedDevCode`, but this is the code base which has the hoDSTTs. 
- `util/` includes all helper functions for aerocapture EOMs, computing the STTs / DSTTs, and propagating perturbations and moments with the DSTTs / STTs
- `inputs/` include input functions for different scenarios
- `2_get_STTs.py`: Computes STTs given EOMs for different time regimes by integrating the variational equations.
- `3a_get_all_hocgts_nonsym.py`: Compute all HOCGTs, sCGTs, qCGTs for the aerocapture trajectory (nonsymmetrical) to pass into MATLAB to symmetrize and solve the tensor eigenpair problem
- `Threeb_compute_all_hocgt_eigenpairs.m` and `Threeb_compute_all_hocgt_eigenpairs_par.m`: Symmetrize all HOCGTs, sCGTs, qCGTs for the aerocapture trajectory and solve higher-order tensor eigenpair problem with SSHOPM solver and save off maximal eigenpairs
- `3c_eval_all_hocgt_eignpairs.m`: Load in eigenpairs from MATLAB and plot and compare them
- `7a_get_DSTTs.py`: Gets the DSTTs from the STTs by rotating them into the selected basis
- `7b_eval_DSTT_Frobenius_norm.py`: Evaluates the difference between different STTs and DSTTs using the Frobenius norm

## updatedDevCode
This code contains the KLE dynamics for atmospheric density and Python code to run SS-HOPM to solve the TE problem
- `util/` includes helpers again, but in this folder we have `tensor_toolbox.py` and `kle.py`
- `inputs/` include input functions for different scenarios
- `TEST_get_KLE.py`: Generates a KLE model of the UranusGRAM atmosphere. You should not need to run this as I provided the KLE data.
- `TEST_DSTTs_with_KLE.py`: Computes the DSTTs/hoDSTTs using KLE atmosphere model either including or not including the KLE parameters in the state.
- `TEST_sshopm_python.py`: Compares MATLAB and Python SS-HOPM from the tensor toolbox
- `TEST_sshopm_python_get_TEs.m`: MATLAB complement to the above code

Resources to read: 
- My ESA GNC paper, which has a brief explanation of the aerocapture KLE dynamics [ESA GNC Paper](https://www.researchgate.net/publication/413793571_Dynamics-Informed_Semi-Analytical_Uncertainty_Propagation_for_Aerocapture)
- Sam Albert's KLE Atmosphere Paper [KLE Atmosphere Modelling](https://doi.org/10.2514/1.A35839)
- Sam Albert and Jack Ridderhof's Linear Covariance Analysis with KLE atmosphere modeling conference paper (using the KLE with STTs/DSTTs is an extension of this) [KLE LinCov](https://arc.aiaa.org/doi/10.2514/6.2022-1216)
- My submitted journal paper on hoDSTTs [hoDSTTs](https://www.researchgate.net/publication/398719969_Improved_Directional_State_Transition_Tensors_for_Accurate_Aerocapture_Performance_Analysis)
