import numpy
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm, trange
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import ray
from ray import remote, get
from scipy.stats import multivariate_normal
import warnings

def computeLAM(MCData, gmm_uq, ts):
    warnings.warn("Warning...there are multiple functions to compute LAM - check which one you want!")
    K = len(MCData)
    # Get list of all non-stale trajectories at time 0
    good_traj_inds = [traj['id'] for traj in gmm_uq.trajectories if 5 in traj['times']]
    L = len(good_traj_inds)

    LAM = np.zeros(len(ts))

    print(f"Computing LAM with {len(good_traj_inds)} mixands")

    # Precompute means, covariances, and weights for all good trajectories to avoid repeated access
    traj_means = np.array([gmm_uq.trajectories[i]['means'] for i in good_traj_inds])
    traj_covs = np.array([gmm_uq.trajectories[i]['covs'] for i in good_traj_inds])
    traj_weights = np.array([gmm_uq.trajectories[i]['weight'] for i in good_traj_inds]).T  # Convert to ndarray

    # Ensure traj_weights has the correct shape (L, 1) to match number of mixands
    traj_weights = traj_weights.reshape(-1, 1)  # Shape: (L, 1)

    # Compute the sum in a vectorized way for all MC samples and good mixands
    for t_ind, t in enumerate(tqdm(ts, desc='Computing LAM')):  # time loop
        # MC samples (K), mixands (L), and time steps (T) can be done in batch.
        MCData_t = MCData[:, :, t_ind]  # Shape: (K, D)

        # Precompute the Gaussian probability densities
        gauss_probs = np.zeros((K, L))  # Shape: (K, L)

        for i, traj_idx in enumerate(good_traj_inds):
            mean = traj_means[i][t_ind]  # Mean for this mixand at this time
            cov = traj_covs[i][t_ind]  # Covariance for this mixand at this time

            # Use multivariate normal PDF on the batch of MCData
            gauss_probs[:, i] = multivariate_normal.pdf(MCData_t, mean=mean, cov=cov, allow_singular=True)

        # Compute the weighted sum for LAM[t_ind]
        weighted_probs = (1 / K) * traj_weights.T * gauss_probs  # Shape: (K, L)
        LAM[t_ind] += np.sum(weighted_probs)  # Sum over all MC samples and mixands to get a scalar

    return LAM


def computeKLD(MCData, gmm_uq, ts):
    N_samples = 10_000  # number of samples to take from each distribution

    # Get list of all non-stale trajectories at time 0
    L = 0
    good_traj_inds = []
    for traj in gmm_uq.trajectories:
        if 5 in traj['times']:
            L += 1
            good_traj_inds.append(traj['id'])

    KLs = []
    for t_ind, t in enumerate(tqdm(ts)):
        # P distribution - empirical Monte Carlo distribution
        gmm_p = GaussianMixture(n_components=len(good_traj_inds), covariance_type='full', random_state=42)
        gmm_p.fit(MCData[:,:,t_ind])

        # Q distribution - GMM at this time
        gmm_q = GaussianMixture(n_components=len(good_traj_inds), covariance_type='full')
        # Get weights, means, and covariances from each mixand of the GMM at this time
        weights_predefined = []
        means_predefined = []
        covariances_predefined = []
        for m in good_traj_inds:
            mixand = gmm_uq.trajectories[m]
            weights_predefined.append(mixand['weight'])
            means_predefined.append(mixand['means'][t_ind])
            covariances_predefined.append(mixand['covs'][t_ind])
        gmm_q.weights_ = np.array(weights_predefined)
        gmm_q.means_ = np.array(means_predefined)
        gmm_q.covariances_ = np.array(covariances_predefined)
        gmm_q.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances_predefined))

        # Sample from GMM P
        samples, _ = gmm_p.sample(N_samples)

        # Compute the log density under both GMMs
        log_p = gmm_p.score_samples(samples)  # Log density under P
        log_q = gmm_q.score_samples(samples)  # Log density under Q

        # Estimate KL divergence
        kl_div = np.mean(log_p - log_q)
        KLs.append(kl_div)

    return KLs


def computeKLD_assumeDMM(MCData, gmm_uq, ts):
    K = len(MCData)  # Number of MC samples

    # Get list of all non-stale trajectories at time 0
    good_traj_inds = [traj['id'] for traj in gmm_uq.trajectories if 5 in traj['times']]
    L = len(good_traj_inds)

    KLD = np.ones(len(ts)) * -np.log(K)  # Initialize KLD for all time steps
    print(f"Computing KLD with {L} mixands")

    # Precompute means, covariances, and weights for all good trajectories
    traj_means = np.array([gmm_uq.trajectories[i]['means'] for i in good_traj_inds])
    traj_covs = np.array([gmm_uq.trajectories[i]['covs'] for i in good_traj_inds])
    traj_weights = np.array([gmm_uq.trajectories[i]['weight'] for i in good_traj_inds])

    # Compute the sum in a vectorized way for each MC sample
    for t_ind, t in enumerate(tqdm(ts, desc='Computing KLD')):  # time loop
        Q_mu_i = np.zeros(K)  # Store Q(mu_i) for each MC sample

        for i in range(L):  # Loop over all good mixands
            mean = traj_means[i][t_ind]  # Mean for this mixand at this time
            cov = traj_covs[i][t_ind]  # Covariance for this mixand at this time
            weight = traj_weights[i]  # Weight for this mixand

            # Compute the Gaussian density for all MC samples at the current mixand's mean
            gauss_probs = multivariate_normal.pdf(MCData[:, :, t_ind], mean=mean, cov=cov, allow_singular=True)

            # Accumulate the weighted Gaussian probabilities for each MC sample
            Q_mu_i += weight * gauss_probs

        # Compute the log of Q(mu_i) and update KLD
        KLD[t_ind] -= (1 / K) * np.sum(np.log(Q_mu_i))  # Sum over all MC samples

    return KLD

def computeKLDInit_assumeDMM(MCData0, m0, P0):
    K = len(MCData0)
    KLD = -np.log(K)

    # Compute the sum
    # for j in trange(K, desc=f'Compute KLD {K} Samples'):  # MC sample loop
    for j in range(K):  # MC sample loop
        # Compute the Gaussian probability density
        Q_mu_i = multivariate_normal.pdf(MCData0[j, :], mean=m0, cov=P0, allow_singular=True)
        # Add the weighted term to LAM
        KLD = KLD - (1 / K) * np.log(Q_mu_i)

    return KLD

def computeCovFrobeniusNorm(MCCovs, covs, ts):
    # MCCovs is a ts length list of stateSize x stateSize covariance matrices
    # Assume this is for a single trajectory
    frob_norms = np.zeros(len(ts))
    for t_ind, t in enumerate(ts):  # time loop
        frob_norm = np.sqrt(np.sum((MCCovs[t_ind] - covs[t_ind]).flatten() ** 2)) / np.sqrt(np.sum((MCCovs[t_ind]).flatten() ** 2))
        frob_norms[t_ind] = frob_norm
    return frob_norms


def computeCvM(MCData, gmm_uq, ts):
    """
    Compute the Cramér–von Mises (CvM) metric between Monte Carlo samples
    and the GMM at each time step, using the 2-norm of marginal CvM statistics.

    For each dimension m, the marginal GMM CDF is:
        F_gmm^m(x) = sum_k weight_k * Phi((x - mean_k^m) / std_k^m)

    The per-dimension CvM statistic is:
        omega_m^2 = sum_j (F_gmm^m(x_j^m) - (2j-1)/(2K))^2 + 1/(12K)

    The overall metric is the 2-norm of the marginal CvM vector:
        ||omega^2||_2 = sqrt( sum_m (omega_m^2)^2 )

    Args:
        MCData:   np.ndarray of shape (K, d, T) — K samples, d dimensions, T time steps
        gmm_uq:   object with a `.trajectories` list; each trajectory dict has:
                      "times"   : iterable of time values
                      "means"   : list of mean vectors (one per time)
                      "covs"    : list of covariance matrices (one per time)
                      "weight"  : scalar mixture weight
        ts:       iterable of time values of length T

    Returns:
        CvM:  np.ndarray of shape (T,) — 2-norm of marginal CvM statistics at each time step
              (np.nan where no active trajectories exist)
    """
    from scipy.stats import norm

    K = MCData.shape[0]
    d = MCData.shape[1]
    CvM = np.zeros(len(ts))

    for t_ind, t in enumerate(tqdm(ts)):

        # -----------------------------
        # Active trajectories at time t
        # -----------------------------
        active_trajs = [
            traj for traj in gmm_uq.trajectories
            if t in traj["times"]
        ]

        if len(active_trajs) == 0:
            CvM[t_ind] = np.nan
            continue

        # Precompute per-trajectory local index, weight, marginal means and stds
        traj_params = []
        for traj in active_trajs:
            local_idx = list(traj["times"]).index(t)
            mean   = traj["means"][local_idx]          # shape (d,)
            cov    = traj["covs"][local_idx]            # shape (d, d)
            weight = traj["weight"]
            stds   = np.sqrt(np.diag(cov))             # marginal std per dimension

            traj_params.append((weight, mean, stds))

        # -----------------------------
        # Compute marginal GMM CDF values for all samples and dimensions
        # gmm_cdf_vals[j, m] = F_gmm^m(x_j^m)
        #                     = sum_k weight_k * Phi((x_j^m - mu_k^m) / sigma_k^m)
        # -----------------------------
        gmm_cdf_vals = np.zeros((K, d))

        for j in range(K):
            x = MCData[j, :, t_ind]   # shape (d,)

            for weight, mean, stds in traj_params:
                for m in range(d):
                    sigma_m = stds[m]
                    if sigma_m < 1e-12:
                        # Degenerate marginal: use step function CDF
                        cdf_m = 1.0 if x[m] >= mean[m] else 0.0
                    else:
                        cdf_m = norm.cdf(x[m], loc=mean[m], scale=sigma_m)

                    gmm_cdf_vals[j, m] += weight * cdf_m

        # -----------------------------
        # Per-dimension CvM statistic (Watson formula)
        # Sort each marginal CDF column, compare to uniform order statistics
        # -----------------------------
        omega_sq = np.zeros(d)
        i_vals = np.arange(1, K + 1)

        for m in range(d):
            u_m = np.sort(gmm_cdf_vals[:, m])
            omega_sq[m] = np.sum((u_m - (2 * i_vals - 1) / (2 * K)) ** 2) + 1 / (12 * K)

        # -----------------------------
        # 2-norm of the marginal CvM vector  ||omega^2||_2
        # -----------------------------
        CvM[t_ind] = np.linalg.norm(omega_sq)

    return CvM


def computeKL(MCData, gmm_uq, ts, stateSize=6):
    N_samples = 10_000
    KLs = []

    for t_ind, t in enumerate(tqdm(ts)):

        # -----------------------------
        # Find active trajectories at time t
        # -----------------------------
        active_trajs = []
        for traj in gmm_uq.trajectories:
            if t in traj["times"]:
                active_trajs.append(traj)

        if len(active_trajs) == 0:
            KLs.append(np.nan)
            continue

        # -----------------------------
        # Build Q-GMM at time t
        # -----------------------------
        weights_predefined = []
        means_predefined = []
        covariances_predefined = []

        for traj in active_trajs:
            # index within this trajectory's time array
            local_idx = list(traj["times"]).index(t)

            weights_predefined.append(traj["weight"])
            means_predefined.append(traj["means"][local_idx][:stateSize])
            covariances_predefined.append(traj["covs"][local_idx][:stateSize, :stateSize])

        weights_predefined = np.array(weights_predefined)
        means_predefined = np.array(means_predefined)
        covariances_predefined = np.array(covariances_predefined)

        # -----------------------------
        # Empirical P-GMM
        # -----------------------------
        gmm_p = GaussianMixture(
            n_components=27,
            covariance_type='full',
            random_state=42
        )
        gmm_p.fit(MCData[:, :, t_ind])

        # -----------------------------
        # Q-GMM (manual assembly)
        # -----------------------------
        gmm_q = GaussianMixture(
            n_components=len(active_trajs),
            covariance_type='full'
        )

        gmm_q.weights_ = weights_predefined
        gmm_q.means_ = means_predefined
        gmm_q.covariances_ = covariances_predefined

        # sklearn expects precisions_cholesky_, not precision of stacked list
        gmm_q.precisions_cholesky_ = np.array([
            np.linalg.cholesky(np.linalg.inv(cov))
            for cov in covariances_predefined
        ])

        # normalize weights just in case
        gmm_q.weights_ /= gmm_q.weights_.sum()

        # -----------------------------
        # KL estimate
        # -----------------------------
        samples, _ = gmm_p.sample(N_samples)

        log_p = gmm_p.score_samples(samples)
        log_q = gmm_q.score_samples(samples)

        KLs.append(np.mean(log_p - log_q))

    return KLs


def computeLAM_v1(MCData, gmm_uq, ts, stateSize=6):

    warnings.warn("Warning...there are multiple functions to compute LAM - check which one you want!")
    K = len(MCData)
    LAM = np.zeros(len(ts))

    for t_ind, t in enumerate(tqdm(ts)):

        # -----------------------------
        # Active trajectories at time t
        # -----------------------------
        active_trajs = [
            traj for traj in gmm_uq.trajectories
            if t in traj["times"]
        ]

        if len(active_trajs) == 0:
            LAM[t_ind] = np.nan
            continue

        # -----------------------------
        # Compute LAM sum
        # -----------------------------
        for j in range(K):
            x = MCData[j, :, t_ind]

            for traj in active_trajs:

                # local time index inside trajectory
                local_idx = list(traj["times"]).index(t)

                mean = traj["means"][local_idx][:stateSize]
                cov = traj["covs"][local_idx][:stateSize, :stateSize]
                weight = traj["weight"]

                gauss_prob = multivariate_normal.pdf(
                    x,
                    mean=mean,
                    cov=cov,
                    allow_singular=True
                )

                LAM[t_ind] += (1 / K) * weight * gauss_prob

    return LAM
