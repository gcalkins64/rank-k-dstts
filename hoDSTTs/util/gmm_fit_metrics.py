import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm, trange
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import ray
from ray import remote, get


def computeLAM(MCData, gmm_uq, ts):
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

