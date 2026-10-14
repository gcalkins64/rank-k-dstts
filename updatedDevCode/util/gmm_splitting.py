# Author: Grace Calkins
# Date: 12/18/2024
# Purpose: Houses functions for gmm splitting methods

from .stt import *


def gmmSplittingLib(numComponents):
    if numComponents == 3:
        iis = [1,2,3]
        omega_tildes = [0.2252246249136750, 0.5495507501726501, 0.2252246249136750]
        m_tildes = [-1.057515461475881, 0, 1.057515461475881]
        sigma_tildes = [0.6715662886640760, 0.6715662886640760, 0.6715662886640760]
    elif numComponents == 4:
        iis = [1,2,3,4]
        omega_tildes = [0.1238046161618835, 0.3761953838381165, 0.3761953838381165, 0.1238046161618835]
        m_tildes = [-1.437464136328835, -0.455886223973523, 0.455886223973523, 1.437464136328835]
        sigma_tildes = [0.5276007226175397, 0.5276007226175397, 0.5276007226175397, 0.5276007226175397 ]
    elif numComponents == 5:
        iis = [1,2,3,4,5]
        omega_tildes = [0.0763216490701042, 0.2474417859474436, 0.3524731299649044, 0.2474417859474436, 0.0763216490701042]
        m_tildes = [-1.689972911128078, -0.800928383429953, 0, 0.800928383429953, 1.689972911128078]
        sigma_tildes = [0.44225553866310084, 0.4422555386310084, 0.4422555386310084, 0.44225553866310084, 0.4422555386310084]
    else:
        raise ValueError('Invalid number of mixands')
    return iis, omega_tildes, m_tildes, sigma_tildes

def splitMixandsCovariance(numComponents, P, mean):
    """
    Splits a given multivariate unimodal Gaussian mixture model into a given number
    of components along the direction of the largest eigenvalue of the covariance matrix.
    """
    if np.size(P) == 1:
        evals = np.array([1])
        evecs = np.array([1])
    else:
        evals, evecs = np.linalg.eig(P)
    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_evec = evecs[:, max_eval_ind]
    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)
    means = []
    weights = []
    covs = []
    Lambda_k = np.diag(evals)
    Lambda_k[max_eval_ind, max_eval_ind] = sigma_tildes[0]**2 * max_eval
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        means.append(mean + np.sqrt(max_eval) * m_tildes[ii] * splitting_evec)
        covs.append(evecs @ Lambda_k @ evecs.T)

    return weights, means, covs, splitting_evec

def splitMixandsCovarianceDir(numComponents, P, mean, R):
    """
    Splits a given multivariate unimodal Gaussian mixture model into a given number
    of components along a given direction
    """
    if np.size(P) == 1:
        evals = np.array([1])
        evecs = np.array([1])
    else:
        evals, evecs = np.linalg.eig(P)
    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_evec = evecs[:, max_eval_ind]
    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)
    means = []
    weights = []
    covs = []
    Lambda_k = np.diag(evals)
    Lambda_k[max_eval_ind, max_eval_ind] = sigma_tildes[0]**2 * max_eval
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        means.append(mean + np.sqrt(max_eval) * m_tildes[ii] * R)
        covs.append(evecs @ Lambda_k @ evecs.T)

    return weights, means, covs, R


def splitMixandsDSTT(numComponents, P, mean, R):
    """
    Splits a given multivariate unimodal Gaussian mixture model into a given number
    of components along the direction specified by the R vector
    """
    mu_y = R @ mean
    R_vec = np.abs(R[:, np.newaxis].T)
    sigma_y2 = R_vec @ P @ R_vec.T
    sigma_y = np.sqrt(sigma_y2[0,0])

    # Get splits (STANDARD NORMAL)
    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)

    # Convert scalar splits out of standard normal
    weights = []
    scalar_means = []
    scalar_vars = []
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        scalar_means.append(sigma_y*m_tildes[ii] + mu_y)
        scalar_vars.append(sigma_y**2 * sigma_tildes[ii]**2)

    # Convert scalar splits back to multivariate splits
    means = []
    covs = []
    for ii in range(numComponents):
        pseudo_inv = R_vec.T @ np.linalg.inv(R_vec @ R_vec.T)
        means.append(np.squeeze((pseudo_inv * scalar_means[ii] - pseudo_inv * mean[:,np.newaxis]) + mean[:,np.newaxis]))
        # means.append(np.squeeze((pseudo_inv * scalar_means[ii])))
        new_var = scalar_vars[ii]
        # new_var = pseudo_inv.T @ P @ pseudo_inv * scalar_vars[ii]
        P_new = P.copy()
        one_ind = np.where(R == 1)[0][0]
        P_new[one_ind,one_ind] = new_var # TODO
        # Also scale all off diagonal terms in the one_ind rows and columns by new var / old var
        scale = np.sqrt(new_var) / np.sqrt(P[one_ind, one_ind])
        for ind in range(len(mean)):
            if ind != one_ind:
                Pij = P_new[one_ind, ind] * scale
                Pji = P_new[ind, one_ind] * scale
                P_new[one_ind, ind] = Pij
                P_new[ind, one_ind] = Pji
        # covs.append(new_var * np.outer(pseudo_inv, pseudo_inv) + P)
        covs.append(P_new)

    return weights, means, covs, R


    # dP = np.diag(P)
    # one_ind = np.where(R == 1)[0][0]
    # evals, evec = np.linalg.eig(P)
    #
    # # evals[one_ind] = scalar_vars[0] * (evals[one_ind])
    # max_eval = np.max(evals)
    # evals[one_ind] = sigma_tildes[0]**2 * (evals[one_ind])
    # Lambda = np.diag(evals)
    # P_split = evec @ Lambda @ evec.T

    # P_split = P.copy()
    # P_split[one_ind, one_ind] = scalar_vars[0] * dP[one_ind]

    # zero out mean position corresponding to the one index
    # mean[one_ind] = 0

def splitMixandsDSTTCGT(numComponents, P, mean, R, DSTTs, index, params, ts):
    """
    Splits a given multivariate unimodal Gaussian mixture model into a given number
    of components along the direction specified by the R vector using the eigenvalues and eigenvectors of the CGT until the end of the trajectory
    """
    # Get CGT until end of trajectory
    # STM_tk_to_tf = np.ones((len(mean), len(mean)))
    # for ind in range(index, len(DSTTs)):
    #     STM_tk_to_tf = STM_tk_to_tf @ DSTTs[ind,0:len(mean)**2].reshape((len(mean), len(mean)))
    #
    #     CGT = STM_tk_to_tf @ STM_tk_to_tf.T

    stateSize = len(mean)
    Rp = gravParams[params['planetName']].get('R_eq')
    g0 = gravParams[params['planetName']].get('g_surf')
    tspan = [ts[index] / np.sqrt(Rp / g0),
             ts[-1] / np.sqrt(Rp / g0)]

    # get STM from time 0 to time f
    Phi_stack = np.hstack((mean, np.eye(stateSize).flatten()))

    X_stack_sol = solve_ivp(
        lambda t, Phi_stack: aerocaptureSTTsODE_general(t, Phi_stack, 1, stateSize, params, '3DOFNonDim'),
        tspan,
        Phi_stack, 'RK45', rtol=1e-10, atol=1e-12)

    stm = X_stack_sol.y[stateSize:, -1].reshape((stateSize, stateSize))
    CGT = stm.T @ stm  # compute Cauchy-Green tensor
    evals, evecs = np.linalg.eig(CGT)

    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_direction = evecs[:, max_eval_ind]
    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)
    means = []
    weights = []
    covs = []
    Lambda_k = np.diag(evals)
    Lambda_k[max_eval_ind, max_eval_ind] = sigma_tildes[0]**2 * max_eval
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        means.append(mean + np.sqrt(max_eval) * m_tildes[ii] * splitting_direction)
        covs.append(evecs @ Lambda_k @ evecs.T)

    return weights, means, covs
def splitMixandsCGT(numComponents, Phi, mean):
    """
    Splits a given multivariate unimodal Gaussian mixture model into a given number
    of components along the direction of the largest eigenvalue of the covariance matrix.
    """
    CGT = Phi.T @ Phi
    evals, evecs = np.linalg.eig(CGT)
    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_evec = evecs[:, max_eval_ind]

    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)
    means = []
    weights = []
    covs = []
    Lambda_k = np.diag(evals)
    Lambda_k[max_eval_ind, max_eval_ind] = sigma_tildes[0]**2 * max_eval
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        means.append(mean + np.sqrt(max_eval) * m_tildes[ii] * splitting_evec)
        covs.append(evecs @ Lambda_k @ evecs.T)

    return weights, means, covs

def splitMixandsArbitaryDirection(numComponents, P, mean, aVec):
    """
    Splits a given multivariate Gaussian along an abitrary direction following Vittaldev Method
    """
    # Get cholesky decomposition of the covariance matrix
    S = np.linalg.cholesky(P) # should be lower triangular matrix

    # Get the a_star direction
    aStar = np.linalg.inv(S) @ aVec / np.linalg.norm(np.linalg.inv(S) @ aVec)

    iis, omega_tildes, m_tildes, sigma_tildes = gmmSplittingLib(numComponents)
    means = []
    weights = []
    covs = []
    for ii in range(numComponents):
        weights.append(omega_tildes[ii])
        means.append(mean + m_tildes[ii] * S @ aStar)
        covs.append(S @ (np.eye(len(mean)) + (sigma_tildes[ii]**2 - 1) * np.outer(aStar, aStar)) @ S.T)

    return weights, means, covs

def getSplittingEvecCGT(Phi):
    CGT = Phi.T @ Phi
    evals, evecs = np.linalg.eig(CGT)
    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_evec = evecs[:, max_eval_ind]
    return splitting_evec

def getSplittingEvecCov(P):
    evals, evecs = np.linalg.eig(P)
    max_eval = np.max(evals)
    max_eval_ind = np.argmax(evals)
    splitting_evec = evecs[:, max_eval_ind]
    return splitting_evec


def solc_R_score(DSTT_list, R_nd, P_nd, whiten=True):
    """R-restricted (W-)US-SOLC: magnitude of DSTT's own psi_2 = G^(2)[R,R],
    evaluated at the 1-sigma displacement along R. whiten=True normalizes by
    the linearly-predicted output covariance Pz = Phi P Phi^T (per eq. 79),
    giving a unit-free score you can compare across timesteps/EOMs."""
    Phi = DSTT_list[0]
    psi_2 = np.squeeze(DSTT_list[1])  # (n,1,1) -> (n,)
    sigma_xi_sq = float(np.einsum('i,ij,j->', np.squeeze(R_nd), P_nd, np.squeeze(R_nd)))
    g2_RR = sigma_xi_sq * psi_2
    if not whiten:
        return float(np.linalg.norm(g2_RR))
    Pz = Phi @ P_nd @ Phi.T
    Pz_inv = np.linalg.inv(Pz)
    return float(g2_RR @ Pz_inv @ g2_RR) / Phi.shape[0]

