# Author: Grace Calkins
# Date: 07/08/12026
# Purpose: Compute a Karhunen Loeve Expansion of density
import numpy as np
from scipy import stats

def getEigenpairsKLE(h, rhoData, alpha, scaleVec = None):
    """
    Gets the eigenvectors, eigenvalues, and truncation level d for a given set of h and rho data.

    Based on cielago/src/UQ/getEProblem from Sam Albert.
    :param h: N_h x N_samp altitudes [m]
    :param rhoData: N_h x N_samp density [kg/m^3]
    :param alpha: accuracy parameter to truncating eigenvalues
    :param scaleVec: N_h vec, optional scaling vector for density

    :return evals: array of eigenvalues
    :return evecs: array of eigenvectors ordered same as evals array
    :return rhoSampMean: sample mean density profile, != GRAM model mean
    :return d: number of eigenvalues kept for density
    :return h: altitude profile array, m

    NOTE:
        d is output but nothing is truncated here, so to set d manually just
        use any value of alpha, call this func., then manually override d value
    """
    # scale
    if scaleVec is not None:
        rhoData *= scaleVec[:, None]

    # get centered density about sample mean
    rhoSampMean = np.mean(rhoData, axis=1)
    rhoCentered = rhoData - rhoSampMean[:, None]

    # get sample covariance matrix
    covmat = np.cov(rhoCentered)  # biased, by default, for sample estimate

    # eigenvalue decomposition of the sample covariance matrix, and sort desc.
    evals, evecs = np.linalg.eig(covmat)
    idx = evals.argsort()[::-1]  # descending order
    evals = evals[idx].real
    evecs = evecs[:, idx].real

    # find d, number of e-vals needed to get at least alpha accuracy
    rat = 0
    d = 0
    while rat < alpha:
        d += 1
        rat = np.sum(evals[0:d]) / np.sum(evals)

    return evals, evecs, rhoSampMean, d, h

def getKLEfun(evals, evecs, mean, d, hvec, perts = False, rhomean = None, scaleVec = None, rng = None):
    '''
    takes eigenproblem input, returns a function handle to compute rho via KLE

    :param evals: array of eigenvalues
    :param evecs: array of eigenvectors ordered same as evals array
    :param mean: profile of mean density, should be SAMPLE mean
    :param d: number of eigen terms to include before truncation
    :param hvec: given altitude profile, m
    :param rhomean: mean density profile, required if perts == True
    :param perts: if true, takes expansion on perturbations and returns density

    :return getKLErho: function handle to interpolate from KLE density given alt.
    :return Ys: values of the standard normal random variables in this instance
    '''
    # Get random generator if none provided
    random_state = rng if rng is not None else np.random.default_rng()

    # generate d i.i.d. standard normal variable instances
    Ys = stats.norm.rvs(size=d, random_state=rng)

    # get density at each given altitude step
    rhovec = []
    for i in range(hvec.size):
        rho = mean[i] + np.sum(np.sqrt(evals[0:d]) * evecs[i, 0:d] * Ys)

        # re-scale if necessary
        if scaleVec is not None:
            rho /= scaleVec[i]

        # convert from perturbations to density if necessary
        if perts:
            rho = (rho + 1) * rhomean[i]

        rhovec.append(rho)
    rhovec = np.asarray(rhovec)

    # ensure sorted in ascending altitude order for interpolation
    rhovec = rhovec[np.argsort(hvec)]
    hvec = hvec[np.argsort(hvec)]

    # now define a function to interpolate rhovec for a given h value
    def getKLErho(h):
        '''
        function for a single instance of the density KLE
        '''
        rho = np.interp(h, hvec, rhovec)
        return np.where(rho < 0, 0, rho)
        # return rho  # WARNING: only use this option when comparing density charts, not to simulate

    return getKLErho, Ys