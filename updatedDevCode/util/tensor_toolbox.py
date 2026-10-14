import numpy as np
import time

def symmetrize(X, grps=None, ver=0):
    """
    Symmetrize a tensor with respect to specified groups of axes.

    Parameters
    ----------
    X : ndarray
        Input tensor.
    grps : list, tuple, or list of lists, optional
        Axes to symmetrize. Examples:
            None            -> all axes
            [0,1,2]         -> symmetrize over axes 0,1,2
            [[0,3],[1,2]]   -> independent symmetry groups
    ver : int
        0 = fast algorithm (default)
        1 = original MATLAB algorithm (not implemented)

    Returns
    -------
    Y : ndarray
        Symmetrized tensor.
    """

    X = np.asarray(X).copy()
    n = X.ndim
    sz = X.shape

    if grps is None:
        grps = [list(range(n))]
    elif isinstance(grps, (list, tuple, np.ndarray)):
        # If user passed a single list of axes, wrap it
        if len(grps) > 0 and not isinstance(grps[0], (list, tuple, np.ndarray)):
            grps = [list(grps)]
        else:
            grps = [list(g) for g in grps]
    else:
        raise ValueError("Invalid grps argument.")

    if ver != 0:
        raise NotImplementedError("Only ver=0 is implemented.")

    for i, thisgrp in enumerate(grps):

        # Check dimensions match
        d = sz[thisgrp[0]]
        if not all(sz[k] == d for k in thisgrp):
            raise ValueError("Dimension mismatch for symmetrization.")

        # Check no overlap
        for latergrp in grps[i + 1:]:
            if set(thisgrp) & set(latergrp):
                raise ValueError("Cannot have overlapping symmetries.")

        # Build multi-index array
        idx = np.indices(sz).reshape(n, -1).T

        # Sort indices within symmetry group
        classidx = idx.copy()
        classidx[:, thisgrp] = np.sort(classidx[:, thisgrp], axis=1)

        # Convert to linear indices
        linclassidx = np.ravel_multi_index(classidx.T, sz)

        data = X.ravel()

        # Skip if already symmetric
        if np.array_equal(data, data[linclassidx]):
            continue

        # Compute class sums and counts
        classsum = np.bincount(
            linclassidx,
            weights=data,
            minlength=data.size
        )

        classnum = np.bincount(
            linclassidx,
            minlength=data.size
        )

        avg = np.zeros_like(classsum, dtype=float)
        nonzero = classnum > 0
        avg[nonzero] = classsum[nonzero] / classnum[nonzero]

        X = avg[linclassidx].reshape(sz)

    return X

def eig_sshopm(
    A,
    shift="adaptive",
    maxits=1000,
    start=None,
    tol=1e-15,
    display=-1,
    concave=False,
    margin=1e-6,
):
    """
    Shifted Symmetric Higher Order Power Method.

    Returns
    -------
    lam, x, flag, its, x0
    FLAG = 0  => Succesfully terminated
    FLAG = -1 => Norm(X) = 0
    FLAG = -2 => Maximum iterations exceeded
    """

    N = A.shape[0]

    if isinstance(shift, str):
        adaptive = True
        shift = 0
    else:
        adaptive = False

    if start is None:
        x0 = np.random.uniform(-1, 1, N)
    else:
        x0 = np.asarray(start).astype(float)

    if np.linalg.norm(x0) == 0:
        raise ValueError("Zero starting vector")

    if shift != 0 and not isinstance(shift, str):
        concave = shift < 0

    x = x0 / np.linalg.norm(x0)

    lam = x @ ttsv(A, x, -1)

    if adaptive:
        shift = adapt_shift(A, x, margin, concave)

    flag = -2

    for its in range(1, maxits + 1):

        newx = ttsv(A, x, -1) + shift * x

        if concave:
            newx = -newx

        nx = np.linalg.norm(newx)

        if nx < np.finfo(float).eps:
            flag = -1
            break

        newx /= nx

        newlam = newx @ ttsv(A, newx, -1)

        if adaptive:
            newshift = adapt_shift(A, newx, margin, concave)
        else:
            newshift = shift

        if abs(newlam - lam) < tol:
            flag = 0

        if display > 0 and (flag == 0 or its % display == 0):
            print(
                f"{its:5d}  "
                f"lambda={newlam:15.8e}  "
                f"|dx|={np.linalg.norm(newx-x):10.3e}  "
                f"shift={shift:10.3e}"
            )

        x = newx
        lam = newlam
        shift = newshift

        if flag == 0:
            break

    return lam, x, flag, its, x0


def compute_eigenpairs_multiguess(
    tensors_all,
    stateSize,
    maxOrder,
    verbFlag=False,
):
    """
    Python version of compute_eigenpairs_multiguess
    """

    num_windows = len(tensors_all)

    lambdas = np.zeros((maxOrder, num_windows))
    evecs = np.zeros((maxOrder, num_windows, stateSize))
    flags = np.zeros((maxOrder, num_windows))

    all_lambdas = [[None for _ in range(num_windows)] for _ in range(maxOrder)]
    all_evecs = [[None for _ in range(num_windows)] for _ in range(maxOrder)]

    N = 100
    angle_tol = 1e-2
    lambda_tol = 1e-6

    totalIter = num_windows * maxOrder
    iterCount = 0

    start_time = time.time()

    for jj in range(num_windows):

        if verbFlag:
            print(f"Running window {jj+1} of {num_windows}")

        for ii in range(maxOrder):

            iterCount += 1
            frac = iterCount / totalIter

            elapsed = time.time() - start_time
            est_total = elapsed / frac if frac > 0 else 0
            remaining = est_total - elapsed

            if verbFlag:
                print(f"--> Running order {ii+1}")

            tensor_ii = tensors_all[jj][ii]

            # ============================================================
            # CASE 1: higher order (multi-start SS-HOPM)
            # ============================================================
            if ii > 0:

                lambda_list = np.zeros(N)
                x_list = [None] * N

                for kk in range(N):

                    x0 = np.random.randn(stateSize)
                    x0 = x0 / np.linalg.norm(x0)

                    shift_param = (stateSize - 1) * np.sum(np.abs(tensor_ii))

                    # lam, x, flag, its, x0
                    lambda_k, x_k, flag_k, num_its, x0_guess = eig_sshopm(
                        tensor_ii,
                        start=x0,
                        shift="adaptive",
                        tol=1e-6,
                        maxits=20000
                    )

                    if flag_k == 0:
                        lambda_list[kk] = lambda_k
                        x_list[kk] = x_k
                    else:
                        lambda_list[kk] = -10000
                        x_list[kk] = np.zeros_like(x_k)

                # sort by eigenvalue descending
                idx = np.argsort(lambda_list)[::-1]
                sorted_lambda = lambda_list[idx]
                sorted_x = [x_list[i] for i in idx]

                # deduplicate
                unique_x = []
                unique_lambda = []

                for i in range(N):

                    candidate_x = sorted_x[i]
                    candidate_lambda = sorted_lambda[i]

                    is_duplicate = False

                    same_lambda_idx = [
                        j for j, l in enumerate(unique_lambda)
                        if abs(candidate_lambda - l) < lambda_tol
                    ]

                    for j in same_lambda_idx:
                        dot_product = abs(np.dot(candidate_x, unique_x[j]))
                        if dot_product > np.cos(angle_tol):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique_x.append(candidate_x)
                        unique_lambda.append(candidate_lambda)

                # final sort
                idx2 = np.argsort(unique_lambda)[::-1]
                sorted_lambda_dedupe = np.array(unique_lambda)[idx2]
                sorted_x_dedupe = [unique_x[i] for i in idx2]

                evec = np.column_stack(sorted_x_dedupe)[:,0]
                lambda_val = sorted_lambda_dedupe[0]
                flag = 0

            # ============================================================
            # CASE 2: first order eigenproblem
            # ============================================================
            elif ii == 0:
                CGT = tensor_ii # must return 2D matrix

                evals, vecs = np.linalg.eig(CGT)

                idx = np.argsort(np.real(evals))[::-1]
                sorted_lambda = np.real(evals)[idx]
                vecs = vecs[:, idx]

                lambda_val = sorted_lambda[0]
                evec = vecs[:, 0]
                flag = 0

                sorted_lambda_dedupe = sorted_lambda
                sorted_x_dedupe = vecs

            # ============================================================
            # fallback
            # ============================================================
            else:

                lambda_val, evec, flag, num_its, x0_guess = eig_sshopm(
                    tensor_ii,
                    shift="adaptive"
                )

                sorted_lambda_dedupe = np.array([lambda_val])
                sorted_x_dedupe = np.array([evec])

            # store outputs
            all_lambdas[ii][jj] = sorted_lambda_dedupe
            all_evecs[ii][jj] = sorted_x_dedupe

            lambdas[ii, jj] = lambda_val
            evecs[ii, jj, :] = evec
            flags[ii, jj] = flag

            if verbFlag:
                print(f"  -> Solver flag: {flag}")

    return lambdas, evecs, flags, all_lambdas, all_evecs


def ttsv(A, x, n=0):
    """
    Tensor-times-same-vector in multiple modes.

    Parameters
    ----------
    A : ndarray
        Symmetric tensor of shape (N,)*d.
    x : ndarray
        Vector of length N.
    n : int, optional
        0  : contract all modes (scalar)
       -1 : leave first mode free (vector)
       -2 : leave first two modes free (matrix)
       -k : leave first k modes free

    Returns
    -------
    ndarray or scalar
    """

    if n > 0:
        raise ValueError("n must be <= 0")

    d = A.ndim
    sz = A.shape[0]

    dnew = -n               # number of remaining (free) modes
    drem = d - dnew         # number of contracted modes

    y = A

    for i in range(drem, 0, -1):
        yy = y.reshape(sz ** (dnew + i - 1), sz)
        y = yy @ x

    if dnew == 0:
        return float(y)

    elif dnew == 1:
        return y

    elif dnew == 2:
        return y.reshape(sz, sz)

    else:
        return y.reshape((sz,) * dnew)

def adapt_shift(A, x, tau, concave):
    """
    Adapt shift computation (Python version of MATLAB adapt_shift)

    Parameters
    ----------
    A : ndarray
        Tensor
    x : ndarray
        Vector
    tau : float
        Threshold parameter
    concave : bool
        Flag for concave vs convex case

    Returns
    -------
    alpha : float
    """

    m = A.ndim

    # Tensor-times-same-vector operation (mode -2 in MATLAB)
    Y = ttsv(A, x, -2)

    # Eigenvalues (MATLAB eig(Y))
    e = np.linalg.eigvals(Y)

    if concave:
        if np.max(e) <= -tau / (m**2 - m):
            alpha = 0.0
        else:
            alpha = -(tau / m) - ((m - 1) * np.max(e))
    else:
        if np.min(e) >= tau / (m**2 - m):
            alpha = 0.0
        else:
            alpha = (tau / m) - ((m - 1) * np.min(e))

    return float(alpha)