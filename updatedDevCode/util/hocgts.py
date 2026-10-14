# Author: Grace Calkins
# Date: 07/08/2026
# Purpose: compute and save HOCGTs for various analysis


import os
from .stt import reshape_STT
import numpy as np
from pyaudi import gdual_double as gdual # type: ignore
from .params_planet import gravParams, G
from .aerocapture import *
from .tensor_toolbox import symmetrize
from scipy.io import loadmat, savemat

def getHOCGTs_nonSym(STT_list, order, stateSize):
    STTs = reshape_STT(STT_list, order, stateSize)
    HOCGTS_nonsym = []

    if order >= 1:
        CGT = STTs[0].T @ STTs[0]  # second order STM CGT
        HOCGTS_nonsym.append(CGT)
    if order >= 2:  # need order of three for next CGT
        CGT = np.einsum('ab, acd -> bcd', STTs[0], STTs[1])
        HOCGTS_nonsym.append(CGT)
    if order >= 3:
        CGT = 1 / 3 * np.einsum('ab, acde -> bcde', STTs[0], STTs[2]) + 1 / 4 * np.einsum('abc, ade -> bcde',STTs[1], STTs[1])
        HOCGTS_nonsym.append(CGT)
    if order >= 4:
        CGT = 1 / 12 * np.einsum('ab, acdef -> bcdef', STTs[0], STTs[3]) + 1 / 6 * np.einsum('abc, adef -> bcdef',STTs[1], STTs[2])
        HOCGTS_nonsym.append(CGT)

    return HOCGTS_nonsym

def getSCGTs_nonSym(STT_list, order, S):
    # S is the selection matrix
    m,n = S.shape
    STTs = reshape_STT(STT_list, order, n)
    SCGTs_nonsym = []

    if order >= 1:
        S2 = STTs[0].T @ S.T @ S @ STTs[0] # second order STM CGT
        SCGTs_nonsym.append(S2)
    if order >= 2:  # need order of three for next CGT
        S_tilde_3 = np.einsum('ij,jk,ij,jlm->klm', S, STTs[0], S, STTs[1])
        SCGTs_nonsym.append(S_tilde_3)
    if order >= 3:
        term1 = (1 / 3) * np.einsum('ij,jk,ij,jlmn->klmn', S, STTs[0], S, STTs[2])
        term2 = (1 / 4) * np.einsum('ij,jkl,ij,jmn->klmn', S, STTs[1], S, STTs[1])
        S_tilde_4 = term1 + term2
        SCGTs_nonsym.append(S_tilde_4)
    if order >= 4:
        # Compute the two terms of the equation
        term1 = (1 / 12) * np.einsum('ij,jk,ij,jlmno->klmno', S, STTs[0], S, STTs[3])
        term2 = (1 / 6) * np.einsum('ij,jklm,ij,jno->klmno', S, STTs[2], S, STTs[1])

        # Combine the terms to compute S_tilde_5
        S_tilde_5 = term1 + term2

        SCGTs_nonsym.append(S_tilde_5)

    return SCGTs_nonsym

def get_Eta_Tensors(order, Xf, q_func, params):
    stateSize = len(Xf)
    eta_first = np.zeros((1, stateSize))
    eta_second = np.zeros((1, stateSize, stateSize))
    eta_third = np.zeros((1, stateSize, stateSize, stateSize))
    eta_fourth = np.zeros((1, stateSize, stateSize, stateSize, stateSize))

    # # set up autodifferentiation stuff with gduals
    # g_order = 4
    # if 'Ln' not in params['eomTag']:
    #     X_i_g = [gdual(Xf[0], "r", order), gdual(Xf[1], "theta", order), gdual(Xf[2], "phi", order),gdual(Xf[3], "V", order), gdual(Xf[4], "gamma", order), gdual(Xf[5], "psi", order)]
    #     dual_str = ["r", "theta", "phi", "V", "gamma", "psi"]
    # else:
    #     X_i_g = [gdual(Xf[0], "r", order), gdual(Xf[1], "theta", order), gdual(Xf[2], "phi", order),gdual(Xf[3], "V", order), gdual(Xf[4], "gamma", order), gdual(Xf[5], "psi", order), gdual(Xf[6], "ln_rho", order)]
    #     dual_str = ["r", "theta", "phi", "V", "gamma", "psi", 'ln_rho']

    # # Evaluate the q function at the final state
    # mu_nd = gravParams[params['planetName']]['mu']  / gravParams[params.get('planetName')].get('R_eq') ** 3 * np.sqrt(gravParams[params.get('planetName')].get('R_eq') / gravParams[params.get('planetName')].get('g_surf'))**2
    # F = q_func(X_i_g, mu_nd)
    #
    # # Compute partials of the q function with respect to the final state
    # for a in range(stateSize):
    #     A_first_gdual = F.partial(dual_str[a])
    #     eta_first[0, a] = A_first_gdual.constant_cf
    #     if order >= 2:
    #         for b in range(stateSize):
    #             A_second_gdual = A_first_gdual.partial(dual_str[b])
    #             eta_second[0, a, b] = A_second_gdual.constant_cf
    #             if order >= 3:
    #                 for c in range(stateSize):
    #                     A_third_gdual = A_second_gdual.partial(dual_str[c])
    #                     eta_third[0, a, b, c] = A_third_gdual.constant_cf
    #                     if order >= 4:
    #                         for d in range(stateSize):
    #                             A_fourth_gdual = A_third_gdual.partial(dual_str[d])
    #                             eta_fourth[0, a, b, c, d] = A_fourth_gdual.constant_cf

    # Also compute the first through third order partials symbolically and compare!
    mu_nd = gravParams[params['planetName']]['mu'] / gravParams[params.get('planetName')].get('R_eq') ** 3 * np.sqrt(gravParams[params.get('planetName')].get('R_eq') / gravParams[params.get('planetName')].get('g_surf')) ** 2
    if q_func == energy_func:
        eta_first = energy_gradient(*Xf, mu_nd).reshape((1, stateSize))
        eta_second = energy_hessian(*Xf, mu_nd).reshape((1, stateSize, stateSize))
        eta_third = energy_third(*Xf, mu_nd).reshape((1, stateSize, stateSize, stateSize))

        # # Check if symbolic and autodifferentiated results are the same
        # print(np.array_equal(eta_first_sym, eta_first))
        # print(np.array_equal(eta_second_sym, eta_second))
        # print(np.array_equal(eta_third_sym, eta_third))

    if q_func == apoRad_func:
        eta_first = ra_gradient(*Xf, mu_nd).reshape((1, stateSize))
        eta_second = ra_hessian(*Xf, mu_nd).reshape((1, stateSize, stateSize))
        eta_third = ra_third(*Xf, mu_nd).reshape((1, stateSize, stateSize, stateSize))

    if q_func == pos_func:
        eta_first = np.zeros(stateSize)
        eta_first[0] = 1
        eta_first = eta_first.reshape((1, stateSize))
        eta_second = np.zeros((1, stateSize, stateSize))
        eta_third = np.zeros((1, stateSize, stateSize, stateSize))

    return eta_first, eta_second, eta_third, eta_fourth

def getQCGTs_nonSym(STT_list, order, Xf, q_func, params, etas):
    # S is the selection matrix
    STTs = reshape_STT(STT_list, order, len(Xf))
    QCGTs_nonsym = []

    eta1, eta2, eta3, eta4 = etas

    # D's are the coefficients of the Taylor series expansion for Q
    # Q's are the coefficients of the squared Taylor series expansion
    if order >= 1:
        # Get coefficients of the q expansion
        D1 = np.einsum('ij,jg->ig', eta1, STTs[0])
        Q2 = np.einsum('ig,il->gl', D1, D1)  # second order qCGT
        QCGTs_nonsym.append(Q2)
    if order >= 2:  # need order of three for next CGT
        D2_a = np.einsum('ijl,lh,jg->igh', eta2, STTs[0], STTs[0])
        D2_b = np.einsum('ij,jgh->igh', eta1, STTs[1])
        D2 = D2_a + D2_b
        Q_tilde_3 = np.einsum('ig,ilm->glm', D1, D2)
        QCGTs_nonsym.append(Q_tilde_3)
    if order >= 3:
        D3_a = np.einsum('ijlm,mz,lh,jg->ighz', eta3, STTs[0], STTs[0], STTs[0])
        D3_b = np.einsum('ijl,lhz,jg->ighz', eta2, STTs[1], STTs[0])
        D3_c = np.einsum('ijl,lh,jgz->ighz', eta2, STTs[0], STTs[1])
        D3_d = np.einsum('ijl,lz,jgh->ighz', eta2, STTs[0], STTs[1])
        D3_e = np.einsum('ij,jghz->ighz', eta1, STTs[2])
        D3 = D3_a + D3_b + D3_c + D3_d + D3_e
        Q_tilde_4a = 1/3*np.einsum('ig,ilmn->glmn', D1, D3)
        Q_tilde_4b = 1/4*np.einsum('igh,iln->ghln', D2, D2)
        Q_tilde_4 = Q_tilde_4a + Q_tilde_4b
        QCGTs_nonsym.append(Q_tilde_4)
    if order >= 4:
        raise NotImplementedError("Fourth order qCGT not implemented yet")

    return QCGTs_nonsym

# Grace removed this old derivation on 9/17/2025 in favor of the new derivation (see lab notebook)
# def getQCGTs_nonSym(STT_list, order, Xf, q_func, params, etas):
#     # S is the selection matrix
#     STTs = reshape_STT(STT_list, order, len(Xf))
#     QCGTs_nonsym = []
#
#     eta1, eta2, eta3, eta4 = etas
#
#     if order >= 1:
#         Q2 = np.einsum('ij,jk,ij,jl->kl',  eta1, STTs[0], eta1, STTs[0])
#         QCGTs_nonsym.append(Q2)
#     if order >= 2:  # need order of three for next CGT
#         term1 = np.einsum('im,mj,inm,nl,ok->jkl', eta1, STTs[0], eta2, STTs[0], STTs[0])
#         term2 = np.einsum('im,ml,in,njk->jkl', eta1, STTs[0], eta1, STTs[1])
#         Q_tilde_3 = term1 + term2
#         QCGTs_nonsym.append(Q_tilde_3)
#     if order >= 3:
#         # First term
#         term1_a = (1 / 4) * np.einsum('ilm,lk,mj,ino,nm,ol->jklm', eta2, STTs[0], STTs[0], eta2, STTs[0], STTs[0])
#         term1_b = (1 / 4) * np.einsum('ilm,lk,mj,in,nml->jklm', eta2, STTs[0], STTs[0], eta1, STTs[1])
#         term1_c = (1 / 4) * np.einsum('il,lkj,ino,nm,ol->jklm', eta1, STTs[1], eta2, STTs[0], STTs[0])
#         term1_d = (1 / 4) * np.einsum('il,lkj,in,nml->jklm', eta1, STTs[1], eta1, STTs[1])
#
#         # Second term
#         term2a = (1 / 3) * np.einsum('inom,ik,ol,nk,mj,kj->jklm',eta3, eta1, STTs[0], STTs[0], STTs[0], STTs[0])
#         term2b = (1 / 3) * np.einsum('inm,ik,kj,njl,mk->jklm',eta2,eta1,STTs[0],STTs[1],STTs[0])
#         term2c = (1 / 3) * np.einsum('inm,ik,kj,nj,mlk->jklm',eta2,eta1,STTs[0],STTs[0],STTs[1])
#         term2d = (1 / 3) * np.einsum('inm,ik,kj,nl,mjk->jklm',eta2,eta1,STTs[0],STTs[0],STTs[1])
#         term2e = (1 / 3) * np.einsum('im,ik,mlkj,kj->jklm',eta1,eta1,STTs[2],STTs[0])
#
#         # Sum all terms
#         Q_tilde_4 = term1_a + term1_b + term1_c + term1_d + term2a + term2b + term2c + term2d + term2e
#
#         QCGTs_nonsym.append(Q_tilde_4)
#     if order >= 4:
#         # Term 1
#         term1a = (1 / 6) * np.einsum('iop,isrq,ok,pj,sn,rm,ql->jklmn', eta2, eta3, STTs[0], STTs[0], STTs[0], STTs[0], STTs[0], optimize=True)
#         term1b1 = (1 / 6) * np.einsum('iop,isrq,ok,pj,sn,rm,ql->jklmn', eta2, eta3, STTs[0], STTs[0], STTs[0], STTs[0],
#                                       STTs[0], optimize=True)
#         term1b2 = (1 / 6) * np.einsum('iop,isrq,ok,pj,sn,rm,ql->jklmn', eta2, eta3, STTs[0], STTs[0], STTs[0], STTs[0],
#                                       STTs[0], optimize=True)
#         term1b3 = (1 / 6) * np.einsum('iop,isrq,ok,pj,sn,rm,ql->jklmn', eta2, eta3, STTs[0], STTs[0], STTs[0], STTs[0],
#                                       STTs[0], optimize=True)
#         term1c = (1 / 6) * np.einsum('iop,iq,ok,pj,qnml->jklmn', eta2, eta1, STTs[0], STTs[0], STTs[2], optimize=True)
#         term1d = (1 / 6) * np.einsum('io,isrq,okj,sn,rm,ql->jklmn', eta1, eta3, STTs[1], STTs[0], STTs[0], STTs[0], optimize=True)
#         term1e = (1 / 6) * np.einsum('io,irq,okj,rln,qm->jklmn', eta1, eta2, STTs[1], STTs[1], STTs[0]) + (1 / 6) * np.einsum('io,irq,okj,rl,qmn->jklmn', eta1, eta2, STTs[1], STTs[0], STTs[1]) + (1 / 6) * np.einsum('io,irq,okj,rn,qlm->jklmn', eta1, eta2, STTs[1], STTs[0], STTs[1], optimize=True)
#         term1f = (1 / 6) * np.einsum('io,iq,okj,qnml->jklmn', eta1, eta1, STTs[1], STTs[2], optimize=True)
#
#         # Term 2
#         term2a = (1 / 12) * np.einsum('io,oj,ipqrs,rk,sn,pl,qm->jklmn', eta1, STTs[0], eta4, STTs[0], STTs[0], STTs[0], STTs[0], optimize=True)
#         term2b = (1 / 12) * np.einsum('io,oj,irqp,rnk,qm,pl->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2c = (1 / 12) * np.einsum('io,oj,irqp,pnl,rk,qm->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2d = (1 / 12) * np.einsum('io,oj,irqp,qmn,rk,pl->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2e = (1 / 12) * np.einsum('io,oj,iprs,pkm,sn,rl->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2f = (1 / 12) * np.einsum('io,oj,ipr,rkmn,rl->jklmn', eta1, STTs[0], eta2, STTs[2], STTs[0], optimize=True)
#         term2g = (1 / 12) * np.einsum('io,oj,ipr,pkm,rln->jklmn', eta1, STTs[0], eta2, STTs[1], STTs[1], optimize=True)
#         term2h = (1 / 12) * np.einsum('io,oj,iprs,rlm,sn,pk->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2i = (1 / 12) * np.einsum('io,oj,ipr,rlmn,pk->jklmn', eta1, STTs[0], eta2, STTs[2], STTs[0], optimize=True)
#         term2j = (1 / 12) * np.einsum('io,oj,ipr,rlm,pkn->jklmn', eta1, STTs[0], eta2, STTs[1], STTs[1], optimize=True)
#         term2k = (1 / 12) * np.einsum('io,oj,iprs,rkl,sn,pm->jklmn', eta1, STTs[0], eta3, STTs[1], STTs[0], STTs[0], optimize=True)
#         term2l = (1 / 12) * np.einsum('io,oj,ipr,rkln,pm->jklmn', eta1, STTs[0], eta2, STTs[2], STTs[0], optimize=True)
#         term2m = (1 / 12) * np.einsum('io,oj,ipr,rkl,pmn->jklmn', eta1, STTs[0], eta2, STTs[1], STTs[1], optimize=True)
#         term2n = (1 / 12) * np.einsum('io,oj,irs,rmlk,sn->jklmn', eta1, STTs[0], eta2, STTs[2], STTs[0], optimize=True)
#         term2o = (1 / 12) * np.einsum('io,oj,ir,rklmn->jklmn', eta1, STTs[0], eta1, STTs[3], optimize=True)
#
#         Q_tilde_5 = term1a + term1b1 + term1b2 + term1b3 + term1c + term1d + term1e + term1f + term2a + term2b + term2c + term2d + term2e + term2f + term2g + term2h + term2i + term2j + term2k + term2l + term2m + term2n + term2o
#
#         QCGTs_nonsym.append(Q_tilde_5)
#
#     return QCGTs_nonsym

def getCGTEigenpairs(STM_flat, stateSize):
    STM = STM_flat.reshape((stateSize, stateSize))
    CGT = STM.T @ STM
    evals, evecs = np.linalg.eig(CGT)
    idx = np.argsort(np.abs(evals))[::-1]
    evals_sorted = evals[idx]
    evecs_sorted = evecs[:, idx]
    # evecs = evecs_sorted[:, idx]

    # return evals, evecs
    return evals_sorted, evecs_sorted

def get_CGT_R_matrix(prefix, regimeTag, dataPath):
    # Load in hoeigenpair datat from MATLAB
    eval_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_evals.mat")
    evec_data = loadmat(f"{dataPath}/{prefix}_{regimeTag}_HOho_Rs.mat")

    all_evals = eval_data['ho_evals']
    all_evecs = evec_data['ho_Rs']

    # Want to make evals and evecs lists of arrays
    evecs = []
    for j in range(all_evals.shape[0]):
        this_evals = []
        this_evecs = []
        for i in range(all_evals.shape[1]):
            if j == 0:
                this_evals.append(np.diag(all_evals[j, i]).flatten())
                this_evecs.append(all_evecs[j, i])
            else:
                this_evals.append(all_evals[j, i].flatten())
                that_evec = []
                for k in range(all_evecs[j, i].shape[1]):
                    that_evec.append(np.squeeze(all_evecs[j, i][0, k]))
                this_evecs.append(np.array(that_evec).T)

        evecs.append(this_evecs)

    # Evecs is a [order] size list of lists with [number of time domains] size where each element is a [stateSize, arbitrary size] np.array of evecs where each column is an evec

    # Need to convert to a times x order-1 x state size matrix
    # We want both "orders" to be the second order CGT eigenpair from the HOCGT
    Rs = np.zeros((len(evecs[0]), len(evecs)-1, evecs[0][0].shape[0]))  # times x order x state size
    for order in range(len(evecs) - 1):
        for time in range(len(evecs[order])):
            Rs[time, order, :] = evecs[0][time][:, 0]  # Take the first eigenvector (for maximum eigenvalue)

    return Rs

# Get first and second direction and make the R array
def getCGTEigenpairsMulti(STTs, nPairs, stateSize):
    d1, d2 = np.shape(STTs)

    v_max_sw = []
    l_sw = []
    for ii in range(d1):
        STT_list = reshape_STT(STTs[ii, :],1, stateSize)
        STM = STT_list[0]
        CGT = STM.T @ STM
        evals, evecs = np.linalg.eig(CGT)
        idx = np.argsort(np.abs(evals))[::-1]
        evals_sorted = evals[idx]
        evecs_sorted = evecs[:, idx]
        v_max = np.zeros((nPairs, stateSize))
        for n in range(nPairs):
            v_max[n,:] = (evecs_sorted[:,n].T)
        v_max_sw.append(v_max)
        l_sw.append(evals_sorted)

    return np.array(v_max_sw)



def get_and_save_HOCGT(STTs, cgtType, timeDomain, runTag, dataPath, params, order, Smat=None, Xs=None, q_func=None, savePrefix=None):
    HOCGTs_nonsym_list = []
    for ii in range(len(STTs) - 1):
        if cgtType == 'HOCGT':
            HOCGT_nonsym = getHOCGTs_nonSym(STTs[ii + 1], order=order, stateSize=7)
        elif cgtType == 'sCGT':
            HOCGT_nonsym = getSCGTs_nonSym(STTs[ii + 1], order=order, S=Smat)
        elif cgtType == 'qCGT':
            # Make sure the Xk we are evaluating is the correct one for the time domain
            if timeDomain == 't0tk': # Evaluate at Xk
                X_eval = Xs[ii+1]
            elif timeDomain == 'tktkp1':  # Evaluate at Xk+1
                X_eval = Xs[ii+1]
            elif timeDomain == 'tktf': # Evaluate at Xf
                X_eval = Xs[-1]
            elif timeDomain == 'regime': # Evaluate at X_regime end
                X_eval = Xs[ii+1]
            elif timeDomain == 't0tf': # Evaluate at Xf
                X_eval = Xs[-1]
            eta1, eta2, eta3, eta4 = get_Eta_Tensors(order, X_eval, q_func, params)
            etas = [eta1, eta2, eta3, eta4]
            HOCGT_nonsym = getQCGTs_nonSym(STTs[ii + 1], order=order, Xf=X_eval, q_func=q_func, params=params, etas=etas)
        else:
            raise ValueError(f"Unsupported cgtType: {cgtType}")
        HOCGTs_nonsym_list.append(HOCGT_nonsym)

    # Print out the max eigenvalue and associated eigenvector for the first HOCGT in the list
    # for ii in range(len(STTs) - 1):
    #     HOCGT = HOCGTs_nonsym_list[ii][0]
    #     evals, evecs = np.linalg.eig(HOCGT)
    #     max_eval_idx = np.argmax(np.abs(evals))
    #     print(f"Time domain {timeDomain} index {ii}: Max eigenvalue = {evals[max_eval_idx]}, associated eigenvector = {evecs[:, max_eval_idx]}")

    for ii, HOCGTs_list in enumerate(HOCGTs_nonsym_list):
        for jj, HOCGT in enumerate(HOCGTs_list):
            # Save each HOCGT as a .mat file
            saveTag = f'{cgtType}_{timeDomain}_{ii}_O{jj + 1}' if savePrefix is None else f'{savePrefix}_{cgtType}_{timeDomain}_{ii}_O{jj + 1}'
            savemat(os.path.join(dataPath, f'{saveTag}_{runTag}.mat'), {f'HOCGT{jj}': HOCGT})



def load_eigenpair_data(prefix, dataPath, ts, order):
    # Load in hoeigenpair datat from MATLAB
    eval_data = loadmat(f"{dataPath}/{prefix}_HOho_evals.mat")
    evec_data = loadmat(f"{dataPath}/{prefix}_HOho_Rs.mat")

    all_evals = eval_data['ho_evals']
    all_evecs = evec_data['ho_Rs']

    # Want to make evals and evecs lists of arrays
    evals, evecs = [], []
    for j in range(all_evals.shape[0]):
        this_evals = []
        this_evecs = []
        for i in range(all_evals.shape[1]):
            if j == 0:
                this_evals.append(np.diag(all_evals[j, i]).flatten())
                this_evecs.append(all_evecs[j, i])
            else:
                this_evals.append(all_evals[j, i].flatten())
                that_evec = []
                for k in range(all_evecs[j, i].shape[1]):
                    that_evec.append(np.squeeze(all_evecs[j, i][0, k]))
                this_evecs.append(np.array(that_evec).T)

        evals.append(this_evals)
        evecs.append(this_evecs)

    # Evals is a [order] size list of lists with [number of time domains] size where each element is a [arbitrary size] np.array of evals
    # Evecs is a [order] size list of lists with [number of time domains] size where each element is a [stateSize, arbitrary size] np.array of evecs where each column is an evec

    # Also load in three HOCGTs
    HOCGTS = {}
    for tt in range(len(ts)-1):
        for oo in range(order):
            HOCGT_data = loadmat(f"{dataPath}/SYM_{prefix}_O{oo}_R{tt}.mat")
            HOCGT = HOCGT_data['tensor']['tensor'][0,0]
            HOCGTS[f'O{oo}t{tt}'] = np.array(HOCGT)

    return evals, evecs, HOCGTS



def getAllRs(order, stateSize, evals, evecs, Sevals, Sevecs):
    # Construct R matrices for DSTTs, hoDSTTs, and sDSTTs
    # hoDSTTs use third-order CGT eigenpairs for second-order DSTT, DSTTs use second-order CGT eigenpairs for second-order DSTT
    # Rs is a times x order-1 x state size matrix
    hoRs, sRs = np.zeros((len(evals[0]), order-1, stateSize)), np.zeros((len(evals[0]), order-1, stateSize))
    Rs = np.zeros((len(evals[0]), order-1, stateSize))

    if order == 3:
        for tt in range(len(evals[0])):
            # Get max second-order CGT eval index
            max_second_order_idx = np.argmax(evals[0][tt])
            max_second_order_idx_s = np.argmax(Sevals[0][tt])
            # Get max third-order CGT eval index
            max_third_order_idx = np.argmax(evals[1][tt])
            max_third_order_idx_s = np.argmax(Sevals[1][tt])
            # Get max fourth-order CGT eval index
            max_fourth_order_idx = np.argmax(evals[2][tt])
            max_fourth_order_idx_s = np.argmax(Sevals[2][tt])
            # Get evec for max eval for second-order CGT for both DSTTs
            Rs[tt, 0, :] = evecs[0][tt][:, max_second_order_idx]
            Rs[tt, 1, :] = evecs[0][tt][:, max_second_order_idx]
            # Get evec for max eval for third-order CGT for hoDSTTs and sDSTTs
            hoRs[tt, 0, :] = evecs[1][tt][:, max_third_order_idx]
            hoRs[tt, 1, :] = evecs[2][tt][:, max_fourth_order_idx]
            sRs[tt, 0, :] = evecs[1][tt][:, max_third_order_idx_s]
            sRs[tt, 1, :] = evecs[2][tt][:, max_fourth_order_idx_s]
    else:
        raise NotImplementedError("Only order=3 is implemented in this function.")

    return Rs, hoRs, sRs


def load_nonsymmetric_tensors(dataPath, regimes, filePrefix, tensorPrefix, saveTag, maxOrder):
    """
    Python version of MATLAB load_nonsymmetric_tensors
    """

    tensors_all = []

    for jj in range(len(regimes)):  # Regime loop
        tensors = []

        for ii in range(1, maxOrder + 1):  # Order loop (MATLAB: 1:maxOrder)

            file_name = (
                f"{filePrefix}_{regimes[jj]}_O{ii}_{saveTag}.mat"
            )
            file_path = os.path.join(dataPath, file_name)

            tensorData = loadmat(file_path)

            # MATLAB: tensorData.(tensorPrefix + string(ii-1))
            key = f"{tensorPrefix}{ii - 1}"

            tensors.append(tensorData[key])

        tensors_all.append(tensors)

    return tensors_all

def symmetrize_tensors(tensors_all, maxOrder):
    """
    Python version of MATLAB symmetrize_tensors
    """

    sym_tensors_all = []

    for jj in range(len(tensors_all)):  # loop over regimes
        sym_tensors = []

        for ii in range(maxOrder):  # 0 to maxOrder-1

            # MATLAB: symmetrize(tensor(tensors_all{jj}{ii}))
            t = tensors_all[jj][ii]
            sym_t = symmetrize(t)

            sym_tensors.append(sym_t)

        sym_tensors_all.append(sym_tensors)

    return sym_tensors_all

def save_sym_tensors(sym_tensors, dataPath, regimes, filePrefix, tensorPrefix, maxOrder):
    """
    Python version of MATLAB save_sym_tensors
    """

    for jj in range(len(regimes)):

        tensors = sym_tensors[jj]

        for ii in range(1, maxOrder + 1):

            # MATLAB: struct('tensor', tensors{1}{ii})
            data = {
                "tensor": tensors[ii - 1]
            }

            file_name = (
                f"{filePrefix}_O{ii-1}_R{regimes[jj]}.mat"
            )
            file_path = os.path.join(dataPath, file_name)

            savemat(file_path, data)


def save_eigenvectors(evecs, dataPath, flag, runTag):

    os.makedirs(dataPath, exist_ok=True)

    if flag == 's':
        data = {'s_Rs': evecs}
        filename = f"{runTag}_s_Rs.mat"

    elif flag == 'q':
        data = {'q_Rs': evecs}
        filename = f"{runTag}_q_Rs.mat"

    else:
        data = {'ho_Rs': evecs}
        filename = f"{runTag}_ho_Rs.mat"

    savemat(os.path.join(dataPath, filename), data)

def save_eigenvalues(evals, dataPath, flag, runTag):

    os.makedirs(dataPath, exist_ok=True)

    if flag == 's':
        data = {'s_evals': evals}
        filename = f"{runTag}_s_evals.mat"

    elif flag == 'q':
        data = {'q_evals': evals}
        filename = f"{runTag}_q_evals.mat"

    else:
        data = {'ho_evals': evals}
        filename = f"{runTag}_ho_evals.mat"

    savemat(os.path.join(dataPath, filename), data)

def save_HOeigenvectors(ho_evecs, dataPath, flag, runTag):

    os.makedirs(dataPath, exist_ok=True)

    if flag == 's':
        data = {'s_Rs': ho_evecs}
        filename = f"{runTag}_HOs_Rs.mat"

    elif flag == 'q':
        data = {'q_Rs': ho_evecs}
        filename = f"{runTag}_HOq_Rs.mat"

    else:
        data = {'ho_Rs': ho_evecs}
        filename = f"{runTag}_HOho_Rs.mat"

    savemat(os.path.join(dataPath, filename), data)

def save_HOeigenvalues(ho_evals, dataPath, flag, runTag):

    os.makedirs(dataPath, exist_ok=True)

    if flag == 's':
        data = {'s_evals': ho_evals}
        filename = f"{runTag}_HOs_evals.mat"

    elif flag == 'q':
        data = {'q_evals': ho_evals}
        filename = f"{runTag}_HOq_evals.mat"

    else:
        data = {'ho_evals': ho_evals}
        filename = f"{runTag}_HOho_evals.mat"

    savemat(os.path.join(dataPath, filename), data)