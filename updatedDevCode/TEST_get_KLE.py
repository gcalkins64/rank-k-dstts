# created by Grace Calkins
# 7/8/2026 10:45 am
# Purpose: generate and test a KLE of the Uranus GRAM atmosphere.

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from dev_dstt_gmm.plotters import *
from dev_dstt_gmm.util.data_manager import createPaths
from dev_dstt_gmm.util.kle import getEigenpairsKLE, getKLEfun


def KLE_tornado_plot(hs, rhoGRAMList, rhoKLEList, rhoGRAMDens, rhoKLEDens, figPath, h_ind=80,):
    # tornado plot
    fig, ax = plt.subplots()

    lab = 'KLE'

    rhoGRAM = np.asarray(rhoGRAMList)
    rhoKLE = np.asarray(rhoKLEList)

    KLEMEAN = np.mean(rhoKLE, axis=0)
    KLESTD = np.std(rhoKLE, axis=0)

    GRAMMEAN = np.mean(rhoGRAM, axis=0)
    GRAMSTD = np.std(rhoGRAM, axis=0)

    for i in range(3):
        ax.plot(rhoGRAMList[i], hs / 1e3, 'C2', alpha=0.5)
        ax.plot(rhoKLEList[i], hs / 1e3, 'C1', alpha=0.5)

    ax.plot(GRAMMEAN - 3 * GRAMSTD, hs / 1e3, 'C2--', linewidth=2)
    ax.plot(GRAMMEAN + 3 * GRAMSTD, hs / 1e3, 'C2--', linewidth=2, label=r'GRAM $3\sigma$')
    ax.plot(KLEMEAN - 3 * KLESTD, hs / 1e3, 'C1--', linewidth=2)
    ax.plot(KLEMEAN + 3 * KLESTD, hs / 1e3, 'C1--', linewidth=2, label=rf"{lab} $3\sigma$")
    ax.plot(GRAMMEAN, hs / 1e3, 'C2', linewidth=3, label='GRAM Mean')
    ax.plot(KLEMEAN, hs / 1e3, 'C1', linewidth=3, label=f"{lab} Mean")
    ax.legend()
    ax.set_ylim(0,2500)
    ax.set_xlim(-4,4)
    ax.set_xlabel('Normalized Density Perturbation')
    ax.set_ylabel('Altitude [km]')
    ax.tick_params()
    plt.tight_layout()
    plt.savefig(f"{figPath}/KLE_tornado_plot.png")

    # histograms
    dd = np.vstack((rhoGRAM[:,h_ind], rhoKLE[:,h_ind])).T
    df = pd.DataFrame(dd, columns=['GRAM', lab])

    fig, ax = plt.subplots()
    hh = sns.histplot(data=df, bins=np.linspace(-0.25, 0.25, 50))

    ax.tick_params()
    ax.set_xlabel('normalized density perturbation')
    ax.set_ylabel('count')
    plt.title("Altitude = " + str(round(hs[h_ind] / 1e3, 2)) + " km")
    plt.tight_layout()
    plt.savefig(f"{figPath}/KLE_hist.png")


    # Overall density
    fig, ax = plt.subplots()
    rhoGRAM = np.asarray(rhoGRAMDens)
    rhoKLE = np.asarray(rhoKLEDens)

    KLEMEAN = np.mean(rhoKLE, axis=0)

    GRAMMEAN = np.mean(rhoGRAM, axis=0)

    ax.set_xscale('log')
    for i in range(3):
        ax.plot(rhoGRAMDens[i], hs / 1e3, 'C2', alpha=0.5)
        ax.plot(rhoKLEDens[i], hs / 1e3, 'C1', alpha=0.5)

    ax.plot(GRAMMEAN, hs / 1e3, 'C2', linewidth=3, label='GRAM Mean')
    ax.plot(KLEMEAN, hs / 1e3, 'C1', linewidth=3, label=f"{lab} Mean")
    ax.legend()
    ax.set_xlabel(r'Density [kg/m$^3$]')
    ax.set_ylabel('Altitude [km]')
    ax.tick_params()
    plt.tight_layout()
    plt.savefig(f"{figPath}/KLE_density.png")


    return hh

def main():
    seabornSettings()
    basePath = '/Users/gracecalkins/Local_Documents/local_code/sttgmm'
    # runTag = 'TEST_get_KLE'
    runTag = 'TEST_get_KLE_Uranus'
    figPath, dataPath = createPaths(basePath, runTag)

    # Load in GRAM
    # GRAMPath = '/Users/gracecalkins/Local_Documents/local_code/GRAM_Suite_21/MyData/DP_1_0_earth'
    # GRAMPath = '/Users/gracecalkins/Local_Documents/local_code/GRAM_Suite_2.0/MyData/DP_1_uranus'  # Old UranusGRAM
    GRAMPath = '/Users/gracecalkins/Local_Documents/local_code/GRAM_Suite_21/MyData/DP_1_0_uranus'  # New UranusGRAM
    NSamples = 1000
    rhoMean = loadmat(f"{GRAMPath}/rho_nom.mat")['rho'].squeeze()  # kg/m^3
    rhos_GRAM = []
    for samp in range(NSamples):
        data = loadmat(f"{GRAMPath}/rho_{samp}.mat")
        h_GRAM = data['alt'].squeeze() * 1000  # m
        rho_GRAM = data['rho'].squeeze()  # kg/m^3
        rhos_GRAM.append(rho_GRAM)
    h = np.array(h_GRAM)
    rhos_GRAM = np.array(rhos_GRAM).T

    # Get Eigenbasis of GRAM Atmosphere
    evals, evecs, rhoSampMean, d, h = getEigenpairsKLE(h, rhos_GRAM, 0.99, scaleVec=None)
    print(f"Number of eigenvalues kept for 99% accuracy: {d}")
    # Save off evals, evecs, rhoSampMean, d, h for later use in STT analysis
    np.savez(f"{dataPath}/KLE_eigenpairs.npz", evals=evals, evecs=evecs, rhoSampMean=rhoSampMean, d=d, h=h)

    # Compute alpha as d increases
    alphas = []
    for d_test in range(len(evals)):
        alphaTrue = np.sum(evals[0:d_test]) / np.sum(evals)
        alphas.append(alphaTrue)

    plt.figure()
    plt.plot(range(len(evals)), alphas)
    plt.xlabel('d')
    plt.ylabel('alpha')
    plt.savefig(f"{figPath}/KLE_alpha_vs_d.png")
    plt.close()

    # Get perturbation values
    # compute and plot density perturbation values
    rhoGRAMDp, rhoGRAMDens = [], []
    rhoKLEDp, rhoKLEDens = [], []
    for i in range(100):
        # get density perturbations from KLE model
        rhoFun, Ys = getKLEfun(evals, evecs, rhoSampMean, d, h,
                                  rhomean=rhoMean, perts=False,
                                  scaleVec=None)
        rhoKLEDp.append(rhoFun(h) / rhoMean - 1)
        rhoKLEDens.append(rhoFun(h))

        # get density perturbations from GRAM
        rhoGRAMDp.append(rhos_GRAM[:, i]/rhoMean - 1)
        rhoGRAMDens.append(rhos_GRAM[:, i])

    # Plot KLE and GRAM to compare
    KLE_tornado_plot(h, rhoGRAMDp, rhoKLEDp, rhoGRAMDens, rhoKLEDens, figPath)
    plt.show()




if __name__ == "__main__":
    main()