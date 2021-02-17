#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.tri as tri
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pickle

import time
import seaborn as sns
from astropy.visualization import hist
from copy import deepcopy

# from awkde import GaussianKDE as AdaptiveGaussianKDE

import logging
logging.basicConfig(
    # filename='myfirstlog.log',
    # level=logging.INFO,
    level=logging.DEBUG,
    format='\n > %(asctime)s | %(name)s | %(levelname)s \n > %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Matplotlib setup
plt.style.use("~/cerfacs.mplstyle")
plt.rcParams['figure.dpi'] = 125
plt.rcParams['figure.figsize'] = 4, 3
plt.rcParams['axes.grid'] = False

PLOT = 0

# Functions
def get_cut_mesh_data(mesh_file):
    """get cut mesh data"""
    logger.info("Read cut mesh:\n\t > %s" % (mesh_file))
    with h5py.File(mesh_file, 'r') as f:
        npts = f['Coordinates/x'].shape
        x = f['Coordinates/x'][:int(npts[0])]
        y = f['Coordinates/y'][:int(npts[0])]
        z = f['Coordinates/z'][:int(npts[0])]

        data = {
            "x": x,
            "y": y,
            "z": z,
        }
    return pd.DataFrame.from_dict(data)


def get_sol_mesh_data(mesh_file):
    """ get sol mesh data"""
    logger.info("Read sol mesh:\n\t > %s" % (mesh_file))
    with h5py.File(mesh_file, 'r') as f:
        npts = f['Coordinates/x'].shape
        x = f['Coordinates/x'][:int(npts[0])]
        y = f['Coordinates/y'][:int(npts[0])]
        z = f['Coordinates/z'][:int(npts[0])]
        volume = f['VertexData/volume'][:int(npts[0])]

        data = {
            "x": x,
            "y": y,
            "z": z,
            "voln": volume
        }
    return pd.DataFrame.from_dict(data)


def get_sol_data(sol_file):
    """ get solution data"""
    logger.info("Read sol/cut:\n\t > %s" % (sol_file))
    with h5py.File(sol_file, 'r') as f:
        npts = f['GaseousPhase/rho'].shape

        rho = f['GaseousPhase/rho'][:int(npts[0])]
        u = f['GaseousPhase/rhou'][:int(npts[0])]
        v = f['GaseousPhase/rhov'][:int(npts[0])]
        w = f['GaseousPhase/rhow'][:int(npts[0])]
        u /= rho
        v /= rho
        w /= rho

        # C3H8 CO CO2 H2O N2 O2
        C3H8 = f['RhoSpecies/C3H8'][:int(npts[0])]
        CO = f['RhoSpecies/CO'][:int(npts[0])]
        CO2 = f['RhoSpecies/CO2'][:int(npts[0])]
        H2O = f['RhoSpecies/H2O'][:int(npts[0])]
        N2 = f['RhoSpecies/N2'][:int(npts[0])]
        O2 = f['RhoSpecies/O2'][:int(npts[0])]
        C3H8 /= rho
        CO /= rho
        CO2 /= rho
        H2O /= rho
        O2 /= rho
        N2 /= rho
        sum_yk = C3H8 + CO + CO2 + H2O + N2 + O2

        dtsum = f['Parameters/dtsum'][0]

        efcy = f['Additionals/efcy'][:int(npts[0])]
        hr = f['Additionals/hr'][:int(npts[0])]
        pressure = f['Additionals/pressure'][:int(npts[0])]
        temperature = f['Additionals/temperature'][:int(npts[0])]
        theta_F = f['Additionals/theta_F'][:int(npts[0])]
        thick = f['Additionals/thick'][:int(npts[0])]
        uprim = f['Additionals/uprim'][:int(npts[0])]
        wall_yplus = f['Additionals/wall_yplus'][:int(npts[0])]

        var_dict = locals()

        data = {}

        for var_name, var_val in var_dict.items():
            if var_name == "sol_file" or var_name == "f" or var_name == "npts" or var_name == "dtsum":
                pass
            else:
                # print(var_name)
                data[var_name] = var_val

        df_out = pd.DataFrame.from_dict(data)

        df_out.time = dtsum
    return df_out


def bin_static(x, y, n_bins=10):
    """ get static binning"""
    logger.info(" > Bin statistics")
    bin_means, bin_edges, binnumber = stats.binned_statistic(
        x, y, statistic='mean', bins=n_bins)
    bin_std, bin_edges, binnumber = stats.binned_statistic(
        x, y, statistic='std', bins=n_bins)

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return bin_centers, bin_means, bin_std


def get_bin_sizes_x(x, algo='scott'):
    """ Smartly get bin size to have a loer bias due to binning"""
    from astropy.stats import freedman_bin_width, scott_bin_width, knuth_bin_width, bayesian_blocks
    logger.info(" > Get smart bin sizes in 1D")

    if algo == 'scott':
        logger.info("use scott rule of thumb")
        width_x, bins_x = scott_bin_width(x, return_bins=True)
    elif algo == 'knuth':
        logger.info("use knuth rule of thumb")
        width_x, bins_x = knuth_bin_width(x, return_bins=True)
    elif algo == 'freedman':
        logger.info("use freedman rule of thumb")
        width_x, bins_x = freedman_bin_width(x, return_bins=True)
    elif algo == 'blocks':
        logger.info("use bayesian blocks rule of thumb")
        width_x, bins_x = bayesian_blocks(x, return_bins=True)
    else:
        raise NotImplementedError("use scott, knuth, freedman or blocks")

    return bins_x, width_x


def get_bin_sizes_xy(x, y, algo='scott'):
    """ Smartly get bin size to have a loer bias due to binning"""
    from astropy.stats import freedman_bin_width, scott_bin_width, knuth_bin_width, bayesian_blocks
    logger.info(" > Get smart bin sizes in 2D")

    if algo == 'scott':
        logger.info("use scott rule of thumb")
        width_x, bins_x = scott_bin_width(x, return_bins=True)
        width_y, bins_y = scott_bin_width(y, return_bins=True)
    elif algo == 'knuth':
        logger.info("use knuth rule of thumb")
        width_x, bins_x = knuth_bin_width(x, return_bins=True)
        width_y, bins_y = knuth_bin_width(y, return_bins=True)
    elif algo == 'freedman':
        logger.info("use freedman rule of thumb")
        width_x, bins_x = freedman_bin_width(x, return_bins=True)
        width_y, bins_y = freedman_bin_width(y, return_bins=True)
    else:
        raise NotImplementedError("use scott or knuth")
    n_bins_x, n_bins_y = len(bins_x), len(bins_y)

    return bins_x, bins_y, width_x, width_y


def get_dataframe_cut(mesh_file, sol_file):
    """
    Read cut data as pandas dataframe
    """
    logger.info("Get DataFrame cut")
    df_mesh = get_cut_mesh_data(mesh_file)
    df_sol = get_sol_data(sol_file)

    for col in df_mesh.columns.values:
        df_sol[col] = df_mesh[col]

    return df_sol


def get_dataframe_solut(mesh_file, sol_file):
    """
    Read sol data as pandas dataframe
    """
    logger.info("Get DataFrame solut")
    df_mesh = get_sol_mesh_data(mesh_file)
    df_sol = get_sol_data(sol_file)

    logger.debug("Add mesh columns")
    for col in df_mesh.columns.values:
        logger.debug(col)
        df_sol[col] = df_mesh[col]

    return df_sol


def get_progress_variable(df_avbp, field='H2O'):
    """ return progress variable """
    logger.info("Get progress variable (bases on %s)" % field)
    res = (df_avbp[field] - df_avbp[field].min())
    res /= (df_avbp[field].max() - df_avbp[field].min())
    return res


def get_flame_tip_and_idx(df):
    logger.info("Get flame tip position and index")
    x_flame_tip = df[df.temperature > 1500.]
    x_flame_tip = x_flame_tip.x.max()
    idx_x = np.argmin(np.abs(df.x - x_flame_tip))
    return x_flame_tip, idx_x


def integrate_mass_fuel(df_sol):
    """ cumulative integral in x direction of mass of fuel"""
    logger.info("Integrate fuel mass")
    df_sol = df_sol.sort_values(by="x")

    df_sol["rhoY_C3H8"] = df_sol.rho * df_sol.C3H8
    df_sol["m_C3H8"] = df_sol["rhoY_C3H8"] * df_sol["voln"]
    df_sol["integral_mass_C3H8_over_x"] = np.cumsum(df_sol["m_C3H8"])
    return df_sol


if __name__ == "__main__":
    # Sol
    logger.info("Load mesh")
    dir_mesh = '/scratch/cfd/qdouasbin/LEFEX/MASRI/BLINDTEST/MESH_20M/MASRI_3D_20M_2ZONES_HIP_20_07'
    dir_sol = '/scratch/cfd/qdouasbin/LEFEX/MASRI/MESH_ADAPTATION_STUDY/Colin2/20M_correction_uprim/RUN_dyn_corrected_V2/SOLUT'
    sol_mesh_file = os.path.join(
        dir_mesh, "MASRI_3D_20M_2ZONES_HIP_20_07.mesh.h5")
    sol_file = os.path.join(dir_sol, "MASRI_3D_00000009.h5")
    df_sol = get_dataframe_solut(sol_mesh_file, sol_file)

    # # Clipping
    df_sol = df_sol[df_sol["x"] < 0.25].dropna()

    # Sort
    df_sol = df_sol.sort_values(by="x")

    # Progress variable
    df_sol["c"] = get_progress_variable(df_sol)

    # Flame tip
    x_flame_tip, idx_x_flame_tip = get_flame_tip_and_idx(df_sol)

    # Integrate fuel
    df_sol = integrate_mass_fuel(df_sol)
    df_sol.m_fuel_ch = df_sol["integral_mass_C3H8_over_x"].max()
    df_sol.m_fuel_ch_before_xtip = df_sol["integral_mass_C3H8_over_x"][idx_x_flame_tip]

    # PDF over the whole field
    # lst_var_pdf = ["efcy"]
    lst_var_pdf = df_sol.columns.values

    # regular spacing
    logger.info("PDF, over the whole domain")

    dict_data_pdf = {}

    for var in lst_var_pdf:
        logger.info("PDF(%s)" % var)
        data_x = df_sol[var].values
        bins_x, width_x = get_bin_sizes_x(data_x)
        hist_var, x_edges = np.histogram(data_x,
                                         bins=bins_x,
                                         density=True)

        dict_data_pdf[var] = {"hist": hist_var,
                              "edges": x_edges,
                              }

        if PLOT:
            plt.figure()
            _ = plt.plot(x_edges[:-1], hist_var, '-',
                         alpha=0.5, label="kde, scott")
            logger.info("plot PDF(%s), kde --> done" % var)

            plt.xlabel("%s" % var)
            plt.ylabel("PDF(%s)" % var)
            plt.xscale('symlog', linthreshx=1e-1)
            plt.yscale('symlog', linthreshy=1e-1)
            plt.ylim(bottom=0)
            plt.savefig("PDF_%s.png" %
                        var, bbox_inches='tight', pad_inches=0.02)
            plt.close()

    # Conditional PDF --> c- < c < c+
    logger.info("PDF, before x_tip")
    c_minus = 0.05
    c_plus = 0.95
    dict_data_pdf_flame = {}
    df_flame = df_sol[df_sol.c < c_plus]
    df_flame = df_flame[c_minus < df_flame.c]

    for var in lst_var_pdf:
        logger.info("PDF(%s)" % var)
        data_x = df_flame[var].values
        bins_x, width_x = get_bin_sizes_x(data_x)
        hist_var, x_edges = np.histogram(data_x,
                                         bins=bins_x,
                                         density=True)

        dict_data_pdf_flame[var] = {"hist": hist_var,
                                    "edges": x_edges,
                                    }

        if PLOT:
            plt.figure()
            _ = plt.plot(x_edges[:-1], hist_var, '-',
                         alpha=0.5, label="kde, scott")
            logger.info("plot PDF(%s), kde --> done" % var)

            plt.xlabel("%s" % var)
            plt.ylabel("PDF(%s)" % var)
            plt.xscale('symlog', linthreshx=1e-1)
            plt.yscale('symlog', linthreshy=1e-1)
            plt.ylim(bottom=0)
            plt.savefig("PDF_flame_%s.png" %
                        var, bbox_inches='tight', pad_inches=0.02)
            plt.close()

    # Conditional PDF
    logger.info("PDF, before x_tip")
    dict_data_pdf_before_tip = {}
    df_before_tip = df_sol[df_sol.x < x_flame_tip]

    for var in lst_var_pdf:
        logger.info("PDF(%s)" % var)
        data_x = df_before_tip[var].values
        bins_x, width_x = get_bin_sizes_x(data_x)
        hist_var, x_edges = np.histogram(data_x,
                                         bins=bins_x,
                                         density=True)

        dict_data_pdf_before_tip[var] = {"hist": hist_var,
                                         "edges": x_edges,
                                         }

        if PLOT:
            plt.figure()
            _ = plt.plot(x_edges[:-1], hist_var, '-',
                         alpha=0.5, label="kde, scott")
            logger.info("plot PDF(%s), kde --> done" % var)

            plt.xlabel("%s" % var)
            plt.ylabel("PDF(%s)" % var)
            plt.xscale('symlog', linthreshx=1e-1)
            plt.yscale('symlog', linthreshy=1e-1)
            plt.ylim(bottom=0)
            plt.savefig("PDF_before_x_tip_%s.png" %
                        var, bbox_inches='tight', pad_inches=0.02)
            plt.close()


    logger.info("Collect data")
    data_out = {
        "PDF_flame": dict_data_pdf_flame,
        "PDF_domain": dict_data_pdf,
        "PDF_before_tip": dict_data_pdf_before_tip,
        "x_tip": x_flame_tip,
        "m_fuel_ch": df_sol.m_fuel_ch,
        "m_fuel_ch_before_xtip": df_sol.m_fuel_ch_before_xtip,
    }

    logger.info("Dump pickle")
    # Export values of interest
    with open(os.path.split(sol_file)[-1].replace(".h5", ".p"), 'wb') as f1:
        pickle.dump(data_out, f1)

    logger.info("Done.")