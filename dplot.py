import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

from .utils import to_np


def plot_params(optimal_params, clip_args=(0.1, 0.5), label=None, axis=None):
    """
    @param label
        will be used as an index for dataframe. E.g. pass yyyymmdd
    """
    if optimal_params.size == 11:
        data_type = "old"
    elif optimal_params.size == 7:
        data_type = "new"
    else:
        raise NotImplementedError


    if data_type == "old":
        if axis is None:
            _, axis = plt.subplots(1, 1)
        axis.set_title("Forward curve")
        axis.plot(optimal_params[:8], label="optimal params")
        axis.legend()
    else:
        axis = plot_forward_var_curve(optimal_params[:4],
                clip_args=clip_args, label="optimal params", axis=axis)
        axis.set_title("Forward curve")
        axis.legend()


    df = pd.DataFrame(data=optimal_params[-3:].reshape(1, -1), columns=['eta', 'rho', 'H'], index=[label])

    return df, axis


###
## 3D plotting utils
###    

def get_fig_axis_3d():
    fig = plt.figure()
    axis = fig.gca(projection="3d")
    return fig, axis


def plot_vol_surface(
        mat_tk, strike_tk, data_tk, 
        axis, cmap=cm.coolwarm,
        log_mat_tk=False, log_and_sqrt_strike_tk=False):
    """
    @param log_and_sqrt_strike_tk:
        if True:     strike_tk = np.log(strike_tk).copy() / np.sqrt(mat_tk)
    """

    if log_and_sqrt_strike_tk:
        strike_tk = np.log(strike_tk).copy() / np.sqrt(mat_tk)
    if log_mat_tk:
        mat_tk = np.log(mat_tk).copy()

        
    surf = axis.plot_surface(
        mat_tk, strike_tk, data_tk,
        rstride=1, cstride=1, cmap=cmap,
        linewidth=0, antialiased=False, alpha=0.5)
    
    axis.set_xlabel(("log" if log_mat_tk else "") + "time to expiry")
    axis.set_ylabel("log strike / sqrt(maturity)" if log_and_sqrt_strike_tk else "strike")

def plot_vol_surface_from_real_data(
        df, axis, cmap=cm.winter,
        log_mat_tk=False, log_and_sqrt_strike_tk=False):
    
    mat_t   = df.Texp.values
    strike_t = (df.Strike / df.Fwd).values
    mid_vlt_t = (df.Bid + df.Ask).values / 2


    if log_and_sqrt_strike_tk:
        strike_t = np.log(strike_t).copy() / np.sqrt(mat_t)
        strike_t = strike_t.clip(-1, 0.4)
    if log_mat_tk:
        mat_t = np.log(mat_t).copy()


    axis.plot_trisurf(mat_t, strike_t, mid_vlt_t, cmap=cmap, alpha=0.5)

    axis.set_xlabel(("log" if log_mat_tk else "") + "time to expiry")
    axis.set_ylabel("log strike / sqrt(maturity)" if log_and_sqrt_strike_tk else "strike")
