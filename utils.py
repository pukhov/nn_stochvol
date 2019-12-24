import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_stochvol.paths import *

def to_np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.clone().detach().numpy()


def get_mat_tk_strike_tk(type):
    """
    @returns
        mat_tk, strike_tk
    """
    assert type in ["new", "old"]

    if type == "new":
        mat_str_grid_tk = pd.read_csv(PATH_TO_NEW_DATA/"maturity_strike_grid.csv", index_col="maturities")
        n_t, n_k = mat_str_grid_tk.values.shape
        
        mat_t = mat_str_grid_tk.index.values
        mat_tk = np.tile(mat_t[:, None], (1, n_k))
        strike_tk = mat_str_grid_tk.values
        
        return mat_tk, strike_tk
    else:
        maturities = np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0 ])
        strikes = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 ])

        mat_ms, strikes_ms = np.meshgrid(maturities, strikes)
        return mat_ms.T, strikes_ms.T


def scale_data(x_train, x_test):
    """
    calls fit_transform on train and only transform on test, thus being causal.
    
    @returns
        x_train, x_test, scaler_inst
    """
    scaler_inst = StandardScaler()

    original_shape = None # case when x_train is 2D
    if x_train.ndim > 2:
        original_shape = tuple([-1] + list(x_train.shape[1:]))
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test  = x_test.reshape(x_test.shape[0], -1)
        
    
    x_train = scaler_inst.fit_transform(x_train)
    x_test = scaler_inst.transform(x_test)
 
    if original_shape is not None:
        x_train = x_train.reshape(original_shape)
        x_test  = x_test.reshape(original_shape)
    
    return x_train, x_test, scaler_inst


def get_train_test_split_nk_nf(raw_data):
    """
    splits data and scales both input and output

    @returns
        dict with keys (params_train_nk, vol_train_nf,
                        params_test_Nk, vol_test_Nf,
                        params_scaler=params_scaler, vol_scaler)
    """
    params_train_nk, params_test_Nk, vol_train_nms, vol_test_Nms = \
            train_test_split(raw_data["params_nk"], raw_data["vols_nms"], test_size=0.15, random_state=42)

    params_train_nk, params_test_Nk, params_scaler = scale_data(params_train_nk, params_test_Nk)
    vol_train_nms,   vol_test_Nms,   vol_scaler    = scale_data(vol_train_nms, vol_test_Nms)

    vol_train_nf = vol_train_nms.reshape(vol_train_nms.shape[0], -1)
    vol_test_Nf  = vol_test_Nms.reshape(vol_test_Nms.shape[0], -1)

    return dict(
        params_train_nk=params_train_nk, vol_train_nf=vol_train_nf,
        params_test_Nk=params_test_Nk, vol_test_Nf=vol_test_Nf,
        params_scaler=params_scaler, vol_scaler=vol_scaler)
