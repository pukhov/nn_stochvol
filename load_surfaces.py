import gzip
import os
from logging import warning
from paths import PATH_TO_OLD_DATA

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm_notebook as tqdm

from nn_stochvol import utils
from nn_stochvol.paths import *


def load_vlt_surfaces(data_type):
    assert data_type in ["new", "old"]
    if data_type == "new":
        return _load_new_vlt_surfaces()
    else:
        return _load_old_vlt_surfaces()

def _load_new_vlt_surfaces():
    """
    Returns:
        new_params_nk, new_vol_nms, new_cols_k, new_maturities_ms, new_strikes_ms
    """
    all_paths_to_surfs = sorted(list((PATH_TO_NEW_DATA/"volsurfaces").glob("*")))
    print(f"Loading {len(all_paths_to_surfs)} surfaces...")

    def get_idx(path):
        return int(path.as_posix().rsplit("/")[-1].split("_")[0])

    def get_values(path):
        return pd.read_csv(path).values
    
    list_of_idx_n = Parallel(n_jobs=8)(delayed(get_idx)(path) for path in all_paths_to_surfs)
    list_of_vols_nms = Parallel(n_jobs=8)(delayed(get_values)(path) for path in tqdm(all_paths_to_surfs))
    
    list_of_idx_n = np.array(list_of_idx_n)
    assert list_of_idx_n.min() == 1 # r has indexing from 1
    list_of_idx_n -= 1
    n_idx_n       = np.argsort(list_of_idx_n)
    if not (list_of_idx_n == np.sort(list_of_idx_n)).all():
        warning("passed non consecutive idces -- not all vlt matrices created.")

    vols_nms = np.array(list_of_vols_nms)

    idx_n = list_of_idx_n[n_idx_n]
    vols_nms = vols_nms[n_idx_n]

    forward_var_cols_4 = ['intercept', 'linear', 'square_root', 'inv_square_root']
    forward_var_params_n4 = pd.read_csv(PATH_TO_NEW_DATA/"forward_var_params.csv").values

    heston_cols_3 = ["eta", "rho", "H"]
    heston_params_n3      = pd.read_csv(PATH_TO_NEW_DATA/"heston_params.csv")[heston_cols_3].values

    params_cols_7 = forward_var_cols_4 + heston_cols_3
    params_n7 = np.concatenate([forward_var_params_n4, heston_params_n3], axis=1)[idx_n]

    mat_ms, strikes_ms = utils.get_mat_tk_strike_tk("new")


    return {"params_nk" : params_n7, "vols_nms"  : vols_nms, "params_cols_k" :  params_cols_7,
            "mat_ms" : mat_ms, "strikes_ms" : strikes_ms}


def _load_old_vlt_surfaces(fname="TrainrBergomiTermStructure.txt.gz"):
    """
    Returns:
        dict with keys old_params_nk, old_vol_nms, old_cols_k, old_maturities_ms, old_strikes_ms
    """
    file = gzip.GzipFile(PATH_TO_OLD_DATA/fname, "r")
    dat_nf = np.load(file)
    
    mat_ms, strikes_ms = utils.get_mat_tk_strike_tk("old")
    
    params_nk = dat_nf[:,:-88]
    assert params_nk.shape[1] in [11], "Now using only this, check and fix this assert"
    
    vols_nms   = dat_nf[:,-88:]
    vols_nms   = vols_nms.reshape(vols_nms.shape[0], mat_ms.shape[0], mat_ms.shape[1])
    assert vols_nms.shape[1:] == (8, 11)


    cols_k = [f"ksi_{i}" for i in range(8)] + ["eta", "rho", "H"]

    return {"params_nk" : params_nk, "vols_nms"  : vols_nms, "params_cols_k" :  cols_k,
            "mat_ms" : mat_ms, "strikes_ms" : strikes_ms}


def get_vlt_surf_ms_from_real_data(df_or_date_yyyymmdd=20130814, *, data_type):
    """
    @param df_or_date_yyyymmdd
        either a dataframe with Bid, Ask, Strike, Fwd and Texp fields
        or yyyymmdd (then make sure the file is in correct folder
        and has name spxVols{df_or_date_yyyymmdd}.csv)
    return
        vlt_surf_ms, df, mat_ms, strikes_ms
    """
    assert data_type in ["new", "old"]

    if isinstance(df_or_date_yyyymmdd, int):
        df = pd.read_csv(PATH_TO_REAL_DATA/f"spxVols{df_or_date_yyyymmdd}.csv")
    else:
        df = df_or_date_yyyymmdd
        assert isinstance(df, pd.DataFrame)
        
    mid_t = (df.Bid + df.Ask).values / 2
    valid_t = np.isfinite(mid_t)

    mid_t = mid_t[valid_t]
    strike_t = (df.Strike / df.Fwd).values[valid_t]
    mat_t = df.Texp.values[valid_t]

    def transform_strike_mat(strike_t, mat_t):
        strike_t = np.log(strike_t) / np.sqrt(mat_t)
        mat_t = np.log(mat_t)
        return strike_t * 4, mat_t # they have same std then
    
    strike_t, mat_t = transform_strike_mat(strike_t, mat_t)
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(np.stack([strike_t, mat_t]).T, mid_t,)

    mat_ms, strikes_ms = utils.get_mat_tk_strike_tk(data_type)

    strike_T, mat_T = transform_strike_mat(strikes_ms.ravel(), mat_ms.ravel())
    vlt_surf_ms = knn.predict(np.stack([strike_T, mat_T]).T).reshape(strikes_ms.shape)
    
    return vlt_surf_ms, df, mat_ms, strikes_ms
