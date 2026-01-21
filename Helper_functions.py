## Helper functions
### Bernardo AO
import numpy as np
from scipy import interpolate 
import scipy.signal as signal
from scipy import stats
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors as pltcolors

from joblib import Parallel, delayed

def create_pupil_data(exp, save_path, sessions_files, output_variables):
    """
    Creates a pd array or reads it in case it already exists.
    """
    pupil_data_path = os.path.join(save_path, "pupil_data",
                                   "pupil_data_" + exp + ".pkl")
    if os.path.isfile(pupil_data_path) and False:
        pupil_data = pd.read_pickle(pupil_data_path)
    else:
        pupil_data_dic = {"session": [file[5:24] for file in sessions_files],
                          "awake": [1 for file in sessions_files]}
        for var in output_variables:
            if var != "session" and var != "awake":
                pupil_data_dic[var] = [np.array([])] * len(sessions_files)
    
        pupil_data = pd.DataFrame(pupil_data_dic)
    
    return pupil_data_path, pupil_data

def handle_exceptions(ROIs_smooth, tv, session):
    if session == "2023-03-16-11-14-37":
        mask = tv < 0.15 * 60
        ROIs_smooth[:,:,mask] = np.nan
    if session[:10] == "2023-04-18":
        ROIs_smooth = np.flip(ROIs_smooth, axis=1)
    return ROIs_smooth

def time_smooth_ROI(ROIs, win, confidence = [], con_thr = 0.8):
    """
    Smooths a sequence of ROIs over time using either a median or 
    confidence weighted average filter. 
    
    Parameters:
    - ROIs : np.ndarray, shape (n_ROIs, 2, T)
    - win : int, Half-window size for the smoothing operation.
    - confidence : np.ndarray, shape (n_ROIs, T)
    - con_thr : float, Confidence threshold 
    
    Returns:
    - ROIs_smooth : np.ndarray, shape (n_ROIs, 2, T)
    """
    ROIs_smooth = np.copy(ROIs)
    
    if confidence:
        for t in range(win, ROIs.shape[2] - win):
            for r in range(ROIs.shape[0]):
                if confidence[r,t] < con_thr:
                    weigth = confidence[r,t-win:t+win]
                    ROIs_smooth[:,:,t] = np.average(ROIs[:,:,t-win:t+win],2,weigth)
    else:
        for t in range(win, ROIs.shape[2] - win):
            ROIs_smooth[:,:,t] = np.median(ROIs[:,:,t-win:t+win],2)
    return ROIs_smooth

def interpolate_outliers(tv, values, n_std=4):
    """
    Interpolates valuesthat are more than n_std from the mean.

    Parameters
    ----------
    tv : np.ndarray, shape (T)
    values : : np.ndarray, shape (T)
    n_std : float, Number of stds to use as the cutoff

    Returns
    -------
    cleaned_values = np.ndarray, shape (T)    
    """
    
    mean = np.mean(values)
    std = np.std(values)

    # Identify outliers
    mask = np.abs(values - mean) > n_std * std

    # Create interpolator using only non-outlier points
    interp_func = interpolate.interp1d(tv[~mask], values[~mask],
        kind='linear', bounds_error=False, fill_value='extrapolate')

    # Replace outliers with interpolated values
    cleaned_values = values.copy()
    cleaned_values[mask] = interp_func(tv[mask])

    return cleaned_values

def get_pupil_size(ROIs_smooth, confidence=[]):
    """
    Estimates pupil size over time by computing the distance between
    opposing ROI point pairs and using the median, or if provided the max 
    confidence.

    Parameters:
    - ROIs_smooth : np.ndarray, shape (n_pupil_ROIs, 2, T)
    - confidence : np.ndarray, shape (n_ROIs, T)

    Returns:
    - pupil_size : np.ndarray, shape (T,)
    """
    opp = ROIs_smooth.shape[0] // 2
    pupil_sizes = np.zeros((opp, ROIs_smooth.shape[2]))
    for pair in range(opp):
        diff = ROIs_smooth[pair,:,:] - ROIs_smooth[pair + opp,:,:]
        pupil_sizes[pair,:] = np.sqrt(diff[0,:]**2 + diff[1,:]**2)
        
    if confidence:
        opp_conf = np.zeros((opp, ROIs_smooth.shape[1]))
        for pair in range(opp):
            opp_conf[pair,:] = confidence[pair,:] + confidence[pair + opp,:]    
        max_opp_conf = np.argmax(opp_conf, 0)
        pupil_size = pupil_sizes[max_opp_conf, np.arange(pupil_sizes.shape[1])]
    else:
        pupil_size = np.nanmedian(pupil_sizes, 0)
    return pupil_size

def get_pupil_center(ROIs_smooth, pupil_size, centered=False):
    """
    Estimates pupil center over time. Uses the pupil size to determine 
    outlayers.

    Parameters:
    - ROIs_smooth : np.ndarray, shape (n_pupil_ROIs, 2, T)
    - pupil_size : np.ndarray, shape (T,)
    - centered: defines if the data should be centered around the mean

    Returns:
    - pupil_center : np.ndarray, shape (2,T)
    """
    mean = np.nanmean(ROIs_smooth, axis=0)
    len_t = ROIs_smooth.shape[2]
    pupil_center = np.zeros((2,len_t))
    for t in range(len_t):
        points = ROIs_smooth[:,:,t]
        d_mean = np.linalg.norm(points - mean[:, t], axis=1)

        mask = d_mean <= pupil_size[t]
        valid_points = points[mask]
        pupil_center[:,t] = np.nanmean(valid_points, axis=0)
    
    if centered:
        return pupil_center - np.nanmean(pupil_center, axis=1).reshape(-1,1)
    else:
        return pupil_center

def import_saccades(session, filename="saccades.txt"):
    """
    Parameters:
    - session : str
    - filename : str
    
    Returns:
    - data : dict
    """

    data = {"temporal": None, "nasal": None}
    current_section = None
    filename="saccades.txt"

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            # Detect section headers
            if line == '"temporal"' or line == '"nasal"':
                current_section = line.strip('"')
                continue

            # Only parse lines inside a section
            if current_section:
                # Split by comma and strip spaces
                parts = [p.strip() for p in line.split(",")]

                # First element is the session string
                row_session = parts[0]

                if row_session == session:
                    # The rest are integers (saccade times)
                    values = [int(x) for x in parts[1:]]
                    data[current_section] = values

    return data

def get_stims(Spke_Bundle):
    """
    Visual stimuli for each recording.

    Parameters:
    - Spke_Bundle : dict
    
    Returns:
    - vis_stim : list
    - colors : list

    """
    vis_stim_all = ["Sl36x22_d_3","Sd36x22_l_3", "mb", 
                "Nat_Mov", "Nat_Mov_sw", "Nat_Mov_sc",
                "csd", "chirp"]
    cmap = plt.get_cmap('jet', len(vis_stim_all))    
    
    vis_stim = []
    colors = []
    for i, s in enumerate(vis_stim_all):
        if s in Spke_Bundle["events"]:
            vis_stim.append(s)
            colors.append(cmap(i))
            
    return vis_stim, colors

def get_events(b, window_pre = 2, window_post = 1, n_std = 3, rp = 1, 
               min_a = 0.05, camara_fs = 200):
    """
    Estimates event times of high behavior, by calculating a moving mean in a 
    pre event window, and comparing it to a post event window mean.

    Parameters:
    - b: np.ndarray, shape (T) or (2,T)
    - window_pre: float, seconds of pre window
    - window_post: float, seconds of post window
    - n_std: float, number of std between pre and post mean to mark an event
    - rp: float, minimum time between events
    - min_a: float, minimum amplitud
    
    Returns:
    - event_indx : list, length (n_events)
    """
    pre_i = window_pre * camara_fs
    post_i = window_post * camara_fs
    
    event_indx = []
    event_types = []
    if b.ndim == 1:
        for ti in range(pre_i, len(b)-post_i):
            if not event_indx or (ti - event_indx[-1]) / camara_fs > rp:
                tw_pre = np.arange(ti-pre_i, ti)
                m_pre = np.mean(b[tw_pre])
                std_pre = np.std(b[tw_pre])
                
                tw_post = np.arange(ti, ti+post_i)
                m_post = np.mean(b[tw_post])
        
                if m_post > m_pre + n_std*std_pre and m_post - m_pre > min_a:
                    event_indx.append(ti)
    else:
        for ti in range(pre_i, b.shape[-1]-post_i):
            if not event_indx or (ti - event_indx[-1]) / camara_fs  > rp:
                tw_pre = np.arange(ti -pre_i, ti)
                m_pre = np.mean(b[:,tw_pre], axis=1)
                
                tw_post = np.arange(ti, ti + post_i)
                m_post = np.mean(b[:,tw_post], axis=1)
                
                vec = m_post - m_pre
                if np.linalg.norm(vec) > n_std:
                    event_indx.append(ti)
                    angle = np.arctan2(vec[1], vec[0])
                    event_types.append(angle)
                    
    return np.array(event_indx), np.array(event_types)
    
def import_spike_data(exp, path_2_spike_bundle, 
                      fs = 30000):
    """
    Imports spike bundle data from the _Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy files,
    and supposedly inhibitory neuron data from the '_local_storage_SIN.npy' files
    """
    # load spike data for each unit
    Spke_Bundle_name = os.path.join(path_2_spike_bundle, exp,
                                    exp + '_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy')
    Spke_Bundle = \
        np.load(Spke_Bundle_name, allow_pickle=True,encoding='latin1').item()
    spiketimes = [unit_times / fs
                  for unit_times in Spke_Bundle['spiketimes_aligned']]  
    
    # load Supposedly inhibitory neurons file
    path_SIN_data =  \
        os.path.join(path_2_spike_bundle, exp, 
                        exp + '_local_storage_SIN.npy')
    SIN_data =  \
        np.load(path_SIN_data,allow_pickle=True,encoding='latin1').item()

    return Spke_Bundle, spiketimes, SIN_data

def import_pupil_data(pupil_data_path, Spke_Bundle, exp, period, fs = 30000):
    """
    Imports pupil data, concatenating all videos for an experiment

    """
    exp_pd_path =  os.path.join(pupil_data_path, "pupil_data_" + exp + ".pkl")
    pupil_data = pd.read_pickle(exp_pd_path)
    
    # Size
    pupil_sizes_pd = pupil_data['pupil_size']
    pupil_size = np.concatenate(pupil_sizes_pd.to_numpy())
    n_points_ses = [len(s) for s in pupil_sizes_pd]
    n_frames = sum(n_points_ses)
    
    # Center
    pupil_center_pd = pupil_data['pupil_center']
    pupil_center = np.concatenate(pupil_center_pd.to_numpy(), axis = 1)
    
    # Saccades
    saccades_pd = pupil_data['saccade_indx'].tolist()
    saccades = {'temporal': np.array([]),
                'nasal': np.array([])}
    off_set = 0
    for s in range(len(saccades_pd)):
        temporal_s = np.array(saccades_pd[s]["temporal"]) + off_set
        saccades["temporal"] = np.append(saccades["temporal"], temporal_s)
        
        nasal_s = np.array(saccades_pd[s]["nasal"]) + off_set
        saccades["nasal"] = np.append(saccades["nasal"], nasal_s)
        
        off_set += n_points_ses[s]
    saccades["temporal"] = saccades["temporal"].astype(np.int64)
    saccades["nasal"] = saccades["nasal"].astype(np.int64)
    
    # Period
    sync_all_cam = Spke_Bundle["Synchronization_TTLs"]["Sync_cam"] / fs
    sync_cam = sync_all_cam[:n_frames]
    
    if period == "all":
        mask = np.ones(len(sync_cam), dtype=bool)
    else:
        p_times = Spke_Bundle["events"][period] / fs
        start, end = p_times[[0, -1]]
        mask = (sync_cam > start) & (sync_cam <= end)

    return sync_cam[mask], pupil_size[mask], pupil_center[:,mask], saccades

def get_valid_cluster(Spke_Bundle, SIN_data, spiketimes, colors,
                      units_for_plot=[]):
    """
    Gets valid units: (TCA, NW, BW).
    Parameters:
    - Spke_Bundle: Spike bundle dictionary
    - SIN_data: Supposedly inhibitory neuron dictionary
    - spiketimes: list, length N
    - colors: dict

    Returns:
    - valid_spiketimes: list, lenght n
    - cluster_type: list, lenght n, name of each valid cluster (TCA, NW, BW)
    - c_types: np.array, shape [n]
    """

    valid_cluster_indx = []
    cluster_type = []

    for neu_indx, GUI_name in enumerate(Spke_Bundle["classif_from_GUI"]["Classification"]):

        if neu_indx in SIN_data["Classif_SIN_indx"]:
            cluster_type.append("NW")
            valid_cluster_indx.append(neu_indx)

        elif neu_indx in SIN_data["Classif_SUR_indx"]:
            cluster_type.append("BW")
            valid_cluster_indx.append(neu_indx)

        elif GUI_name == 'MPW-Axon':
            cluster_type.append("TCA")
            valid_cluster_indx.append(neu_indx)
    
    valid_spiketimes = [spiketimes[i] for i in valid_cluster_indx]
    
    if units_for_plot:
        valid_spiketimes = [valid_spiketimes[u] for u in units_for_plot]
        cluster_type = [cluster_type[u] for u in units_for_plot]
        
    c_types = np.array([colors[n] for n in cluster_type])
    
    return valid_spiketimes, cluster_type, c_types

def get_firing_rate(spike_times, bt, win=0.1, n_jobs=-1):
    """
    Calculates a matrix of firing rates and z scored fr (n_units x time) using 
    the 2 indices method, considering a centered window around each camara frame. 

    Parameters:
    - spike_times: list of length n_units with np.ndarrays, shape (n_spikes)
    - bt: np.ndarray, shape (T)
    - win: float, time window size in seconds
    - n_jobs: int, number of parallel jobs (default -1 = use all cores)

    Returns:
    - firing_rate: ndarray of shape (n_units, len(bt))
    - z_fr: ndarray of shape (n_units, len(bt))
    """
    half_win = win / 2

    def compute_unit_rate(st):
        firing_rate = np.zeros(len(bt))
        start_idx, end_idx = 0, 0

        for ti, t in enumerate(bt):
            start_time = t - half_win
            end_time = t + half_win

            while start_idx < len(st) and st[start_idx] < start_time:
                start_idx += 1

            while end_idx < len(st) and st[end_idx] <= end_time:
                end_idx += 1

            firing_rate[ti] = end_idx - start_idx

        return firing_rate

    # Parallelize across neurons
    firing_rate = Parallel(n_jobs=n_jobs)(
        delayed(compute_unit_rate)(st) for st in spike_times
    )

    firing_rate = np.array(firing_rate) / win
    
    m_fr = np.expand_dims(np.mean(firing_rate, axis=1), axis=1)
    std_fr = np.expand_dims(np.std(firing_rate, axis=1), axis=1)    
    z_fr = (firing_rate - m_fr) / std_fr
    
    return firing_rate, z_fr

def get_mean_fr_size(fr, state,  start = 0.12, stop = 0.42, step = 0.02):
    """
    Calculates the mean and std of the firng rate for a given state.
    
    Parameters:
    - fr: np.ndarray, shape (n_neu, T)
    - state: np.ndarray, shape (T)
    - start, stop, step: float, edges for the bining of the state vector

    Returns:
    - mean_fr: np.ndarray, shape (n_neu,n_bins)
    - std_fr: np.ndarray, shape (n_neu,n_bins)
    - s_bins: np.ndarray, shape (n_bins)
    """
    
    s_edges = np.arange(start, stop, step)
    s_bins = (s_edges[:-1] + s_edges[1:]) / 2
    
    bin_indx = np.digitize(state, s_edges) - 1
    valid_mask = (bin_indx >= 0) & (bin_indx < len(s_bins))
    
    mean_fr = np.empty((fr.shape[0], len(s_bins)))
    std_fr = np.empty((fr.shape[0], len(s_bins)))
    
    for b in range(len(s_bins)):
        in_bin = (bin_indx == b) & valid_mask
        if np.any(in_bin):
            mean_fr[:, b] = np.mean(fr[:, in_bin], axis=1)
            std_fr[:, b] = np.std(fr[:, in_bin], axis=1)

    return mean_fr, std_fr, s_bins

def get_mean_fr_center(fr, pupil_center, edges):
    """
    Calculates the mean firng rate for 2d coordinates.
    
    Parameters:
    - fr: np.ndarray, shape (n_neu, T)
    - pupil_center: np.ndarray, shape (2,T)
    - edges: np.ndarray, shape (2, n_bins + 1)

    Returns:
    - mean_fr: np.ndarray, shape (n_neu,n_bins)
    """
    ix = np.digitize(pupil_center[0, :], edges) - 1
    iy = np.digitize(pupil_center[1, :], edges) - 1
    
    ix = np.clip(ix, 0, len(edges) - 2)
    iy = np.clip(iy, 0, len(edges) - 2)

    nx = len(edges) - 1
    n_bins = nx * (len(edges) - 1)
    bin_indx = ix + iy * nx

    mean_fr = np.empty((fr.shape[0], n_bins))    
    for b in range(n_bins):
        in_bin = bin_indx == b
        if np.any(in_bin):
            mean_fr[:, b] = np.mean(fr[:, in_bin], axis=1)
    
    return mean_fr

def get_correlation(z_fr, bt):
    """Gets the correlation between the firing rate matrix z_fr and the 
    behavioral variable bt
    
    Parameters:
    - z_fr: np.ndarray, shape (n_neu, T)
    - bt: np.ndarray, shape (T)

    Returns:
    - neu_bt_corr: np.ndarray, shape (n_neu)
    """
    n_neu = z_fr.shape[0]
    neu_bt_corr = np.empty(n_neu)
    for n in range(n_neu):
        neu_bt_corr[n] = np.corrcoef(z_fr[n,:], bt)[0,1]

    return neu_bt_corr
    
def get_similarity(mean_fr, cluster_type, colors):
    """
    Calculates the cosine similarity matrix of the classes for each type 
    of unit.

    Parameters:
    mean_fr : np.ndarray, shape (n_neu, n_class)
    cluster_type : np.ndarray, shape (n_neu)
    colors: dictionary with the keys of valid units names

    Returns:
    similarity_type : dictionary with matrices shape (n_class, n_class)
    """
    cluster_type = np.asarray(cluster_type)
    unique_type = np.unique(cluster_type)
    similarity_type = dict.fromkeys(colors, None)
    
    for neu_type in unique_type:
        mean_type = mean_fr[cluster_type == neu_type, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_mean_type = mean_type / np.linalg.norm(mean_type,axis=0)
        similarity_type[neu_type] = norm_mean_type.T @ norm_mean_type
        
    return similarity_type

def get_fr_aligned(fr, align_indx, win=[-5,5], camara_fs = 200):
    """
    Mean firing rate centered on the align indices
    
    Parameters:    
    fr : np.ndarray, shape (n_neu, T)
    align_indx: np.ndarray, shape (n_events)
    win: [pre,post] time window (tw) around the align times [seconds] 

    Returns:
    mean_fr : np.ndarray, shape (n_neu, Tw)
    tw: np.ndarray, shape (Tw)
    """
    
    n_unit = fr.shape[0]
    
    tiw = np.arange(win[0]*camara_fs, win[1]*camara_fs, dtype=int)
    tw = tiw / camara_fs
    
    mean_fr = np.empty((n_unit, len(tiw)))
    for i, ti in enumerate(tiw):
        mean_fr[:, i] = np.mean(fr[:, align_indx + ti], axis=1)

    return mean_fr, tw

def get_response_times(fr, align_indx, win=[-1,-0.2, 1], thres=0.1, n_tw = 5,
                       camara_fs = 200):
    """
    Obtains the reaction times for each neuron, i.e. when AUROC with a basal 
    distribution goes over a threshold for n_tw consecutive windows.
    
    Parameters:    
    fr : np.ndarray, shape (n_neu, T)
    align_indx: np.ndarray, shape (n_events)
    thres: float, threshold for the difference of auroc to 0.5
    n_tw: int, number of consecutive windows above the thres
    win: [basal_start, basal_end, post_end]  

    Returns:
    rts : np.ndarray, shape (n_neu)
    """
    
    n_unit = fr.shape[0]
    
    tiw_basal = np.arange(win[0]*camara_fs, win[1]*camara_fs, dtype=int)   
    tiw_post = np.arange(win[1]*camara_fs, win[2]*camara_fs, dtype=int)    
         
    rts = np.full(n_unit, np.nan)
    for n in range(n_unit):
        
        idx = align_indx[:, None] + tiw_basal[None, :]
        basal_fr = fr[n, idx].flatten() # shape (n_trials, tw)
        
        idx = align_indx[:, None] + tiw_post[None, :]
        post_fr = fr[n, idx]

        i = 0
        sig_tw = 0
        while i < len(tiw_post) and np.isnan(rts[n]):
            post_fr_t = post_fr[:,i]
            
            y_true = np.concatenate([np.zeros_like(basal_fr), 
                                     np.ones_like(post_fr_t)])
            scores = np.concatenate([basal_fr, post_fr_t])

            auc = roc_auc_score(y_true, scores)

            if np.abs(auc - 0.5) > thres:
                sig_tw += 1
                if sig_tw == n_tw:
                    rts[n] = (tiw_post[i] - n_tw) / camara_fs
            else:
                sig_tw = 0
            
            i += 1

    return rts

def get_mean_fr_2d(mean_fr, embedding, emb_p, c_types, sigma=1):
    """
    Compute the mean fr weighted by a gaussian distances of 2D points from 
    a list of centers. Also gives a mean color based on c_types.
    
    Parameters:
    mean_fr: np.ndarray, shape (n,Tw)
    embedding: np.ndarray, shape (n, 2)
    emb_p: np.ndarray, shape (n_points, 2)
    c_types: np.ndarray, colors in Hex, shape (n)
    sigma: float, standard deviation of the Gaussian.
    
    Returns:
    mean_fr_p: ndarray, shape (n_points,Tw)
    mean_c_p: list, lenght (n_points)
    """
    
    rgb_colors = np.array([pltcolors.to_rgb(c) for c in c_types])
    
    mean_fr_p = np.empty((emb_p.shape[0], mean_fr.shape[1]))
    mean_c_p = []
    
    for p in range(emb_p.shape[0]):
        diff = embedding - emb_p[p,:]
        dist2 = np.sum(diff**2, axis=1)  
        dist = np.exp(-dist2 / (2 * sigma**2))

        mean_fr_p[p,:] = np.average(mean_fr,axis=0,weights=dist)

        avg_rgb = np.average(rgb_colors, axis=0, weights=dist) 
        mean_c_p.append(pltcolors.to_hex(avg_rgb))
        
    return mean_fr_p, mean_c_p

def neuron_PCA(fr, types, n_components = 1):
    """
    Applies PCA and projects the data, concatenating time and class for every 
    unit type.

    Parameters:
    fr: np.ndarray, shape (n,t,c)
    types: np.ndarray, shape (n)
    n_components: int
    
    Returns:
    projection_typ: ndarray, shape (n_typ, n_components, t, c)
    """
    n, t, c = fr.shape
    n_typ = len(np.unique(types))
    projection_typ = np.empty((n_typ, n_components, t, c))
    
    for typ_i, ty in enumerate(np.unique(types)):
        fr_ty = fr[ty == types, :, :]
        n = fr_ty.shape[0]
        
        X = fr_ty.reshape(n, t * c).T

        pca = PCA(n_components=n_components)
        projection = pca.fit_transform(X)  
        projection_typ[typ_i,:,:,:] = projection.T.reshape(n_components, t, c)

    return projection_typ

def get_t_significance(all_ps_corr, all_types_cat, n_p=1000):
    """
    Gets significance using a permutation t-test for TCA>NW and NW>BW.

    Parameters:
    all_ps_corr : np.array, [n]
    all_types_cat : list, [n]
    n_p : int

    Returns:
        None
    """
    all_types_cat = np.asarray(all_types_cat)
    TCA = all_ps_corr[all_types_cat == "TCA"]
    NW  = all_ps_corr[all_types_cat == "NW"]
    BW  = all_ps_corr[all_types_cat == "BW"]

    def t_stat(x, y):
        return np.mean(x) - np.mean(y)

    # TCA vs NW
    res1 = stats.permutation_test((TCA, NW), statistic=t_stat, n_resamples=n_p,
                                  alternative="greater", random_state=0)

    # NW vs BW
    res2 = stats.permutation_test((NW, BW), statistic=t_stat, n_resamples=n_p,
                                  alternative="greater", random_state=0)

    print(f"TCA > NW: p = {res1.pvalue:.4g}, "
          f"NW > BW: p = {res2.pvalue:.4g}")

##m Plotting

def plot_pupil_results(tv, pupil_size, pupil_size_clean, pupil_center, 
                      eyelids_mean, saccade_indx, name, sp):
    colors = {"size":"#9b5de5",
              "size_clean":"#D3B9F4",
              "x":"#00bbf9",
              "y":"#00f5d4",
              "temporal":"navy",
              "nasal":"violet"}
    
    tv = tv / 60 # minutes
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])  # 2 rows, 2 columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # First subplot: pupil size    
    ax1.plot(tv, pupil_size, color=colors["size"])
    ax1.plot(tv, pupil_size_clean, color=colors["size_clean"])

    ax1.set_xlim([tv[0],tv[-1]])
    ax1.set_ylim([0,0.5])
    ax1.set_xticks([])
    ax1.set_ylabel("norm pupil size")
    ax1.spines[['right', 'top']].set_visible(False)
    
    # Second subplot: pupil center
    ax2.plot(tv, pupil_center[0, :], color=colors["x"], label="x")
    ax2.plot(tv, pupil_center[1, :], color=colors["y"], label="y")
    
    ylim = ax2.get_ylim()
    for s in ["temporal","nasal"]:
        saccades = saccade_indx[s]
        for idx in saccades:
            ax2.vlines(tv[idx], ylim[0], ylim[1], 
                       colors=colors[s], linestyles="--")
    ax2.set_ylim(ylim)
    
    ax2.set_xlim([tv[0], tv[-1]])
    ax2.set_xlabel("time [m]")
    ax2.set_ylabel("Coordinate")
    ax2.legend()
    ax2.spines[['right', 'top']].set_visible(False)

    # Third subplot: 2D histogram
    xedges = np.linspace(np.min(eyelids_mean[:,0]),np.max(eyelids_mean[:,0]),200)
    yedges = np.linspace(np.min(eyelids_mean[:,1]),np.max(eyelids_mean[:,1]),200)
    h = ax3.hist2d(pupil_center[0, :], pupil_center[1, :], bins=[xedges,yedges], 
                      density=True, cmap='hot_r')
    ax3.scatter(eyelids_mean[:,0], eyelids_mean[:,1], color="dodgerblue",zorder=2)
    cbar = fig.colorbar(h[3], ax=ax3, orientation='horizontal')
    cbar.set_label("Density")
    
    ax3.yaxis.set_inverted(True) 
    ax3.set_axis_off()
    
    fig.suptitle(name)
    plt.tight_layout()
    plt.savefig(os.path.join(sp, "plots", name + "_pupil_plot.svg"))
    plt.show()

def plot_exp(Spke_Bundle, sync_cam, vis_stim, colors, 
             name, sp, fs = 30000, y = 1.0, lw=0.25):
    
    # Visual stimulus
    ns = len(vis_stim)
    for s, stim in enumerate(vis_stim):
        times = Spke_Bundle["events"][stim] / fs
        times = (times - sync_cam[0]) / 60
        plt.vlines(times, 1 - s / ns, 1 - (s + 1) / ns,
                   color=colors[s], label=vis_stim[s], linewidth=lw)
    
    # Camaras
    change_i = np.argwhere(np.diff(sync_cam) > 5).squeeze()
    periods_i = [[0, change_i[0]]] 
    for i in range(len(change_i)-1):
        periods_i.append([change_i[i] + 1, change_i[i+1]])
    periods_i.append([change_i[-1] + 1, -1])
    periods = np.array([sync_cam[idx] for idx in periods_i])
    periods = (periods - sync_cam[0]) / 60
    
    cmap = plt.get_cmap('Greys', periods.shape[0]+1)
    colors = [cmap(i+1) for i in range(periods.shape[0])]
    for p in range(periods.shape[0]):
        plt.plot(periods[p,:], [y + 0.1, y + 0.1], color=colors[p])    
    plt.legend()
    plt.xlabel("t")
    plt.yticks([])
    for s in ['right', 'top', 'left']:
        plt.gca().spines[s].set_visible(False)
    plt.savefig(os.path.join(sp,"plots", name + "_session.svg"))
    plt.show()

def plot_ps_pc(all_ps, all_pc, sp):
    
    ps_pc = np.zeros((len(all_ps)))
    for i in range(len(all_ps)):
        dps = signal.savgol_filter(all_ps[i], window_length=500, 
                                  polyorder=2, deriv=1)

        dx = signal.savgol_filter(all_pc[i][0,:], window_length=500, 
                                  polyorder=2, deriv=1)
        dy = signal.savgol_filter(all_pc[i][1,:], window_length=500, 
                                  polyorder=2, deriv=1)
        dpc = np.sqrt(dx**2 + dy**2)

        ps_pc[i] = np.corrcoef(dps, dpc)[0,1]
    
    plt.bar(0, np.mean(ps_pc), color="black")
    plt.scatter(np.zeros((len(all_ps))), ps_pc,c= 'grey')
    print(ps_pc)
    
    plt.ylabel("r coef")
    plt.ylim([-0.51,0.51])
    plt.xticks([])
    for s in ['right', 'top', 'bottom']:
        plt.gca().spines[s].set_visible(False)
    
    plt.savefig(os.path.join(sp,"plots", "ps_pc_corr.svg"))
    plt.show()

def plot_pupil_stimuli(pupil_size, pupil_center, sync_cam, periods,
                       vis_stim, colors,  exp, sp, fs = 30000):

    colors.append((0.5, 0.5, 0.5, 0.5)) # gray / no stim
    
    ps_stim = []
    pc_stim = []
    for stim in vis_stim:
        stim_times = periods[stim] / fs
        start, end = stim_times[[0, -1]]
        mask = (sync_cam > start) & (sync_cam <= end)
        if stim==vis_stim[0]:
            mask_first = (sync_cam < start)
            
        ps_stim.append(pupil_size[mask])
        pc_stim.append(pupil_center[:,mask])
        
    ps_stim.append(pupil_size[mask_first])
    pc_stim.append(pupil_center[:,mask_first])
    vis_stim.append("grey")
    
    # Size
    fig, ax = plt.subplots()
    ax.set_ylabel('Pupil size')
    
    bplot = ax.boxplot(ps_stim, patch_artist=True, sym="", labels = vis_stim)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylim([0, 0.5])
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)
    plt.savefig(os.path.join(sp,"plots", exp + "_ps_stim.svg"))
    plt.show()

    # Position
    for s in range(len(vis_stim)):
        plt.subplot(3,3,s+1)
        plt.plot(pc_stim[s][0,:], pc_stim[s][1,:], color=colors[s], 
                 linewidth=0.5)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(os.path.join(sp,"plots", exp + "_pc_stim.svg"))
    plt.show()

def plot_correlation_hist(corr, cluster_type, colors, edges, name, sp):
    
    cluster_type = np.asarray(cluster_type)
    unique_type = np.unique(cluster_type)
    mean_type = dict.fromkeys(colors, None)
    
    # Hist
    for neu_type in unique_type:
        corr_type = corr[cluster_type == neu_type]
        mean_type[neu_type] = np.mean(corr_type)
        plt.hist(corr_type, edges, 
                 density=True, histtype='step', fill=False,
                 edgecolor=colors[neu_type], label=f"{neu_type}")
    
    _, ylim = plt.ylim()
    
    #Scatter
    x = list(mean_type.values())
    y = [ylim] * len(mean_type)
    c = [colors[k] for k in mean_type.keys()]    
    plt.scatter(x, y, c=c, marker='v')
    
    plt.vlines(0,0,ylim,colors="gray",linestyles="dashed")
    plt.xlabel("r coef")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim([edges[0],edges[-1]])
    plt.ylim(0,ylim)
    for s in ['right', 'top']:
        plt.gca().spines[s].set_visible(False)
    plt.title(name)
    
    plt.savefig(os.path.join(sp,"plots", name + "_corr.svg"))
    plt.show()

def plot_metric_typ_cum(metric, cluster_type, colors, edges, name, sp):
    
    cluster_type = np.array(cluster_type)
    unique_type = np.unique(cluster_type)
    
    # Hist
    for neu_type in unique_type:
        metric_type = metric[(cluster_type == neu_type) & (~np.isnan(metric))]
        plt.hist(metric_type, edges, 
                 density=True, histtype='step', fill=False, cumulative=1,
                 edgecolor=colors[neu_type], 
                 label=f"{neu_type}" + str(len(metric_type)))
    
    plt.ylim([0,1])
    plt.yticks([1, 0.5, 0])
    plt.xlim([edges[0], edges[-1] - 1e-2])

    plt.vlines(0,0,1,colors="gray",linestyles="dashed")
    plt.xlabel("time [s]")
    plt.ylabel("Cum density")
    plt.legend()
    
    for s in ['right', 'top']:
        plt.gca().spines[s].set_visible(False)
    plt.title(name)
    
    plt.savefig(os.path.join(sp,"plots", name + "_cum.svg"))
    plt.show()

def plot_similarity_2d(similarity_type, plot_bin, edges, name, sp, clim=[-1,1]):
    bins_1d = (edges[1:] + edges[:-1]) / 2
    nx = len(bins_1d)
    row, col = divmod(plot_bin, nx)
    
    fig, axes = plt.subplots(1,len(similarity_type)+1, figsize=(12, 6))

    # Plot each similarity matrix
    for i, nt in enumerate(similarity_type.keys()):
        sim_matrix = similarity_type[nt][plot_bin, :]
        im = axes[i].imshow(sim_matrix.reshape(nx, nx), cmap="Spectral",
                            extent=[edges[0], edges[-1], edges[0], edges[-1]], 
                            vmin=clim[0], vmax=clim[1], origin='lower')
        axes[i].scatter(bins_1d[col], bins_1d[row], color="black")
        #axes[i].invert_yaxis()
        axes[i].axis('off')
        axes[i].set_title(nt)
        
    
    cbar = fig.colorbar(im, ax=axes[3])
    axes[3].axis('off')
    cbar.set_label("Cosine similarity")
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(os.path.join(sp,"plots", name + "_bin_" + str(plot_bin) +
                             "_CS.svg"))
    plt.show()

def plot_windows_and_events(b, sync_cam, event_t=[], pl = 10, ylim = [0, 0.5],
                            name="size"):
    
    nframes = b.shape[-1]  
    p = nframes//pl
    tws = [np.arange(t*p, (t+1)*p) for t in range(pl)]
    
    for tw in tws:
        fig, ax1 = plt.subplots()
        
        if b.ndim == 1:
            ax1.plot(sync_cam[tw], b[tw], color="#D3B9F4")
        else:
            ax1.plot(sync_cam[tw], b[0,tw], color="#00bbf9")
            ax1.plot(sync_cam[tw], b[1,tw], color="#00f5d4")
        
        if len(event_t) > 0:
            mask = (event_t >= sync_cam[tw[0]]) & \
                   (event_t < sync_cam[tw[-1]])
            win_change = event_t[mask]
            for ti in win_change:
                ax1.vlines(ti, ylim[0], ylim[1], colors="k", linestyles="--", alpha=0.3)

        ax1.set_ylim(ylim)
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel(name)
        plt.show()
        
def plot_fr_aligned(tw, mean_fr, c_types, sp="none", name="fr_aligned"):        
    for n in range(mean_fr.shape[0]):
        fig = plt.figure()
        if mean_fr.ndim == 2:
            plt.plot(tw, mean_fr[n,:], 
                     color=c_types[n])
        else:
            plt.plot(tw, mean_fr[n,:,0], 
                     color=c_types[n])
            plt.plot(tw, mean_fr[n,:,1], 
                     color=c_types[n], linestyle="dashed")
        plt.xlim([tw[0],tw[-1]])
        plt.xlabel("time [s]")
        plt.ylabel("firing rate")
        for s in ['right', 'top']:
            plt.gca().spines[s].set_visible(False)
        if sp == "none":
            plt.show()
        else:
            plt.savefig(os.path.join(sp,"plots", "Neurons",
                                     str(n) + name +".png"))
            plt.close(fig)

def plot_raster(st, sync_cam, align_indx, colors, 
                tw, mean_fr, c_types, sp="none", name="fr_aligned.png"):
    
    for n, spikes in enumerate(st):

        aligned_spikes = []
        for ti in align_indx:
            spk = np.array(spikes) - sync_cam[ti]
            tw_spks = (spk > tw[0]) & (spk < tw[-1]) 
            aligned_spikes.append(spk[tw_spks])
        
        fig, axes = plt.subplots(2,1, figsize=(10, 8))
        
        if colors:
            axes[0].eventplot(aligned_spikes, colors=colors)  
        else:
            axes[0].eventplot(aligned_spikes)
            
        axes[0].set_xlim([tw[0],tw[-1]])
        axes[0].axis('off')
        
        if mean_fr.ndim == 3:
            axes[1].plot(tw, mean_fr[n,:,0], color=c_types[n])
            axes[1].plot(tw, mean_fr[n,:,1], color=c_types[n], 
                         linestyle="dashed")
        else:
            axes[1].plot(tw, mean_fr[n,:], color=c_types[n])
            
        axes[1].set_xlim([tw[0],tw[-1]])
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("firing rate")
        plt.tight_layout()
        for s in ['right', 'top']:
            axes[1].spines[s].set_visible(False)
            
        if sp == "none":
            plt.show()
        else:
            plt.savefig(os.path.join(sp,"plots", "Neurons", str(n) + name))
            plt.close(fig)
        
def plot_umap(embedding, c_types, sp, emb_p = [], mean_emb_c = []):
    plt.scatter(embedding[:,0], embedding[:,1],c=c_types,
                alpha=0.8, edgecolors="none")
    if emb_p:
        plt.scatter(emb_p[:,0], emb_p[:,1],c=mean_emb_c, 
                    marker="s",edgecolors="white")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(os.path.join(sp,"plots", "ps_UMAP.svg"))
    plt.show()

def plot_angle(pc_angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    
    for a in pc_angles:
        ax.plot([a, a], [0, 1], alpha=0.7, color="black")
    
    ax.set_yticklabels([])
    plt.show()
    
def plot_event(events, b, name, exp, sp, win = [-0.25, 0.25], camara_fs=200):
    
    tiw = np.arange(win[0]*camara_fs, win[1]*camara_fs, dtype=int)
    tib = np.arange(win[0]*camara_fs, 0, dtype=int)
    tw = tiw / camara_fs
    
    if isinstance(events, dict):
        for e in events["temporal"]:
            event_b = b[e + tiw] - np.mean(b[e + tib])
            plt.plot(tw, event_b, color="navy")
        for e in events["nasal"]:
            event_b = b[e + tiw] - np.mean(b[e + tib])        
            plt.plot(tw, event_b, color="violet")    

    else:
        for e in events:
            event_b = b[e + tiw] - np.mean(b[e + tib])
            plt.plot(tw, event_b, color="black", alpha=0.6)
        
    plt.xlabel("time [s]")
    plt.ylabel(name)
    for s in ['right', 'top']:
        plt.gca().spines[s].set_visible(False)
    plt.savefig(os.path.join(sp,"plots", exp + name + ".svg"))
    plt.show()
    
def plot_pca(tw, pca_pc, colors, sp, multi_d=False, name="PCA_sc.svg"):
    types = colors.keys()
    if not multi_d:
        for ti, typ in enumerate(types):       
            
            fig, axes = plt.subplots(pca_pc.shape[1], 1, figsize=(10, 8))
            for c in range(pca_pc.shape[1]): 
                
                axes[c].plot(tw, pca_pc[ti,c,:,0], color="navy")
                axes[c].plot(tw, pca_pc[ti,c,:,1], color="violet")
                
                ylim = axes[c].get_ylim()
                axes[c].vlines(0, ylim[0], ylim[1], 
                                colors="k", linestyles="--", alpha=0.3)
                
                axes[c].tick_params(axis='y', labelcolor=colors[typ])
                axes[c].set_ylabel("PC" + str(c+1))
                if c < pca_pc.shape[1] - 1:
                    axes[c].set_xticks([])
                else:
                    axes[c].set_xlabel("time [s]")
                    axes[c].tick_params(axis='x', labelcolor=colors[typ])
                    
                for s in ['right', 'top']:
                    axes[c].spines[s].set_visible(False)
                for s in ['left', 'bottom']:
                    axes[c].spines[s].set_color(colors[typ])
                    
            plt.tight_layout()
            plt.savefig(os.path.join(sp,"plots", typ + name))

            plt.show()
    else:
        fig, axes = plt.subplots(1, len(types), figsize=(12, 8), 
                                 subplot_kw={'projection': '3d'})
        for ti, typ in enumerate(types):
            axes[ti].plot3D(pca_pc[ti,0,:,0], pca_pc[ti,1,:,0], pca_pc[ti,2,:,0],
                        color=colors[typ])
            axes[ti].plot3D(pca_pc[ti,0,:,1], pca_pc[ti,1,:,1], pca_pc[ti,2,:,1],
                        color=colors[typ], linestyle="dashed")
            
            axes[ti].set_xlabel("PC1")
            axes[ti].set_ylabel("PC2")
            axes[ti].set_zlabel("PC3")
            
            axes[ti].set_xticks([])
            axes[ti].set_yticks([])
            axes[ti].set_zticks([])
            
        plt.tight_layout()
        plt.show()

def plot_types(experiments, all_types, colors, sp = "none"):
    
    exp = [e[:10] for e in experiments]
    all_counts = {n_type: [] for n_type in colors.keys()}
    for e in range(len(exp)):
        unique, counts = np.unique(all_types[e], return_counts=True)
        for n,n_type in enumerate(unique):
            all_counts[n_type].append(counts[n])     

    fig, ax = plt.subplots()
    bottom = np.zeros(len(exp))
    
    for types, color in colors.items():
        p = ax.bar(exp, all_counts[types], bottom=bottom, color=color)
        bottom += all_counts[types]
    
        ax.bar_label(p, label_type='center', color="white")
    
    ax.set_ylabel('n units')
    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)    
    
    if not sp == "none":
        plt.savefig(os.path.join(sp,"plots", "n neurons.svg"))
    plt.show()
    
