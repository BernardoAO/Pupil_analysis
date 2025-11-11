## Helper functions
### Bernardo AO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
import pandas as pd
from joblib import Parallel, delayed
import os

def create_pupil_data(output_variables):
    """
    Creates a dictionary, with each experiment day as a key, of panda arrays 
    with the columns names specified in output_variables. Each column is 
    initialized as an empty array, with the exception of session, corresponding
    to the video names, and awake, which is True for awake sessions.
    
    Parameters:
    - output_variables : list
    
    Returns:
    - pupil_data : dictionary
    """
    spike_bundle_path = r"D:\NP data\analysis\data-single-unit" 
    video_path = r"D:\NP data\Bernardo_awake_cx\DLC\left_eye\all_videos"
    
    experiments = sorted([exp for exp in os.listdir(spike_bundle_path) 
                          if exp.startswith("20")])
    videos = sorted([video for video in os.listdir(video_path) 
              if video.startswith("cam")])
    num_awake = np.array([0,0, #2021
                          4,4, #2022
                          4,4,4,4,4,3,5,5,3,5,4,5,7,7,6,0,0,0]) #2023
    pupil_data = {}

    for e, exp in enumerate(experiments):
        date = exp[:10]
        exp_videos = [v[5:24] for v in videos if v[5:15] == date]
        
        if exp == "2023-03-15_11-05-00": # days with 2 exp
            exp_videos = [v for v in exp_videos if int(v[16:18]) < 15]
        elif exp == "2023-03-15_15-23-14":
            exp_videos = [v for v in exp_videos if int(v[16:18]) <= 15]
        elif exp == "2023-08-10_13-07-52":
            exp_videos = [v for v in exp_videos if int(v[16:18]) < 16]
        elif exp == "2023-08-10_16-32-27":
            exp_videos = [v for v in exp_videos if int(v[16:18]) <= 16]
        
        awake = [i <= num_awake[e] for i in range(len(exp_videos))]
        data = {
            "session": exp_videos,
            "awake": awake,
        }
        for var in output_variables:
            if var != "session" and var != "awake":
                data[var] = [np.array([])] * len(exp_videos)

        pupil_data[exp] = pd.DataFrame(data)

    return pupil_data    

def handle_exceptions(ROIs_smooth, tv, session):
    if session == "2023-03-16-11-14-37":
        mask = tv < 0.15 * 60
        ROIs_smooth[:,:,mask] = np.nan
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

def get_saccades(retina_center, thr=3):
    """
    Estimates saccade times.

    Parameters:
    - retina_center : np.ndarray, shape (2, T)
    - thr: float, threshold of detection

    Returns:
    - saccade_indx : np.ndarray, shape (T,)
    """
    diff = np.diff(retina_center, axis=1)
    distance = np.sqrt(diff[0,:]**2 + diff[1,:]**2)
    saccade_indx = np.where(distance > thr)[0]
    return saccade_indx
    
def import_spike_data(exp, path_2_spike_bundle, 
                      sampling_rate = 30000):
    """
    Imports spike bundle data from the _Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy files,
    and supposedly inhibitory neuron data from the '_local_storage_SIN.npy' files
    """
    # load spike data for each unit
    Spke_Bundle_name = os.path.join(path_2_spike_bundle, exp,
                                    exp + '_Complete_spiketime_Header_TTLs_withdrops_withGUIclassif.npy')
    Spke_Bundle = \
        np.load(Spke_Bundle_name, allow_pickle=True,encoding='latin1').item()
    spiketimes = [unit_times / sampling_rate
                  for unit_times in Spke_Bundle['spiketimes_aligned']]  
    
    # load Supposedly inhibitory neurons file
    path_SIN_data =  \
        os.path.join(path_2_spike_bundle, exp, 
                        exp + '_local_storage_SIN.npy')
    SIN_data =  \
        np.load(path_SIN_data,allow_pickle=True,encoding='latin1').item()

    return Spke_Bundle, spiketimes, SIN_data

def get_pupil_data(pupil_data, Spke_Bundle, exp, period, fs = 30000):

    # Exceptions
    if exp == '2023-03-16_12-16-07':
        pupil_data = pupil_data.iloc[1:]
    
    pupil_sizes_pd = pupil_data.loc[pupil_data['awake'], 'pupil_size']
    pupil_size = np.concatenate(pupil_sizes_pd.to_numpy())
    pupil_center_pd = pupil_data.loc[pupil_data['awake'], 'pupil_center']
    pupil_center = np.concatenate(pupil_center_pd.to_numpy(), axis = 1)
    
    n_points_ses = [len(s) for s in pupil_sizes_pd]
    n_frames = sum(n_points_ses)
    
    sync_all_cam = Spke_Bundle["Synchronization_TTLs"]["Sync_cam"] / fs
    sync_cam = sync_all_cam[:n_frames]
    
    if period == "all":
        mask = np.ones(len(sync_cam), dtype=bool)
    else:
        p_times = Spke_Bundle["events"][period] / fs
        start, end = p_times[[0, -1]]
        mask = (sync_cam > start) & (sync_cam <= end)
        

    return sync_cam[mask], pupil_size[mask], pupil_center[:,mask]

def get_valid_cluster(Spke_Bundle, SIN_data):
    """
    Gets valid units: (TCA, NW, BW).
    Parameters:
    - Spke_Bundle: Spike bundle dictionary
    - SIN_data: Supposedly inhibitory neuron dictionary

    Returns:
    - valid_cluster_indx: indices where valid_cluster is True
    - cluster_type: Name of each valid cluster (TCA, NW, BW)
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
    
    return valid_cluster_indx, cluster_type

def get_firing_rate(spike_times, bt, win=0.1, n_jobs=-1):
    """
    Calculates a matrix of firing rates (n_units x time), by looking at
    a time window around each cam frame.

    Parameters:
    - spike_times: list of length n_units with np.ndarrays, shape (n_spikes)
    - bt: np.ndarray, shape (T)
    - win: float, time window size in seconds
    - n_jobs: int, number of parallel jobs (default -1 = use all cores)

    Returns:
    - firing_rates: ndarray of shape (n_units, len(bt))
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
    firing_rates = Parallel(n_jobs=n_jobs)(
        delayed(compute_unit_rate)(st) for st in spike_times
    )

    firing_rates = np.array(firing_rates)
    return firing_rates / win

def get_mean_fr_size(fr, state, 
                     start = 0.05, stop = 0.35, step = 0.02):
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

##m Plotting

def plot_pupil_results(tv, pupil_size, pupil_size_clean, pupil_center, 
                      eyelids_mean, saccade_indx, name, sp):

    tv = tv / 60 # minutes
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])  # 2 rows, 2 columns
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # First subplot: pupil size    
    ax1.plot(tv, pupil_size, color="#9b5de5")
    ax1.plot(tv, pupil_size_clean, color="#D3B9F4")

    ax1.set_xlim([tv[0],tv[-1]])
    ax1.set_ylim([0,0.5])
    ax1.set_xticks([])
    ax1.set_ylabel("norm pupil size")
    ax1.spines[['right', 'top']].set_visible(False)
    
    # Second subplot: pupil center
    ax2.plot(tv, pupil_center[0, :], color="#00bbf9", label="x")
    ax2.plot(tv, pupil_center[1, :], color="#00f5d4", label="y")
    
    ylim = ax2.get_ylim()
    for idx in saccade_indx:
        ax2.vlines(tv[idx], ylim[0], ylim[1], colors="k", linestyles="--", alpha=0.3)
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

def plot_correlation(corr, cluster_type, colors, edges, name, sp):
    
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

def plot_exp(Spke_Bundle, sync_cam, name, sp, fs = 30000, y = 1.0):
    
    vis_stim = ["Sl36x22_d_3","Sd36x22_l_3", "mb", 
                "Nat_Mov", "Nat_Mov_sw", "Nat_Mov_sc",
                "csd", "chirp"] 
    vis_name = ["□ on light", "□ on dark","Moving bars",
                "Nat Mov", "Nat Mov swapped", "Nat Mov scrambled",
                "Current Source Density", "Chirp"]
    cmap = plt.get_cmap('jet', len(vis_stim))
    colors = [cmap(i) for i in range(len(vis_stim))]
    ns = len(vis_stim)
    for s, stim in enumerate(vis_stim):
        times = Spke_Bundle["events"][stim] / fs
        times = (times - sync_cam[0]) / 60
        plt.vlines(times,1 - s / ns, 1 - (s + 1) / ns,
                   color=colors[s], label=vis_name[s])
    
    change_index = np.argwhere(np.diff(sync_cam) > 5).squeeze()
    change_t = sync_cam[change_index]
    periods = np.array([[sync_cam[0],change_t[0]],
                        [change_t[0] + 1, change_t[1]],
                        [change_t[1] + 1, sync_cam[-1]]])
    periods = (periods - sync_cam[0]) / 60
    
    cmap = plt.get_cmap('Greys', periods.shape[0]+1)
    colors = [cmap(i+1) for i in range(periods.shape[0])]
    for p in range(periods.shape[0]):
        plt.plot(periods[p,:],[y + 0.1, y + 0.1],color=colors[p])    
    plt.legend()
    plt.xlabel("t")
    #plt.xlim([0,4000])
    plt.yticks([])
    for s in ['right', 'top', 'left']:
        plt.gca().spines[s].set_visible(False)
    plt.savefig(os.path.join(sp,"plots", name + "_session.svg"))
    plt.show()