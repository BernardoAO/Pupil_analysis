## Archive of neural pupil coding
### Bernardo AO

ps_corr_edges = np.arange(-0.3, 0.32, 0.01)
pc_corr_edges = np.arange(0., 0.2, 0.005)

    if plot == "proj":
        trial_fr, _, tw = hf.get_fr_aligned(firing_rate, saccades_all, win=win)    
        proj = np.tensordot(models["x"][:, 0], trial_fr, axes=(0, 0))
        hf.plot_proj(tw, proj, delta_x, sac_colors)

def corr_analysis(z_fr, pupil_size, cluster_type, colors, ps_corr_edges, 
                  save_path, exp, plot="none", cum=True, m_name="norm_pupil_size"):
        
    if isinstance(plot, int):

        # Mean fr vs pupil size
        stats_fr, s_bins = \
            hf.get_mean_fr_size(z_fr, pupil_size, per=[20,80])
            
        hf.plot_ps_exp(stats_fr, s_bins, colors, cluster_type, plot, save_path)
        
    # fr pupil size correlation
    neu_pupil_corr = hf.get_correlation(z_fr, pupil_size)
    
    if m_name == "pupil x":
        neu_pupil_corr = np.abs(neu_pupil_corr)
        xlabel = "|r|"
    else:
        xlabel = "r coef."
    
    if plot == "hist":
        
        hf.plot_hist_typ(neu_pupil_corr, cluster_type, colors, ps_corr_edges, 
                         save_path, exp, m_name, cum, xlabel)
    
    return neu_pupil_corr

def ps_events_analysis(pupil_size, fr, valid_spiketimes, sync_cam, c_types, 
                       exp, save_path, win = [-0.5,2], plot="none"):
    
    ps_change_indx, _ = hf.get_events(pupil_size, window_pre = 10, rp = 10)
    
    fr_ps, tw = hf.get_fr_aligned(fr, ps_change_indx, win = win)
    
    if plot == "all" or plot == "pupil":
        hf.plot_windows_and_events(pupil_size, sync_cam, sync_cam[ps_change_indx])
        hf.plot_event(ps_change_indx, pupil_size, "pupil size", exp, save_path, 
                      win = win)
        
    if plot == "all" or plot == "raster":
        hf.plot_raster(valid_spiketimes, sync_cam, ps_change_indx, [],
                       tw, fr_ps, c_types, save_path, name="fr_ps.svg")
        
    return fr_ps

def pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
                save_path, center_edges = np.arange(105, 145, 5)):
    
    mean_fr_center = hf.get_mean_fr_center(firing_rate, pupil_center, center_edges)

    similarity_type = hf.get_similarity(mean_fr_center, cluster_type, colors)

    plot_bin = 19
    hf.plot_similarity_2d(similarity_type, plot_bin, 
                          center_edges, plot_name, save_path,clim=[-1,1])
    

## Main script
analysis = "" # ps_pc_corr
              # ps_corr, pc_corr, ps_ev, pc_sim, 

    # Pupil size
    hf.plot_pupil_stimuli(pupil_size, pupil_center, sync_cam, 
                          Spke_Bundle["events"], vis_stim, stim_colors, 
                          exp, save_path)
    
    elif analysis == "ps_pc_corr":
        results["ps"].append(pupil_size)
        results["pc"].append(pupil_center)

        ## Pupil size
        
        if analysis == "ps_corr": # correlation
            neu_pupil_corr = \
                corr_analysis(z_fr, pupil_size, cluster_type, colors, 
                              ps_corr_edges, save_path, exp) # plot=1
            results["ps_corr"].append(neu_pupil_corr)
        
        
        if analysis == "pc_corr": # correlation
            neu_pupil_corr = \
                corr_analysis(z_fr, pupil_center[0,:], cluster_type, colors, 
                              pc_corr_edges, save_path, exp, plot="hist", 
                              m_name = "pupil x", cum=False)
                
            results["pc_corr"].append(neu_pupil_corr)

        elif analysis == "ps_ev": # size change events
            z_fr_ps = ps_events_analysis(pupil_size, firing_rate, valid_spiketimes, sync_cam, 
                                         c_types, exp, save_path, plot = "raster")
            results["fr_ps"].append(z_fr_ps)

        elif analysis == "pc_sim": # similarity
            print("TODO")
            #pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
            #            save_path)


## All plots


if analysis == "ps_pc_corr":
    hf.plot_ps_pc(results["ps"], results["pc"], save_path)

    elif analysis == "ps_corr":
        all_ps_corr = np.concatenate(results["ps_corr"])
        hf.plot_metric_typ_cum(all_ps_corr, all_types_cat, colors, 
                 ps_corr_edges, "all exp", save_path)
        
        hf.get_t_significance(all_ps_corr, all_types_cat)
    
    elif analysis == "ps_ev":
        all_fr_ps_cat = np.concatenate(results["fr_ps"])
        np.save(os.path.join(save_path,"fr_ps.npy"), all_fr_ps_cat) 
        
        embedding = np.load(os.path.join(save_path,"fr_ps_umap.npy"))
        
        #emb_p = np.array([[4,-7], [8,-5], [10,-7], [13,-3]]) # w,n,s,e
        #mean_emb_fr, mean_emb_c = hf.get_mean_fr_2d(z_fr_ps_slow, embedding, 
        #                                            emb_p, c_types)
        #hf.plot_fr_aligned(tw, mean_emb_fr, mean_emb_c)
        
        hf.plot_umap(embedding, c_types_all, save_path) # emb_p, mean_emb_c

"""
indices = [355]#[8,41,48,87,74,229,230,273,385,407,355]


def filter_mutual_info(mi, win_m = 3, thresh = 0.1,  f_type="median"):
    filtered_mi = np.zeros_like(mi)
    N, T = mi.shape
    
    if f_type == "median":
        for t in range(T):
            filtered_mi[:,t] = np.median(mi[:,t-win_m:t+win_m],1)
    elif f_type == "thresh":
        
        for n in range(N):
            high_mi = mi[n,:] > thresh
            for t in range(T):
                if t < T-2 and high_mi[t] and high_mi[t+1] and high_mi[t+2]:
                    for ti in range(3):
                        filtered_mi[n,t+ti] = mi[n,t+ti]
        
    return filtered_mi

def find_double_coders(tw, sac_dir, tww = [-0.4,0.4]):
    double_n = []
    for n in range(sac_dir.shape[0]):
        t_mask = (tw > tww[0]) & (tw <= tww[1])
        if (0 in sac_dir[n, t_mask]) and (1 in sac_dir[n, t_mask]):
            double_n.append(n)
    
    return double_n


# plot coding and connectivity

n = np.bincount(connected_pairs[:,1]).argmax()
hf.plot_coding(n, tw, sac_dir, cluster_type, connected_pairs, 
               colors, sac_colors)

#
def plot_pref_sc_conn(connected_pairs, pref_sc, rts_sc, cluster_type,
                      save_path, exp, nc=[1,1], pre_post=["TCA","NW"]):
    cluster_type = np.asarray(cluster_type)
    
    responsive_mask = (~np.isnan(rts_sc[0,:])) | (~np.isnan(rts_sc[0,:]))
    pre_mask = (cluster_type == pre_post[0]) & responsive_mask 
    post_mask = (cluster_type == pre_post[1]) & responsive_mask
    
    pre_post_mask = (pre_mask[connected_pairs[:,0]] &
                     post_mask[connected_pairs[:,1]])
    pre_post_pair = connected_pairs[pre_post_mask,:]
    
    x = pref_sc[pre_post_pair[:,0]]
    y =pref_sc[pre_post_pair[:,1]]
    r = np.corrcoef(x, y)[0,1]
    
    plt.hist2d(x, y,bins=2, range=[[0, 1], [0, 1]], cmap='Blues')
    plt.text(0.6,0.1, "r_coef = " + str(np.round(r,2)))
    plt.colorbar(label='Count')
    plt.xticks([0, 1], labels=["temp","nasal"])
    plt.yticks([0, 1], labels=["temp","nasal"])
    plt.xlabel("presyn. " + pre_post[0])
    plt.ylabel("postsyn. " + pre_post[1])

    plt.show()
    
all_pref_sc = np.concatenate(results["pref_sc"], axis = 0)
all_rts_sc = np.concatenate(results["rts_sc"], axis = 0)
plot_pref_sc_conn(connected_pairs, all_pref_sc, all_rts_sc, cluster_type,
                  save_path, exp, pre_post=["TCA","NW"])

def check_mb_sac(saccades,mov_bar,sync_cam, sac_colors):
    y = 1
    for si,s in enumerate(saccades):
        sac = np.array(saccades[s])
        sac_times = (sync_cam[sac] - sync_cam[0]) / 60
        plt.vlines(sac_times, y + 0.05 + si*0.05, y + 0.1 + si*0.05, 
                   colors=sac_colors[si], linewidth=0.5)
    y = 1.3 
    for si,s in enumerate(mov_bar):
        sac = np.array(mov_bar[s])
        sac_times = (sync_cam[sac] - sync_cam[0]) / 60
        plt.vlines(sac_times, y + 0.05 + si*0.05, y + 0.1 + si*0.05, 
                   colors=sac_colors[si], linewidth=0.5)
    plt.show()

"""

# Helper Functions

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

def get_mean_fr_size(fr, state,  start = 0.1, stop = 0.3, step = 0.02, 
                     per=[5,95]):
    """
    Calculates the mean and std of the firng rate for a given state.
    
    Parameters:
    - fr: np.ndarray, shape (n_neu, T)
    - state: np.ndarray, shape (T)
    - start, stop, step: float, edges for the bining of the state vector
    - per: list of two ints
    
    Returns:
    - stats: np.ndarray, shape (n_neu, n_bins, 3) # mean,upper/lower percentil 
    - s_bins: np.ndarray, shape (n_bins)
    """
    
    s_edges = np.arange(start, stop, step)
    s_bins = (s_edges[:-1] + s_edges[1:]) / 2
    
    bin_indx = np.digitize(state, s_edges) - 1
    valid_mask = (bin_indx >= 0) & (bin_indx < len(s_bins))
    
    stats = np.empty((fr.shape[0], len(s_bins), 3))
    
    for b in range(len(s_bins)):
        in_bin = (bin_indx == b) & valid_mask
        if np.any(in_bin):
            stats[:, b, 0] = np.mean(fr[:, in_bin], axis=1)
            stats[:, b, 1] = np.percentile(fr[:, in_bin], per[0], axis=1)
            stats[:, b, 2] = np.percentile(fr[:, in_bin], per[1], axis=1)
    
    return stats, s_bins

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
    #ax.set_ylim([0, 0.5])
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

def plot_ps_exp(stats_fr, s_bins, colors, cluster_type, n, sp, 
                ylim=[-1,3]):

    median = stats_fr[n,:,0]
    p5 = stats_fr[n,:,1]
    p95 = stats_fr[n,:,2]
    c = colors[cluster_type[n]]
    
    plt.plot(s_bins, median, alpha=0.2, color=c, marker="o")
    #plt.fill_between(s_bins, p5, p95, alpha=0.3,
    #                 facecolor=c)
    #plt.ylim(ylim)            
    plt.xlabel("pupil size")
    plt.ylabel("z-scored fr")

    for s in ['right', 'top']:
        plt.gca().spines[s].set_visible(False)
    plt.savefig(os.path.join(sp,"plots", str(n) + "ps_fr.svg"))
    plt.show()

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

def plot_proj(tw, proj, delta_x, sac_colors):
    #(t,e)
    for s in range(proj.shape[1]):
        sign = int((np.sign(delta_x[s]) + 1) / 2)
        strenght = np.abs(delta_x[s]) / np.max(np.abs(delta_x))
        plt.plot(tw, proj[:,s], color=sac_colors[sign], alpha=strenght)
    plt.xlabel(" time [s]")
    plt.ylabel("proj. fr")

    for s in ['right', 'top']:
        plt.gca().spines[s].set_visible(False)
    #plt.savefig(os.path.join(sp,"plots", str(n) + "ps_fr.svg"))

    plt.show()









