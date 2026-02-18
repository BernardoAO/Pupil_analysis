## Neural pupila coding
### Bernardo AO
import os 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import Helper_functions as hf
assert False

# TODO saccades per stimuli

def ps_analysis(z_fr, pupil_size, cluster_type, colors, 
                ps_corr_edges, plot_name, save_path, plot="none"):
        
    if isinstance(plot, int):

        # Mean fr vs pupil size
        stats_fr, s_bins = \
            hf.get_mean_fr_size(z_fr, pupil_size, per=[20,80])
            
        hf.plot_ps_exp(stats_fr, s_bins, colors, cluster_type, plot, save_path)
        
    # fr pupil size correlation
    neu_pupil_corr = hf.get_correlation(z_fr, pupil_size)
    
    if plot == "hist":
        hf.plot_correlation_hist(neu_pupil_corr, colors, cluster_type, ps_corr_edges, 
                                plot_name, save_path)
    elif plot == "cum":
        hf.plot_metric_typ_cum(neu_pupil_corr, cluster_type, colors, ps_corr_edges, 
                                plot_name, save_path)
    
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

def saccade_analysis(saccades, pupil_center, firing_rate, valid_spiketimes, 
                     sync_cam, c_types, save_path, cluster_type, colors,
                     win = [-0.25,1], nc=10, an_type="PCA",plot="none"):    
    
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    msc_colors = ["navy" if sc < len(saccades["temporal"]) else 
                  "violet" for sc in range(len(saccades_all))]
    
    # Get saccade align fr
    trial_fr_t, fr_sc_t, tw = hf.get_fr_aligned(firing_rate, 
                                                saccades["temporal"], win=win)    
    trial_fr_n, fr_sc_n, tw = hf.get_fr_aligned(firing_rate, 
                                                saccades["nasal"], win=win)    
    trial_fr = [trial_fr_t, trial_fr_n]
    fr_sc = np.stack((fr_sc_t, fr_sc_n), axis=-1)
    
    if plot == "all" or plot == "pupil":
        hf.plot_event(saccades, pupil_center[0,:], "x coordinate", exp, save_path)
    if plot == "all" or plot == "raster":
        hf.plot_raster(valid_spiketimes, sync_cam, saccades_all, msc_colors,
                    tw, fr_sc, c_types, save_path, name="fr_man_sac.svg")
    
    if an_type == "RT":
        # Preferred direction
        max_fr = np.max(fr_sc, axis=1)
        pref_sc = np.argmax(max_fr, axis=1)
        
        # Response times
        rts_sc_t = hf.get_response_times(firing_rate, saccades["temporal"], thres=0.15)
        rts_sc_n = hf.get_response_times(firing_rate, saccades["nasal"], thres=0.15)
        rts_sc = [np.where(pref_sc == 0, rts_sc_t, rts_sc_n), 
                  np.where(pref_sc == 1, rts_sc_t, rts_sc_n)]
        
        return tw, fr_sc, rts_sc
        
    elif an_type == "PCA":
        pca_pc, exp_var, _ = hf.neuron_PCA(fr_sc, cluster_type, n_components=nc)
        exp_var_n = hf.noise_PCA(fr_sc, trial_fr, cluster_type, n_components=nc)
        if plot == "all" or plot == "pca":
            hf.plot_pca(tw, pca_pc[:,:3,:,:], colors, save_path)
            hf.plot_pca_var(exp_var, exp_var_n, colors, save_path, exp)
        
        return tw, fr_sc, [exp_var, exp_var_n]

#def main():
    
# data file names
pupil_data_path = r"D:\NP data\Bernardo_awake_cx\Results\pupil_data"
spike_bundle_path = r"D:\NP data\analysis\data-single-unit"
save_path = r"D:\NP data\Bernardo_awake_cx\Results"

# Session information
fs = 30000 # Hz
camara_fs = 200 # Hz
colors =  {"TCA":"orchid", "NW":"salmon", "BW":"black"} 

# Parameters
analysis = "sac_PCA" # exp, ps_pc_corr, ps_corr, ps_ev, pc_sim, sac_RT
period =  "all" # "chirp"
ps_corr_edges = np.arange(-0.3,0.32,0.010)
units_for_plot = [] # [357(1),368,404] #ps #22 ps_exp 
                    # [30,176,355] #pc 

experiments = [exp[11:-4] for exp in os.listdir(pupil_data_path)]
experiments.sort()

#experiments = [experiments[1]]

results = defaultdict(list)

# file loop
for exp in tqdm(experiments, desc="Files processed"):
    
    plot_name = exp + "_" + period
    
    ## Import
    
    # spike data
    Spke_Bundle, spiketimes, SIN_data = \
        hf.import_spike_data(exp, spike_bundle_path)
    vis_stim, stim_colors = hf.get_stims(Spke_Bundle)

    # merge pupil data for the exp
    sync_cam, pupil_size, pupil_center, saccades = \
        hf.import_pupil_data(pupil_data_path, Spke_Bundle, exp, period)
    
    
    if analysis == "exp":
        hf.plot_exp(Spke_Bundle, sync_cam, vis_stim, stim_colors, exp, save_path)
        hf.plot_pupil_stimuli(pupil_size, pupil_center, sync_cam, 
                              Spke_Bundle["events"], vis_stim, stim_colors, 
                              exp, save_path)
    
    elif analysis == "ps_pc_corr":
        results["ps"].append(pupil_size)
        results["pc"].append(pupil_center)
        
    else:
        # get valid clusters
        valid_spiketimes, cluster_type, c_types = \
            hf.get_valid_cluster(Spke_Bundle, SIN_data, spiketimes, colors, 
                                 units_for_plot)            
        results["types"].append(cluster_type)
        
        ## Firing rate
        
        tqdm.write("Firing rate...")
        firing_rate, z_fr = hf.get_firing_rate(valid_spiketimes, sync_cam)
        tqdm.write(analysis + " analysis...")
         
        ## Pupil size
        
        if analysis == "ps_corr": # correlation
            neu_pupil_corr = ps_analysis(z_fr, pupil_size, cluster_type, colors, 
                                         ps_corr_edges, plot_name, save_path, 1)
            results["ps_corr"].append(neu_pupil_corr)
        
        elif analysis == "ps_ev": # size change events
            z_fr_ps = ps_events_analysis(pupil_size, firing_rate, valid_spiketimes, sync_cam, 
                                         c_types, exp, save_path, plot = "raster")
            results["fr_ps"].append(z_fr_ps)
        
        ## Pupil center
        
        elif analysis == "pc_sim": # similarity
            pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
                        save_path)
        
        elif analysis == "sac_PCA": # saccades                
            tw, fr_sc, PCA_var = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors)
                
            results["fr_sc"].append(fr_sc)
            results["PCA_var"].append(PCA_var)
        
        elif analysis == "sac_RT": # saccades                
            tw, fr_sc, rts_sc = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, an_type="RT")
                
            results["fr_sc"].append(fr_sc)
            results["rts_sc"].append(rts_sc)
            

## All plots

if analysis == "ps_pc_corr":
    hf.plot_ps_pc(results["ps"], results["pc"], save_path)

else:    
    all_types_cat = [x for exp in results["types"] for x in exp]
    c_types_all = np.array([colors[n] for n in all_types_cat])
    
    
    if analysis == "exp": # n 
        hf.plot_types(experiments, results["types"], colors, save_path)
    
    
    elif analysis == "ps_corr":
        all_ps_corr = np.concatenate(results["ps_corr"])
        hf.plot_metric_typ_cum(all_ps_corr, all_types_cat, colors, 
                 ps_corr_edges, "all exp", save_path)
        
        hf.get_t_significance(all_ps_corr, all_types_cat)
    
    
    elif analysis == "ps_ev":
        all_fr_ps_cat = np.concatenate(results["fr_ps"])
        np.save(os.path.join(save_path,"fr_ps.npy"), all_fr_ps_cat) 
        
        embedding = np.load(os.path.join(save_path,"fr_ps_umap.npy"))
        
        hf.plot_umap(embedding, c_types_all, save_path) # emb_p, mean_emb_c
        
        #emb_p = np.array([[4,-7], [8,-5], [10,-7], [13,-3]]) # w,n,s,e
        #mean_emb_fr, mean_emb_c = hf.get_mean_fr_2d(z_fr_ps_slow, embedding, 
        #                                            emb_p, c_types)
        #hf.plot_fr_aligned(tw, mean_emb_fr, mean_emb_c)
        
    elif analysis == "sac_PCA":
        # projection
        all_fr_sc_cat = np.concatenate(results["fr_sc"], axis = 0)
        pca_pc, _, w = hf.neuron_PCA(all_fr_sc_cat, c_types_all, n_components=10)
        hf.plot_pca(tw, pca_pc[:,:3,:,:], colors, save_path)
        
        # variance
        exp_var = np.stack([var[0] for var in results["PCA_var"]], axis=2)
        exp_var_n = np.stack([var[1] for var in results["PCA_var"]], axis=2)
        hf.plot_pca_var(exp_var, exp_var_n, colors, save_path, "all")
        
        # weights
        hf.plot_weights(w, colors, save_path)    

    elif analysis == "sac_RT":
        edges = np.arange(-0.2, 1, 0.01)
        for i in range(2):
            all_rts_sc_i = np.concatenate([rt[i] for rt in results["rts_sc"]])
            hf.plot_metric_typ_cum(all_rts_sc_i, all_types_cat, colors, edges, 
                                   "rts_"+ str(i), save_path)

#if __name__ == "__main__":
#    main()




