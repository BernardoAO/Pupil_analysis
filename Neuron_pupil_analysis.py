## Neural pupil coding
### Bernardo AO

import os 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import Helper_functions as hf
#assert False

# TODO saccades per stimuli

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

def sac_amp_analysis(saccades, pupil_center, valid_spiketimes, sync_cam, 
                     firing_rate, save_path, cluster_type, colors, exp, 
                     win=[-0.25,1], plot="none", dx = 0.05, n_plot = [],
                     m_names=["x", "|x|", "sign(x)"], 
                     sac_colors = ["navy", "darkorange"]):
    
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    
    delta_x, delta_fr = hf.get_sac_amp(valid_spiketimes, sync_cam, 
                                       saccades_all, pupil_center[0,:])
    
    
    models, sig_ws = hf.lin_model_sac(delta_x, delta_fr, m_names, sig=False)
        
    if plot == "m_w":
        m = np.mean(delta_fr,axis=1)
        w = models[m_names[0]][:,0]
        hf.plot_type_scatter(m, w, colors, cluster_type, save_path, exp)
        
    if plot == "hist": 
        edges = np.arange(-1, 1+dx, dx)
        for m_name, m  in models.items():
            hf.plot_hist_typ(m[:,0], cluster_type, colors, edges, save_path,
                             exp, "sac_amp", xlabel=m_name, cum=False)
    
    if plot == "hist_sig":
        edges = np.arange(0, 0.5+dx, dx)
        
        for i in range(len(m_names)):
            hf.plot_hist_typ(sig_ws[:,i], cluster_type, colors, edges, save_path,
                             exp, "sac_amp", xlabel=m_names[i], cum=False)
    
    if plot == "example":
        hf.plot_sac_amp_ex(delta_x, delta_fr, models["x"], n_plot, cluster_type, 
                        colors, save_path, exp)
    
    if plot == "diagram":
        hf.plot_sac_amp_diagram(delta_x, save_path)
        
    if plot == "proj":
        trial_fr, _, tw = hf.get_fr_aligned(firing_rate, saccades_all, win=win)    
        proj = np.tensordot(models["x"][:, 0], trial_fr, axes=(0, 0))
        hf.plot_proj(tw, proj, delta_x, sac_colors)

    return delta_fr, models, sig_ws

def pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
                save_path, center_edges = np.arange(105, 145, 5)):
    
    mean_fr_center = hf.get_mean_fr_center(firing_rate, pupil_center, center_edges)

    similarity_type = hf.get_similarity(mean_fr_center, cluster_type, colors)

    plot_bin = 19
    hf.plot_similarity_2d(similarity_type, plot_bin, 
                          center_edges, plot_name, save_path,clim=[-1,1])

def saccade_analysis(saccades, pupil_center, firing_rate, valid_spiketimes, 
                     sync_cam, c_types, save_path, cluster_type, colors, exp,
                     win = [-0.25,1], nc=10, an_type="PCA", plot="none",
                     sac_colors = ["navy", "darkorange"]):    
    
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    msc_colors = [sac_colors[0] if sc < len(saccades["temporal"]) else 
                  sac_colors[1] for sc in range(len(saccades_all))]
    
    # Get saccade align fr
    trial_fr_t, fr_sc_t, tw = hf.get_fr_aligned(firing_rate, 
                                                saccades["temporal"], win=win)    
    trial_fr_n, fr_sc_n, tw = hf.get_fr_aligned(firing_rate, 
                                                saccades["nasal"], win=win)    
    trial_fr = [trial_fr_t, trial_fr_n]
    fr_sc = np.stack((fr_sc_t, fr_sc_n), axis=-1)

    if plot == "raster":
        plts_sp = os.path.join(save_path, "plots", "Neurons", "sac_dir")
        hf.plot_raster(valid_spiketimes, sync_cam, saccades_all, sac_colors, 
                       msc_colors, tw, fr_sc, c_types, cluster_type, 
                       sp = plts_sp, name="sac.svg")

    if an_type[:2] == "RT":
        
        if an_type == "RT_dir":
            # Response times
            rts_sc_t = hf.get_response_times(firing_rate, saccades["temporal"], p=0.01, permute=False)
            rts_sc_n = hf.get_response_times(firing_rate, saccades["nasal"], p=0.01, permute=False)
            
            # Preferred direction
            max_fr = np.max(fr_sc, axis=1)
            pref_sc = np.argmax(max_fr, axis=1)
            
            rts_sc = np.array([np.where(pref_sc[:, None] == 0, rts_sc_t, rts_sc_n),
                               np.where(pref_sc[:, None] == 1, rts_sc_t, rts_sc_n)])
            # [(pref.,nonp.),n, (rt,dir)]
            
        elif an_type == "RT_all":
            # Response times
            rts_sc = hf.get_response_times(firing_rate, saccades_all, p=0.01, permute=False)
            pref_sc = np.zeros((fr_sc.shape[0]))
            rts_sc = np.array([rts_sc,rts_sc])
            
        if plot == "raster_RT":
            plts_sp = os.path.join(save_path, "plots", "Neurons", "reaction time")
            hf.plot_raster(valid_spiketimes, sync_cam, saccades_all, sac_colors, 
                           msc_colors, tw, fr_sc, c_types, cluster_type, 
                           rts=[rts_sc_t, rts_sc_n], sp = plts_sp, 
                           name = exp + "_sac.svg")
        
        elif plot == "hist":
            rt_edges = np.arange(-0.2, 0.5, 0.02)
            hf.plot_sc_hist(rts_sc, c_types, rt_edges, save_path, exp)
            
        elif plot == "scat":
            hf.plot_sc_scat(rts_sc, c_types, save_path, exp)
            
        return tw, fr_sc, rts_sc, pref_sc
    
    elif an_type == "MI":
        
        s = np.array([0 if s == sac_colors[0] else 1 for s in msc_colors])
        mutual_info_raw = hf.get_MI(trial_fr, s)
        

        
        mutual_info = hf.get_sig_MI(mutual_info_raw, tw)
        
        if plot == "MI":
            hf.plot_mean_mi(tw, mutual_info, cluster_type, colors, save_path, exp)

        
        return tw, mutual_info
        
    elif an_type == "dir":
        sac_dir = hf.get_class_coding(firing_rate, saccades["temporal"], 
                                saccades["nasal"], win=win)
        
        if plot == "raster_dir":
            plts_sp = os.path.join(save_path, "plots", "Neurons", "coding")
            hf.plot_raster(valid_spiketimes, sync_cam, saccades_all, sac_colors, 
                           msc_colors, tw, fr_sc, c_types, cluster_type,
                           coding = sac_dir, sp = plts_sp, name = exp + "_sac.png")
        elif plot == "nratio":
            hf.plot_nratio_code(sac_dir, cluster_type, colors, tw, save_path, exp)
            
        return tw, fr_sc, sac_dir
    
        
    elif an_type == "PCA":
        pca_results = hf.neuron_PCA(fr_sc, cluster_type, n_components=nc)
        exp_var_n = hf.noise_PCA(fr_sc, trial_fr, cluster_type, n_components=nc)
        
        if plot == "pca":
            hf.plot_pca(tw, pca_results, colors, sac_colors, save_path, exp)
            hf.plot_pca_var(pca_results, exp_var_n, colors, save_path, exp)
                
        return tw, fr_sc, pca_results, exp_var_n

#def main():
    
# data file names
pupil_data_path = r"D:\NP data\Bernardo_awake_cx\Results\pupil_data\right_eye"
spike_bundle_path = r"D:\NP data\analysis\data-single-unit"
save_path = r"D:\NP data\Bernardo_awake_cx\Results"

# Session information
fs = 30000 # Hz
camara_fs = 200 # Hz
colors =  {"TCA":"orchid", "NW":"salmon", "BW":"black"} 
sac_colors = ["navy", "darkorange"]

# Parameters 
analysis = "sac_RT" # exp, exp_neu, ps_pc_corr, sac_vis
                     # ps_corr, pc_corr, ps_ev, pc_sim, 
                     # sac_amp; sac_RT, sac_MI, sac_dir, sac_PCA
period =  "all" # "chirp"

fr_win_name = "_100ms_causal.npy" #40c, 100c
fr_win = [-0.1, 0.] #[-0.04, 0.], [-0.1, 0.]

ps_corr_edges = np.arange(-0.3, 0.32, 0.01)
pc_corr_edges = np.arange(0., 0.2, 0.005)

save_rts = False

units_for_plot = [] # [357(1),368,404] #SCE #22 ps_corr 
                    # [30,70,108] [nw,tca,bw] sac_RT, 128,407
                    # [8,355,379] 412 [nw,bw,tca] sac_dir
                    # double_sac [135,146,297,407]
pre_load = False if units_for_plot else True
    
experiments = [exp[11:-4] for exp in os.listdir(pupil_data_path)]
experiments.sort()

#experiments = ["2022-12-20_15-08-10"] 
# ["2022-12-20_15-08-10"] #,"2023-03-16_12-16-07","2023-04-18_12-10-34"]

results = defaultdict(list)

# file loop
for exp in tqdm(experiments, desc="Files processed"):
    
    ## Import
    
    # spike data
    Spke_Bundle, spiketimes, SIN_data, connected_pairs_all = \
        hf.import_spike_data(exp, spike_bundle_path)
        
        
    exp_pd_path =  os.path.join(pupil_data_path, "pupil_data_" + exp + ".pkl")

    # merge pupil data for the exp
    sync_cam, pupil_size, pupil_center, saccades = \
        hf.import_pupil_data(pupil_data_path, Spke_Bundle, exp, period)
    
    # stimulus
    vis_stim, stim_colors, mov_bar = hf.get_stims(Spke_Bundle)

    if analysis == "exp":
        # Stimuli
        hf.plot_exp(Spke_Bundle, sync_cam, vis_stim, stim_colors, exp, 
                    save_path, saccades, sac_colors)
        # Pupil size
        hf.plot_pupil_stimuli(pupil_size, pupil_center, sync_cam, 
                              Spke_Bundle["events"], vis_stim, stim_colors, 
                              exp, save_path)
        
        # Saccades
        hf.plot_sac_trayectory(saccades, pupil_center, sac_colors, 
                               save_path, exp)
        hf.plot_event(saccades, pupil_center[0,:], sac_colors, 
                      "x coordinate", exp, save_path)
        
        hf.plot_saccades_2d(saccades, pupil_center, sac_colors, exp, save_path)
        
        sac_var = hf.saccade_variance(pupil_center, saccades)
        results["sac_var"].append(sac_var)
    
    elif analysis == "sac_vis":
        
        off_sets = [0,-0.1]
        screen_t = []
        for off_set in off_sets:
            sac_vis = hf.get_sac_vis(saccades, sync_cam, vis_stim, Spke_Bundle,
                                     off_set=off_set)
            sac_vis = hf.get_screen(sac_vis, Spke_Bundle, vis_stim)
    
            dif_screen = hf.get_screen_dir(sac_vis)
            title_name = "offset of " + str(off_set) + "s"
            hf.plot_mean_screen(dif_screen, title_name)

            #for n in range(len(sac_vis["screen"])):
            #    if sac_vis["screen"][n].size > 1:
            hf.plot_screen_ex(sac_vis, save_path, n=30)
            screen_t.append(screen_t)
            
        results["dif_screen"].append(screen_t)
    
    elif analysis == "ps_pc_corr":
        results["ps"].append(pupil_size)
        results["pc"].append(pupil_center)
        
    else:
        # get valid clusters
        valid_spiketimes, cluster_type, c_types, connected_pairs = \
            hf.get_valid_cluster(Spke_Bundle, SIN_data, spiketimes,
                                 connected_pairs_all, colors, units_for_plot)
            
        results["types"].append(cluster_type)
        results["connected_pairs"].append(connected_pairs)

        ## Firing rate
        
        tqdm.write("Firing rate...")
        spk_count = os.path.join(save_path, "spk_count", exp + fr_win_name)
        firing_rate, z_fr = hf.get_firing_rate(valid_spiketimes, sync_cam, 
                                               spk_count, win=fr_win,
                                               pre_load=pre_load)
        tqdm.write(analysis + " analysis...")
        
        #fr_all[exp] = firing_rate
        
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
        
        elif analysis == "exp_neu":
            hf.plot_conn(connected_pairs, cluster_type, colors, save_path, exp)
        
        elif analysis == "ps_ev": # size change events
            z_fr_ps = ps_events_analysis(pupil_size, firing_rate, valid_spiketimes, sync_cam, 
                                         c_types, exp, save_path, plot = "raster")
            results["fr_ps"].append(z_fr_ps)
        
        ## Pupil center
        
        elif analysis == "pc_sim": # similarity
            print("TODO")
            #pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
            #            save_path)
        
        elif analysis == "sac_amp": # saccades
            delta_fr, models, sig_ws = sac_amp_analysis(
                saccades, pupil_center, valid_spiketimes, sync_cam, firing_rate,
                save_path, cluster_type, colors, exp, 
                plot="example", dx = 0.01, n_plot = [8,355,379])
            
            results["delta_fr"].append(delta_fr)
            results["models"].append(models)
            results["sig_ws"].append(sig_ws)
            
            sac_mb = False
            if sac_mb:
                w_sac = models["|x|"][:,0]
                m_sac = np.mean(delta_fr, axis=1)
                
                delta_x, delta_fr = hf.get_delta_fr(valid_spiketimes, sync_cam, mov_bar)
                m_mb = np.mean(delta_fr, axis=1)

                m_names=["x", "|x|", "sign(x)"]
                models, sig_ws = hf.lin_model_sac(delta_x, delta_fr, m_names, sig=False)
                w_mb = models["|x|"][:,0]
                
                hf.plot_type_scatter(w_mb, w_sac, colors, cluster_type, save_path, 
                                     exp + "all_mb_sac", corr=True, xlabel="w_mb",ylabel="w_sac")
        
            
        elif analysis == "sac_RT":
            tw, fr_sc, rts_sc, pref_sc = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="RT_dir", plot="scat")
                
            results["fr_sc"].append(fr_sc)
            results["rts_sc"].append(rts_sc) 
            results["pref_sc"].append(pref_sc)
            
            if save_rts:
                np.save(os.path.join(save_path, "rts", exp), rts_sc)
                
        elif analysis == "sac_MI":        
            tw, mutual_info = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="MI", plot="MI", win = [-0.5,1])
                
            results["mutual_info"].append(mutual_info) 
            
            if save_rts:
                rts_sc = np.load(os.path.join(save_path, "rts", exp+ ".npy"))
                results["rts_sc"].append(rts_sc) 
            
                    
        elif analysis == "sac_dir":                
            tw, fr_sc, sac_dir = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="dir", plot="nratio")
            results["fr_sc"].append(fr_sc)
            results["sac_dir"].append(sac_dir) 
        
        elif analysis == "sac_PCA":              
            tw, fr_sc, pca_results, exp_var_n = \
                saccade_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp)
                
            results["fr_sc"].append(fr_sc)
            results["PCA_var"].append([pca_results, exp_var_n])
        

## All plots

if analysis == "ps_pc_corr":
    hf.plot_ps_pc(results["ps"], results["pc"], save_path)
    
elif analysis == "exp":
    hf.plot_sac_var(results["sac_var"], save_path)

else:    
    all_types_cat = [x for exp in results["types"] for x in exp]
    c_types_all = np.array([colors[n] for n in all_types_cat])
    
    
    if analysis == "exp_neu": # n 
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
        
        #emb_p = np.array([[4,-7], [8,-5], [10,-7], [13,-3]]) # w,n,s,e
        #mean_emb_fr, mean_emb_c = hf.get_mean_fr_2d(z_fr_ps_slow, embedding, 
        #                                            emb_p, c_types)
        #hf.plot_fr_aligned(tw, mean_emb_fr, mean_emb_c)
        
        hf.plot_umap(embedding, c_types_all, save_path) # emb_p, mean_emb_c
        
    elif analysis == "sac_amp":
        delta_fr_all = np.concatenate(
            [np.mean(df, axis=1) for df in results["delta_fr"]])
        lin_ws_all = np.concatenate(
            [m["x"][:,0] for m in results["models"]])
        sig_ws_all = np.concatenate(
            [sig_ws for sig_ws in results["sig_ws"]], axis=0)
        
        hf.plot_sig_ws(sig_ws_all, all_types_cat, colors, save_path, "all")
        hf.plot_type_scatter(delta_fr_all, lin_ws_all, colors, all_types_cat, save_path)
        
        edges_m = np.arange(-10,50,0.5)
        hf.plot_hist_typ(delta_fr_all, all_types_cat, colors, edges_m, 
                          save_path, "all", "delta_fr", cum=False, xlabel="Δfr")
        
        edges_w = np.arange(-1.5,1.5,0.05)
        hf.plot_hist_typ(lin_ws_all, all_types_cat, colors, edges_w, 
                          save_path, "all", "w", cum=False, xlabel="w")

        
    elif analysis == "sac_RT":
        rt_edges = np.arange(-0.2, 0.5, 0.02)
        # rts_sc pre,n,sign
        rts_sc_all = np.concatenate([rts for rts in results["rts_sc"]], axis=1)
        
        hf.plot_sc_hist(rts_sc_all, c_types_all, rt_edges, save_path)
        hf.plot_sc_scat(rts_sc_all, c_types_all, save_path)
        
        rts_sc_pr = [rt[0,:,0] for rt in results["rts_sc"]]
        hf.plot_all_rt(rts_sc_pr, results["types"], colors, save_path)
    
    elif analysis == "sac_MI":
        
        # RT mask
        #rts_sc_all = np.concatenate([rts for rts in results["rts_sc"]], axis=1)
        #rt_mask = ~np.isnan(rts_sc_all[0,:,0]) | ~np.isnan(rts_sc_all[1,:,0])
        
        mutual_info_all = np.concatenate([
            mi for mi in results["mutual_info"]], axis = 0)
        #mutual_info_all = np.concatenate([mi for i, mi in enumerate(results["mutual_info"]) if i != 2], axis=0)
        #  all_types_cat = [x for i, exp in enumerate(results["types"]) if i != 2 for x in exp]
        hf.plot_mean_mi(tw, mutual_info_all, all_types_cat, colors, save_path, "all_3")

    
    elif analysis == "sac_dir":
        all_sac_dir = np.concatenate(results["sac_dir"], axis = 0)
        hf.plot_nratio_code(all_sac_dir, all_types_cat, colors, tw, 
                            save_path, "all")
        
    elif analysis == "sac_PCA":
        # projection
        all_fr_sc_cat = np.concatenate(results["fr_sc"], axis = 0)
        
        pca_results = hf.neuron_PCA(all_fr_sc_cat, all_types_cat)
        
        
        analysis2 = "" #sac_amp"
        if analysis2 == "conn":
            hf.plot_weights_conn(connected_pairs, pca_results, cluster_type,
                                 save_path, exp, nc=[1,1], pre_post=["TCA","NW"])
        
        elif analysis2 == "sac_amp":
            hf.plot_cs_pc_w(pca_results, lin_ws_all, all_types_cat, 
                          "amp vec", save_path)
            hf.plot_cs_pc_w(pca_results, delta_fr_all, all_types_cat, 
                          "resp vec", save_path)
            hf.plot_cs_ws(lin_ws_all, delta_fr_all, all_types_cat, colors,
                          "amp resp", save_path)
            
        elif analysis2 == "all":
            pca_results_all = hf.all_PCA(all_fr_sc_cat)
            hf.plot_type_scatter(pca_results_all["w"][0,:], pca_results_all["w"][1,:], 
                        colors, all_types_cat, save_path, name="PCA_wall")

            
        else:
            hf.plot_pca(tw, pca_results, colors, sac_colors, save_path)
            hf.plot_multi_pca(tw, pca_results, colors, sac_colors, save_path)
            
            # variance
            pca_results_list = [var[0] for var in results["PCA_var"]]
            exp_var_n_list = [var[1] for var in results["PCA_var"]]
            
            sig_nc = hf.pca_var_sig(pca_results_list, exp_var_n_list)
            hf.plot_pca_var(pca_results_list, exp_var_n_list, 
                            colors, save_path, "all", sig_nc)
            
            # weights
            hf.plot_weights(pca_results, colors, save_path, scatter=True)
            



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





#if __name__ == "__main__":
#    main()
