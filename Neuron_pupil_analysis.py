## Neural pupil coding
### Bernardo AO

import os 
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import Helper_functions as hf
#assert False

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
        
    return delta_fr, models, sig_ws

def sac_res_analysis(saccades, firing_rate, valid_spiketimes, sync_cam, 
                     c_types, save_path, cluster_type, colors, exp,
                     plot_win = [-0.3,0.6], rt_win=[-0.5,-0.2, 0.5], 
                     dt=0.02, plot="none"):    
    
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    
    # Get saccade align fr
    _, fr_sc, tw = hf.get_fr_aligned(firing_rate, saccades_all, win=plot_win)    

    # Response times
    rts_sc = hf.get_response_times(firing_rate, saccades_all,
                                   win=rt_win, p=0.01, permute=True)
        
    if plot == "raster_RT":
        plts_sp = os.path.join(save_path, "plots", "Neurons", "reaction time")
        hf.plot_raster(valid_spiketimes, fr_sc, sync_cam, saccades_all,
                       tw, c_types, cluster_type, rts=rts_sc, sp = plts_sp, 
                       name = exp + "_sac.svg")

    elif plot == "hist":
        rt_edges = np.arange(plot_win[0], plot_win[1], dt)
        hf.plot_sc_hist(rts_sc, c_types, rt_edges, save_path, exp)
        
        
    return tw, fr_sc, rts_sc

def sac_dir_analysis(saccades, pupil_center, firing_rate, valid_spiketimes, 
                     sync_cam, c_types, save_path, cluster_type, colors, exp,
                     win = [-0.3,0.6], nc=10, an_type="PCA", plot="none",
                     sac_colors = ["navy", "darkorange"]):    
    
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    msc_colors = [sac_colors[0] if sc < len(saccades["temporal"]) else 
                  sac_colors[1] for sc in range(len(saccades_all))]
    
    # Get saccade align fr
    trial_fr_t, fr_sc_t, _ = hf.get_fr_aligned(
        firing_rate, saccades["temporal"], win=win)    
    trial_fr_n, fr_sc_n, tw = hf.get_fr_aligned(
        firing_rate, saccades["nasal"], win=win)
    
    trial_fr = [trial_fr_t, trial_fr_n]
    fr_sc = np.stack((fr_sc_t, fr_sc_n), axis=-1)

    if plot == "raster":
        plts_sp = os.path.join(save_path, "plots", "Neurons", "sac_dir")
        hf.plot_raster(valid_spiketimes, fr_sc, sync_cam, saccades_all, tw, 
                       c_types, cluster_type, sac_colors, msc_colors,  
                       sp = plts_sp, name="sac.svg")

    if an_type == "RT":
        
        # Response times
        rts_sc_t = hf.get_response_times(firing_rate, saccades["temporal"], p=0.01, permute=False)
        rts_sc_n = hf.get_response_times(firing_rate, saccades["nasal"], p=0.01, permute=False)
        
        # Preferred direction
        max_fr = np.max(fr_sc, axis=1)
        pref_sc = np.argmax(max_fr, axis=1)
        
        rts_sc = np.array([np.where(pref_sc[:, None] == 0, rts_sc_t, rts_sc_n),
                           np.where(pref_sc[:, None] == 1, rts_sc_t, rts_sc_n)])
        # [(pref.,nonp.),n, (rt,dir)]
                        
        if plot == "raster_RT":
            plts_sp = os.path.join(save_path, "plots", "Neurons", "reaction time")
            hf.plot_raster(valid_spiketimes, fr_sc, sync_cam, saccades_all, 
                           tw, c_types, cluster_type, sac_colors, msc_colors,  
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
analysis = "sac_RT_all" # exp, exp_neu, sac_vis
                     # sac_RT_all,
                     # sac_amp; sac_RT_dir, sac_MI, sac_dir, sac_PCA
period =  "all" # "chirp"

if analysis[:6] == "sac_RT":
    fr_win_name = "_40ms_causal.npy" 
    fr_win = [-0.04, 0.] 
else:
    fr_win_name = "_100ms_causal.npy" 
    fr_win = [-0.1, 0.] 
    
save_rts = True

units_for_plot = []# [30,70,108] [nw,tca,bw] sac_RT, 128,407
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
        
        # Saccades
        hf.plot_sac_trayectory(saccades, pupil_center, sac_colors, 
                               save_path, exp)
        hf.plot_event(saccades, pupil_center[0,:], sac_colors, 
                      "x coordinate", exp, save_path)
        
        hf.plot_saccades_2d(saccades, pupil_center, sac_colors, exp, save_path)
        
        sac_var = hf.saccade_variance(pupil_center, saccades)
        results["sac_var"].append(sac_var)
    
    elif analysis == "sac_vis":
        
        off_sets = [-0.2, -0.1, 0, 0.1, 0.2]
        screen_t = []
        for off_set in off_sets:
            sac_vis = hf.get_sac_vis(saccades, sync_cam, vis_stim, Spke_Bundle,
                                     off_set=off_set)
            sac_vis = hf.get_screen(sac_vis, Spke_Bundle, vis_stim)
    
            dif_screen = hf.get_screen_dir(sac_vis)
            
            #for n in range(len(sac_vis["screen"])):
            #    if sac_vis["screen"][n].size > 1:
            #hf.plot_screen_ex(sac_vis, save_path, n=30)
            screen_t.append(dif_screen)
            
        hf.plot_mean_screen(screen_t, off_sets, save_path, exp)    
        results["dif_screen"].append(screen_t)
    
        
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
        
        if analysis == "exp_neu":
            hf.plot_conn(connected_pairs, cluster_type, colors, save_path, exp)
                
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
        
        elif analysis == "sac_RT_all":
            tw, fr_sc, rts_sc = \
                sac_res_analysis(saccades, firing_rate, valid_spiketimes, 
                                 sync_cam, c_types, save_path, cluster_type, 
                                 colors, exp, plot="hist")
                
            results["fr_sc"].append(fr_sc)
            results["rts_sc"].append(rts_sc) 
            
            if save_rts:
                np.save(os.path.join(save_path, "rts", "all", exp), rts_sc)
        
        elif analysis == "sac_RT_dir":
            tw, fr_sc, rts_sc, pref_sc = \
                sac_dir_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="RT", plot="scat")
                
            results["fr_sc"].append(fr_sc)
            results["rts_sc"].append(rts_sc) 
            results["pref_sc"].append(pref_sc)
            
            if save_rts:
                np.save(os.path.join(save_path, "rts", "dir", exp), rts_sc)
                
        elif analysis == "sac_MI":        
            tw, mutual_info = \
                sac_dir_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="MI", plot="MI", win = [-0.5,1])
                
            results["mutual_info"].append(mutual_info) 
            
            if save_rts:
                rts_sc = np.load(os.path.join(save_path, "rts", exp+ ".npy"))
                results["rts_sc"].append(rts_sc) 
            
        elif analysis == "sac_dir":                
            tw, fr_sc, sac_dir = \
                sac_dir_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp,
                                 an_type="dir", plot="nratio")
            results["fr_sc"].append(fr_sc)
            results["sac_dir"].append(sac_dir) 
        
        elif analysis == "sac_PCA":              
            tw, fr_sc, pca_results, exp_var_n = \
                sac_dir_analysis(saccades, pupil_center, firing_rate, 
                                 valid_spiketimes, sync_cam, c_types, 
                                 save_path, cluster_type, colors, exp)
                
            results["fr_sc"].append(fr_sc)
            results["PCA_var"].append([pca_results, exp_var_n])
        
assert False
## All plots

if analysis == "exp":
    hf.plot_sac_var(results["sac_var"], save_path)

else:    
    all_types_cat = [x for exp in results["types"] for x in exp]
    c_types_all = np.array([colors[n] for n in all_types_cat])
    
    
    if analysis == "exp_neu": # n 
        hf.plot_types(experiments, results["types"], colors, save_path)
            

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

    elif analysis[:6] == "sac_RT_all":
        rt_edges = np.arange(-0.3, 0.6, 0.02)

        rts_sc_all = np.concatenate([rts for rts in results["rts_sc"]], axis=0)
        
        hf.plot_sc_hist(rts_sc_all, c_types_all, rt_edges, save_path)
                
        rts_sc_list = [rt[:,0] for rt in results["rts_sc"]]
        hf.plot_all_rt(rts_sc_list, results["types"], colors, save_path)
        
    elif analysis[:6] == "sac_RT":
        rt_edges = np.arange(-0.3, 0.6, 0.02)

        rts_sc_all = np.concatenate([rts for rts in results["rts_sc"]], axis=1)
        
        hf.plot_sc_hist(rts_sc_all, c_types_all, rt_edges, save_path)
        
        rt_an_edges = np.arange(30, 60, 1)
        hf.plot_sc_rt_angle(rts_sc_all, c_types_all, rt_an_edges, save_path)
            
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
            


