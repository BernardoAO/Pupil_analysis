## Neural pupila coding
### Bernardo AO
import os 
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
import Helper_functions as hf
#assert False

def ps_analysis(z_fr, pupil_size, cluster_type, colors, plot_name, save_path):
    
    # Mean fr vs pupil size
    mean_fr_size, _, s_bins = \
        hf.get_mean_fr_size(z_fr, pupil_size)
    
    n_plot = 100
    for n in np.random.randint(0,z_fr.shape[0],n_plot):
        plt.plot(s_bins, mean_fr_size[n,:], 
                 alpha=0.2, color=colors[cluster_type[n]])
    plt.xlabel("pupil size norm")
    plt.ylabel("z-scored firing rate")
    plt.show()
        
    # fr pupil size correlation
    neu_pupil_corr = hf.get_correlation(z_fr, pupil_size)
    
    corr_edges = np.arange(-0.3,0.32,0.010)
    hf.plot_correlation_cum(neu_pupil_corr, colors, cluster_type, corr_edges, 
                            plot_name, save_path)

def ps_events_analysis(pupil_size, z_fr, c_types, umap_data_path="none"):
    
    ps_change_fast_indx, _ = hf.get_events(pupil_size)
    ps_change_slow_indx, _ = hf.get_events(pupil_size, 10)
    #hf.plot_windows_and_events(pupil_size, sync_cam, sync_cam[ps_change_slow_indx])

    z_fr_ps_slow, tw = hf.get_fr_aligned(z_fr, ps_change_slow_indx)
    
    #np.save(os.path.join(save_path,"z_fr_ps_slow.npy"), z_fr_ps_slow)
    
    if not umap_data_path == "none":
        embedding = np.load(os.path.join(umap_data_path))
        
        #hf.plot_fr_aligned(tw, z_fr_ps_slow, c_types, save_path, name="fr_psc_slow")
        
        emb_p = np.array([[4,-7], [8,-5], [10,-7], [13,-3]]) # w,n,s,e
        mean_emb_fr, mean_emb_c = hf.get_mean_fr_2d(z_fr_ps_slow, embedding, 
                                                    emb_p, c_types)
        hf.plot_fr_aligned(tw, mean_emb_fr, mean_emb_c,)
    
        hf.plot_umap(embedding, emb_p, c_types, mean_emb_c)

def pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
                save_path, center_edges = np.arange(105, 145, 5)):
    
    mean_fr_center = hf.get_mean_fr_center(firing_rate, pupil_center, center_edges)

    similarity_type = hf.get_similarity(mean_fr_center, cluster_type, colors)

    plot_bin = 19
    hf.plot_similarity_2d(similarity_type, plot_bin, 
                          center_edges, plot_name, save_path,clim=[-1,1])

def saccade_analysis(saccades, firing_rate, valid_spiketimes, sync_cam, 
                     c_types, save_path, colors):
    saccades_all = np.concatenate((saccades["temporal"], 
                                saccades["nasal"]), axis=0)
    msc_colors = ["navy" if sc < len(saccades["temporal"]) else 
                  "violet" for sc in range(len(saccades_all))]
    
    win = [-0.25,1]
    fr_sc_t, tw = hf.get_fr_aligned(firing_rate, saccades["temporal"], win=win)    
    fr_sc_n, tw = hf.get_fr_aligned(firing_rate, saccades["nasal"], win=win)
    
    fr_sc = np.stack((fr_sc_t, fr_sc_n), axis=-1)

    hf.plot_raster(valid_spiketimes, sync_cam, saccades_all, msc_colors,
                tw, fr_sc, c_types, save_path, name="fr_man_sac")


    pca_pc = hf.neuron_PCA(fr_sc, c_types, n_components=10)
    
    hf.plot_pca(tw, pca_pc[:,:3,:,:], colors)
    hf.plot_pca(tw, pca_pc[:,:3,:,:], colors, multi_d=True)
    
    """ # TODO 
    hf.plot_angle(pc_angles)
    """

def main():
    # data file names
    pupil_data_path = r"D:\NP data\Bernardo_awake_cx\Results\pupil_data"
    spike_bundle_path = r"D:\NP data\analysis\data-single-unit"
    save_path = r"D:\NP data\Bernardo_awake_cx\Results"

    # Session information
    fs = 30000 # Hz
    camara_fs = 200 # Hz
    colors =  {"TCA":"orchid", "NW":"salmon", "BW":"black"} 

    # Parameters
    period =  "all" # "chirp"

    experiments = [exp for exp in os.listdir(spike_bundle_path) 
                   if exp[:2] == "20"]
    experiments.sort()


    # file loop
    for exp in ['2023-03-16_12-16-07']: #tqdm(experiments, desc="Files processed"):
        
        plot_name = exp + "_" + period
        
        ## Import
        
        # spike data
        Spke_Bundle, spiketimes, SIN_data = \
            hf.import_spike_data(exp, spike_bundle_path)
        
        # merge pupil data for the exp
        sync_cam, pupil_size, pupil_center, saccades = \
            hf.import_pupil_data(pupil_data_path, Spke_Bundle, exp, period)
        
        #hf.plot_exp(Spke_Bundle, sync_cam, exp, save_path)
        #hf.plot_pupil_stimuli(pupil_size, pupil_center, sync_cam, Spke_Bundle["events"])

        # get valid clusters
        valid_cluster_indx, cluster_type = \
            hf.get_valid_cluster(Spke_Bundle, SIN_data)            
        valid_spiketimes = [spiketimes[i] for i in valid_cluster_indx]
        c_types = np.array([colors[n] for n in cluster_type])
        
        ## Firing rate
        
        firing_rate, z_fr = hf.get_firing_rate(valid_spiketimes, sync_cam)
            
        ## Pupil size
        
        # correlation
        ps_analysis(z_fr, pupil_size, cluster_type, colors, plot_name, save_path)
        
        # size change events
        ps_events_analysis(pupil_size, z_fr, c_types, umap_data_path="none")
        
        ## Pupil center
        
        # similarity
        pc_analysis(firing_rate, pupil_center, cluster_type, colors, plot_name, 
                    save_path)
        
        # saccades    
        saccade_analysis(saccades, firing_rate, valid_spiketimes, sync_cam, 
                         c_types, save_path, colors)

if __name__ == "__main__":
    main()


