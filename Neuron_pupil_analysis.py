## Neural pupila coding
### Bernardo AO
import os 
import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
import Helper_functions as hf
assert False
# data file names
pupil_data_path = r"D:\NP data\Bernardo_awake_cx\Results\pupil_data.npy"
spike_bundle_path = r"D:\NP data\analysis\data-single-unit"
save_path = r"D:\NP data\Bernardo_awake_cx\Results"

# Session information
fs = 30000 # Hz
camara_fs = 200 # Hz
colors =  {"TCA":"orchid", "NW":"salmon", "BW":"black"} 

# Parameters
period =  "all" # "chirp"

# Get pupil data
pupil_data_all = np.load(pupil_data_path,allow_pickle=True).item()

experiments = [exp for exp in os.listdir(spike_bundle_path) if exp[:2] == "20"]
experiments.sort()


# file loop
for exp in ['2023-03-16_12-16-07']: #tqdm(experiments, desc="Files processed"):
    plot_name = exp + "_" + period
    
    ## Import 
    # spike data
    Spke_Bundle, spiketimes, SIN_data = \
        hf.import_spike_data(exp, spike_bundle_path)
    
    # merge pupil data for the exp
    sync_cam, pupil_size, pupil_center = \
        hf.get_pupil_data(pupil_data_all[exp], Spke_Bundle, exp, period)
    
    hf.plot_exp(Spke_Bundle, sync_cam, exp, save_path)
    
    # get valid clusters
    valid_cluster_indx, cluster_type = \
        hf.get_valid_cluster(Spke_Bundle, SIN_data)            
    valid_spiketimes = [spiketimes[i] for i in valid_cluster_indx]
    
    
    ## Firing rate
    print("getting fr")
    firing_rate = hf.get_firing_rate(valid_spiketimes, sync_cam)
    
    m_fr = np.expand_dims(np.mean(firing_rate, axis=1), axis=1)
    std_fr = np.expand_dims(np.std(firing_rate, axis=1), axis=1)    
    z_fr = (firing_rate - m_fr) / std_fr
        
    ## Pupil size
    mean_fr_size, _, s_bins = \
        hf.get_mean_fr_size(z_fr, pupil_size, 0.12, 0.42)
    
    n_plot = 100
    for n in np.random.randint(0,z_fr.shape[0],n_plot):
        plt.plot(s_bins,mean_fr_size[n,:], 
                 alpha=0.2, color=colors[cluster_type[n]])
    plt.xlabel("pupil size norm")
    plt.ylabel("z-scored firing rate")
    plt.show()
        
    neu_pupil_corr = hf.get_correlation(z_fr, pupil_size)
    
    corr_edges = np.arange(-0.3,0.32,0.010)
    hf.plot_correlation_cum(neu_pupil_corr, cluster_type, colors, 
                        corr_edges, plot_name, save_path)
     
    
    ## Pupil center
    center_edges = np.arange(105, 145, 5)
    mean_fr_center = hf.get_mean_fr_center(firing_rate, pupil_center, center_edges)

    similarity_type = hf.get_similarity(mean_fr_center, cluster_type, colors)

    plot_bin = 19
    hf.plot_similarity_2d(similarity_type, plot_bin, 
                          center_edges, plot_name, save_path,clim=[-1,1])


    
    ps_change_fast = sync_cam[hf.get_events(pupil_size, sync_cam)]
    ps_change_slow = sync_cam[hf.get_events(pupil_size, sync_cam, 50)]


