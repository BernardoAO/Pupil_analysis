## Pupila movement analysis
### Bernardo AO
import numpy as np
import os
import pickle
from tqdm import tqdm
#import matplotlib.pyplot as plt
import scipy.signal as signal
import Helper_functions as hf

data_path = r"D:\NP data\Bernardo_awake_cx\DLC\left_eye\labeled_videos"
save_path = r"D:\NP data\Bernardo_awake_cx\Results"
pd_path = os.path.join(save_path,"pupil_data.npy")

n_pupil = 8
n_eyelid = 4
fps = 200 # Hz
smooth_window = 10
output_variables = ["session", "awake","ROIs_smooth", "pupil_center", 
                    "pupil_size", "saccade_indx"]

if os.path.isfile(pd_path):
    pupil_data = np.load(pd_path, allow_pickle=True).item()
else:
    pupil_data = hf.create_pupil_data(output_variables)

os.chdir(data_path)
all_files = [f for f in os.listdir() if f[-11:] == "full.pickle"]

work_files = all_files

for file in tqdm(work_files, desc="Files processed"):
    try:
        session = file[5:24]
        exp = [exp for exp in pupil_data 
               if pupil_data[exp]["session"].eq(session).any()][0]
        
        ## Import data
        with open(file, 'rb') as f:
            data = pickle.load(f)
    
        n_frames = data["metadata"]['nframes']
        tv = np.arange(0, n_frames / fps, 1 / fps)
        ROIs = np.zeros((n_pupil + n_eyelid, 2, n_frames)) # 12, (x,y), n_frames
        confidence = np.zeros((n_pupil + n_eyelid, n_frames)) 
        keys = list(data.keys())
    
        for frame in range(n_frames):
            frame_name = keys[frame + 1]
            ROI = np.array(data[frame_name]['coordinates']).reshape(n_pupil + n_eyelid, 2)
            ROIs[:,:,frame] = ROI
            confidence[:,frame] = np.squeeze(np.array(data[frame_name]['confidence']))
        
        
        
        ## Time smooth
        ROIs_smooth = hf.time_smooth_ROI(ROIs, smooth_window)
        for r in range(ROIs_smooth.shape[0]):
            for x in range(ROIs_smooth.shape[1]):
                ROIs_smooth[r,x,:] = hf.interpolate_outliers(tv, ROIs_smooth[r,x,:])
        
        # Exceptios
        ROIs_smooth = hf.handle_exceptions(ROIs_smooth, tv, session)

        ## Get size
        eyelids_mean = np.nanmean(ROIs_smooth[n_pupil:,:,:],axis=2)
        eye_lenght = np.linalg.norm(eyelids_mean[1,:] - eyelids_mean[3,:])
        
        pupil_size = hf.get_pupil_size(ROIs_smooth[:n_pupil,:,:]) / eye_lenght
        
        non_nan = ~np.isnan(pupil_size) 
        filtered = signal.savgol_filter(pupil_size[non_nan], 
                                                window_length=200, polyorder=2)
        pupil_size_clean = np.copy(pupil_size)
        pupil_size_clean[non_nan] = filtered
        
        ## Get position
        pupil_center = hf.get_pupil_center(ROIs_smooth[:n_pupil,:,:], 
                                           pupil_size_clean * eye_lenght)
        
        ## Get saccades
        saccade_indx = []#hf.get_saccades(retina_center)
        
        # Plot
        hf.plot_pupil_results(tv, pupil_size, pupil_size_clean, pupil_center, 
                              eyelids_mean, saccade_indx, session, save_path)
    
        ## Save
        mask = pupil_data[exp]["session"] == session
    
        row_index = pupil_data[exp].loc[mask].index[0]
        pupil_data[exp].at[row_index, "ROIs_smooth"] = ROIs_smooth
        pupil_data[exp].at[row_index, "pupil_center"] = pupil_center
        pupil_data[exp].at[row_index, "pupil_size"] = pupil_size
        pupil_data[exp].at[row_index, "saccade_indx"] = saccade_indx
    except:
        tqdm.write(" Error when processing file " + file)


#np.save(pd_path, pupil_data)



