## Pupila movement analysis
### Bernardo AO
import numpy as np
import os
import pickle
#import matplotlib.pyplot as plt
import scipy.signal as signal
import Helper_functions as hf

data_path = r"D:\NP data\Bernardo_awake_cx\DLC\left_eye\labeled_videos"
save_path = r"D:\NP data\Bernardo_awake_cx\Results"

n_pupil = 8
n_eyelid = 4
fps = 200 # Hz
smooth_window = 10
output_variables = ["session", "awake","ROIs_smooth", "pupil_center", 
                    "pupil_size", "saccade_indx"]

os.chdir(data_path)
all_exp = [d for d in os.listdir()]

work_exp = all_exp#['2023-04-18_12-10-34']

for exp in work_exp:
    
    # Import or create pupil_data
    os.chdir(exp)    
    sessions_files = [f for f in os.listdir() if f[-11:] == "full.pickle"]
    
    pupil_data_path, pupil_data = hf.create_pupil_data(exp, save_path, 
                                     sessions_files, output_variables)
    
    for file in sessions_files:
        
        ## Import data
        session = file[5:24]
        with open(file, 'rb') as f:
            DLC_data = pickle.load(f)
        
        n_frames = DLC_data["metadata"]['nframes']
        tv = np.arange(0, n_frames / fps, 1 / fps)
        ROIs = np.zeros((n_pupil + n_eyelid, 2, n_frames)) # 12, (x,y), n_frames
        confidence = np.zeros((n_pupil + n_eyelid, n_frames)) 
        keys = list(DLC_data.keys())
    
        for frame in range(n_frames):
            frame_name = keys[frame + 1]
            ROI = np.array(DLC_data[frame_name]['coordinates']).reshape(n_pupil + n_eyelid, 2)
            ROIs[:,:,frame] = ROI
            confidence[:,frame] = np.squeeze(np.array(DLC_data[frame_name]['confidence']))
        
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
        saccade_indx = hf.import_saccades(session, exp)
        
        # Plot
        hf.plot_pupil_results(tv, pupil_size, pupil_size_clean, pupil_center, 
                              eyelids_mean, saccade_indx, session, save_path)
    
        ## Save
        mask = pupil_data["session"] == session
    
        row_index = pupil_data.loc[mask].index[0]
        pupil_data.at[row_index, "ROIs_smooth"] = ROIs_smooth
        pupil_data.at[row_index, "pupil_center"] = pupil_center
        pupil_data.at[row_index, "pupil_size"] = pupil_size_clean
        pupil_data.at[row_index, "saccade_indx"] = saccade_indx
    

    pupil_data.to_pickle(pupil_data_path)
    os.chdir("..")



