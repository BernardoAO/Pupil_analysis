# Pupil_analysis

This code serves to process pupil data obtained from DeepLabCut (DLC) and analyse it with neural activity.

## Contents:
- Pupil_movement.py : Script that takes .pickle files with the pupil trackers from DLC and outputs a .npy file with a dictionary of every exp. and its analysed variables.
- Neuron_pupil_analysis.py : Script that takes the output from Pupil_movement.py, together with the neural data.
- Helper_functions.py : File with helper functions used for the rest of the scripts