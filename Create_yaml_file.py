### Bernardo AO

import os

maindir = r"D:\NP data\Bernardo_awake_cx\DLC\left_eye\models\Left_pupil-2023-03-16\videos"

files = [file for file in os.listdir(maindir)]
files.sort()

output = ""
for file in files:
    output += "\"D:/NP data/Bernardo_awake_cx/DLC/left_eye/models/Left_pupil-2023-03-16/videos/" + \
                file + "\":\n\tcrop: 0, 256, 0, 260\n"