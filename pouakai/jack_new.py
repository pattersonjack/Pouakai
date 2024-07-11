from pouakai import consume_moa

import os
from glob import glob
import numpy as np
import pandas as pd
import time

start = time.time()

files = np.genfromtxt('/local/jack/src/Pouakai/pouakai/jack_files_12P.csv', delimiter = ',', dtype = str)
# files = np.genfromtxt('/home/users/jlp89/jack_files_415029.csv', delimiter = ',', dtype = str)
# files = np.genfromtxt('/local/jack/src/Pouakai/pouakai/jack_files_1685_Toro.csv', delimiter = ',', dtype = str)
print(files)
# files = files[4:]
# print(files[0])

# files = ['/home/phys/astro8/MJArchive/octans/20231123/2023ygp-0003_i.fit']
# print(files)

savepath = '/local/jack/src/Pouakai/output/1685_Toro/'

if not os.path.exists(savepath):
    os.makedirs(savepath)

consume_moa(files,cores=1,savepath=savepath,rescale=True,update_cals=True,time_tolerence=45,dark_tolerence=1,plot=False, limit_source=[12,17])
# print()