import pandas as pd
import numpy as np
import os
import scipy.io

dataFolder = "./new1234/new"
txtFiles = os.listdir(dataFolder)
excel = pd.read_excel("./shaoyuan_recode/mockCrimeME_Final.xlsx")[["sub","count","StartFrame","EndFrame"]]

for each in txtFiles:
    print(each)
    data = pd.read_csv(f'{dataFolder}/{each}', sep='\t', header=None,engine='python')
    data = pd.DataFrame(data, columns=[0, 1, 2, 3, 4])[1:]

    neededExcel = excel[excel['sub']==int(each.split(".")[0])]
    neededExcel = neededExcel.reset_index(drop=True)

    print(neededExcel.shape[0])
    for i in range(neededExcel.shape[0]):
        print(neededExcel.loc[i,"sub"])
        figureName = f"{neededExcel.loc[i,'sub']}_{neededExcel.loc[i,'count']}"
        startFrame = neededExcel.loc[i, "StartFrame"]
        endFrame = neededExcel.loc[i, "EndFrame"]
        startTime = int(startFrame / 30 * 200)
        endTime = int(endFrame / 30 * 200)
        # x = data.loc[startTime:endTime, 0].values
        y4 = data.loc[startTime:endTime, 4].values
        print(figureName)
        y_norm = (y4 - np.min(y4))/(np.max(y4)-np.min(y4))
        scipy.io.savemat('./eda/'+figureName+'_eda.mat',{"array":y_norm})




