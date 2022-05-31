from tokenize import Double
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat
def times_15WMode():
    arrFP32 = np.loadtxt("outputfiles/averages_15WMode/avg_time_ms_FP32_1000_itr_400000000_bytes.txt")
    arrFP16 = np.loadtxt("outputfiles/averages_15WMode/avg_time_ms_FP16_1000_itr_400000000_bytes.txt")
    arrDLA = np.loadtxt("outputfiles/averages_15WMode/avg_time_ms_FP16_DLA_1000_itr_400000000_bytes.txt")

    meanFP16 = stat.mean(arrFP16)
    meanFP32 = stat.mean(arrFP32)
    meanDLA = stat.mean(arrDLA)
    listAvgTime = [meanFP32, meanFP16, meanDLA]
    listLabels = ['FP32', 'FP16', 'FP16 with DLA']
    colors = ['red', 'blue', 'green']
    width = 0.35

    fig = plt.axes()
    fig.bar(listLabels, listAvgTime, width, color=colors)

    plt.show()

def power_GPU():
    arrFP32 = np.loadtxt("outputfiles/averages_15WMode/avg_mw_GPU_FP32_1000_itr_400000000_bytes.txt")
    arrFP16 = np.loadtxt("outputfiles/averages_15WMode/avg_mw_GPU_FP16_1000_itr_400000000_bytes.txt")
    arrDLA = np.loadtxt("outputfiles/averages_15WMode/avg_mw_GPU_FP16_DLA_1000_itr_400000000_bytes.txt")

    data = [arrFP32, arrFP16, arrDLA]
    listLabels = ['FP32', 'FP16', 'FP16 with DLA']
    fig = plt.axes()
    ax = fig.boxplot(data, column = 'area_mean', by = listLabels)

    plt.show()

times_15WMode()