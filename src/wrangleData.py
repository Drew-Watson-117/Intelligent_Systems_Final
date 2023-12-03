import pandas as pd
import matplotlib.pyplot as plt

# Data from https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-flsum_science/sci_xrsf-l2-flsum_g15_s20100331_e20200304_v1-0-0.nc

def main():

    #evalSumFile("../data/sci_xrsf-l2-flsum_g15_s20100331_e20200304_v1-0-0.nc")
    #evalAvgFile("../data/sci_xrsf-l2-avg1m_g15_s20100331_e20200304_v1-0-0.nc")
    
    flareSummary = pd.read_csv("../data/xrsSummary.csv",header=0)
    date_time = pd.to_datetime(flareSummary.pop('Date_Time'), format='%Y-%m-%d %H:%M:%S')
    # flareSummary.index = date_time
    
    # print(date_time.head())
    # print(flareSummary.head())

    flareSummary.plot(y='XRS_B_Flux',figsize=(9,6),logy=True)
    
    N = len(flareSummary)
    trainSplit = 0.7
    validSplit = 0.2
    testSplit = 1.0 - trainSplit - validSplit
    trainData = flareSummary[0:int(N*trainSplit)].get("XRS_B_Flux")
    validData = flareSummary[int(N*trainSplit):int(N*validSplit)].get("XRS_B_Flux")
    testData = flareSummary[int(N*validSplit):].get("XRS_B_Flux")
    print(trainData.head())
    print(testData.head())

if __name__=="__main__":
    main()

