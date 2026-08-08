import pandas as pd, numpy as np
wt = pd.read_excel('/data/inputs/workload_trace.xlsx')
wt['Duration_h'] = np.ceil(wt['EstimatedDuration_min']/60).astype(int)
print("cols:", wt.columns.tolist())
pri = {'RealTimeInference':0,'BatchInference':1,'AITraining':2}
wt['p'] = wt['TaskType'].map(pri)
print("cols after 'p':", wt.columns.tolist())
wt.sort_values(['ArrivalHour','p','TaskID'], inplace=True, kind='mergesort')
print("cols after sort:", wt.columns.tolist())
wt.reset_index(drop=True, inplace=True)
print("cols after reset:", wt.columns.tolist())
print("Duration_h exists?", 'Duration_h' in wt.columns)
print(wt[['TaskID','Duration_h']].head())
