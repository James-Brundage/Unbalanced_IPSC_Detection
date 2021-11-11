"""
Takes the true positive peaks as detected by experts in MiniAnalysis, generates all possible peaks from the matching
.abf recording, and matches the true positives with the generated peaks. This dataset is created without filtering,
meaning it will likely be severly unbalanced.
"""

# Imports
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import pyabf
from bisect import bisect_left
import tqdm
import os

# Read in true positives from raw data
print('Reading')
df = pd.read_excel('/Users/jamesbrundage/Box/James data.xlsx')
print('Done Reading')
#
path = '/Users/jamesbrundage/Box/Current 10K/7_29_2019 245 Continuous Export.abf'


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def align_peaks (df, recording_name='7_29_2019 245 Continuous Export.abf', abf_path=path, width=10):

    #Read .abf and grab info
    abf = pyabf.ABF(abf_path)
    x = abf.sweepX
    y = abf.sweepY

    testdf = df[df['ABF File'] == recording_name]

    # Grab the peak time indices and align with the index of the .abf
    peaks = testdf['Time (ms)'] * 10

    # Find peaks sensitive
    peak_x, properties = list(find_peaks((y * -1), width=width))

    # Align Peaks
    true_peaks = []
    distances = []
    for p in peaks:
        adjusted_peak = take_closest(peak_x, p)
        d = (adjusted_peak - p) / 10

        # Accounts for mujlktiple peaks at same place. Assumes the first one is correct.
        # Cheapest option computationally, but loses 1-5 true positives each time.
        if not adjusted_peak in true_peaks:
            true_peaks.append(adjusted_peak)
            distances.append(abs(d))

    # Create initial df
    dff = pd.DataFrame()
    dff['Time Index'] = peak_x
    dff['Current'] = y[peak_x]

    # Add Labels and distances from original
    tpdf = dff[dff['Time Index'].isin(true_peaks)].copy()
    fpdf = dff[~dff['Time Index'].isin(true_peaks)].copy()

    tpdf['Label'] = [1] * len(tpdf)

    tpdf['Offset'] = distances
    fpdf['Label'] = [0] * len(fpdf)
    fpdf['Offset'] = [0] * len(fpdf)

    dfff = pd.concat([tpdf, fpdf])

    dfff = dfff.sort_values('Time Index')

    # Eliminate edge peaks
    dfff = dfff[dfff['Time Index'] > 2000]
    dfff = dfff[dfff['Time Index'] < len(y) - 2000]

    dfff['File_Name'] = [recording_name]*len(dfff)

    return dfff

def grab_traces (target_peaks_df, abf_path=path, limit=0):

    # Read .abf and grab info
    abf = pyabf.ABF(abf_path)
    y = abf.sweepY

    # Function to grab trace
    def grab_trace(ind):
        return y[ind - 2000:ind + 2000]

    # Limiter
    if limit > 0:
        target_peaks_df = target_peaks_df[:limit]

    traces = []
    for i in tqdm.tqdm(target_peaks_df['Time Index']):
        traces.append(grab_trace(i))

    arr = np.array(traces)
    traces_df = pd.DataFrame(arr).reset_index()
    rdf = pd.concat([target_peaks_df.reset_index(), traces_df], axis=1)

    return rdf


# # Grab the file locations
abf_folder = '/Users/jamesbrundage/Box/Current 10K/'
abf_files = os.listdir(abf_folder)

peak_dfs = []
# for abf in tqdm.tqdm(list(set(df['ABF File']))):
for i in tqdm.tqdm(range(0,3)):
    abf = list(set(df['ABF File']))[i]

    pth = os.path.join(abf_folder, abf)
    peaks = align_peaks(df=df, recording_name=abf, abf_path=pth)
    peak_dfs.append(grab_traces(peaks, abf_path=pth, limit=0))


final_df = pd.concat(peak_dfs)
print(final_df)