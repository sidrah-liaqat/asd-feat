# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky, Halil Helvaci
# This script takes behavior model predictions
# and converts them to aggregate statistics
# April 2022 : Converted from colab notebook to python file by Sidrah

#Extract video file names
import pandas as pd
import numpy as np
import sys
import os
import glob
import ntpath

# usage example: temporal_to_aggregate.py test F

split = str(sys.argv[1])
version = str(sys.argv[2])
asd_data_path = 'PATH/TO/FEATURES/results/hj_{}_{}/'.format(split,version)
FPS = 30
social_frames =  60#0.5sec
OUTDIR='PATH/TO/FEATURES/aggregate/'
os.makedirs(OUTDIR, mode=0o777, exist_ok=True)
all_files = glob.glob(os.path.join(asd_data_path, "*.csv"))

raw_data = []
# read in the aggregate
#aggregate_gt_df = pd.read_csv(split+'_agg.csv', usecols=['filename', 'sub_id', 'asd_longitudinal', 'visit', 'gender'])
aggregate_gt_df = pd.read_csv('PATH/TO/FEATURES/labels/session_labels.csv',
                              usecols=['filename', 'sub_id', 'asd_longitudinal', 'visit', 'gender'])
#smile
smile_indx = []
smile_indx_start = []
smile_indx_end= []
total_smile_frequency = 0
smile_rate = 0
smile_prop = 0
smile_dur = 0
#lookface
lf_indx = []
lf_indx_start = []
lf_indx_end= []
total_lf_frequency = 0
lf_rate = 0
lf_prop = 0
lf_dur = 0
#vocal
vocal_indx = []
vocal_indx_start = []
vocal_indx_end= []
total_vocal_frequency = 0
vocal_rate = 0
vocal_prop = 0
vocal_dur = 0
#lookobject
lo_indx = []
lo_indx_start = []
lo_indx_end= []
total_lo_frequency = 0
lo_rate = 0
lo_prop = 0
lo_dur = 0

for in_files in all_files:
    # Import dataset
    my_data = pd.read_csv(in_files)
    # Drop frames that are not valid
    # old version my_data = my_data[my_data.valid != 0]
    my_data.dropna()
    # Reset index
    my_data = my_data.reset_index(drop=True)

    # SMILE CALCULATIONS
    # smile_indx_start = []
    # smile_indx_end= []
    total_smile_frequency = 0
    smile_rate = 0
    smile_prop = 0
    smile_dur = 0

    # select the smile column from the dataframe
    smile_loop = my_data["smile"]
    # create empty list for smile indexes
    smile_indx = []
    # if smile == 1 append it to the empty list that we created
    for a in range(len(smile_loop)):
        if my_data["smile"][a] == 1:
            smile_indx.append(a)

    # select the starting indexes of the smile window
    smile_indx_start = []
    # when there are no smile events (zero length) avoid appending element as this
    # will crash the code
    if len(smile_indx) > 0:
        # append the very first element since it is left out in the for loop below
        smile_indx_start.append(smile_indx[0])
    # if the difference between two smile indexes is larger than 1 it means that a new
    # smile window has started
    for b in range(len(smile_indx)):
        if (smile_indx[b] - smile_indx[b - 1]) > 1:
            # append the starting smile index
            smile_indx_start.append(smile_indx[b])

    # select the ending indexes of the smile windows
    smile_indx_end = []
    for c in range(len(smile_indx)):
        if (smile_indx[c] - smile_indx[c - 1]) > 1:
            # append the ending index of smile
            smile_indx_end.append(smile_indx[c - 1])

    # when there are no smile events (zero length) avoid appending element as this
    # will crash the code
    if len(smile_indx) > 0:
        # append the very last element since it is left out in the for loop above
        smile_indx_end.append(smile_indx[-1])

    # calculate the total smile duration in the video
    smile_dur_list = []
    for k in range(len(smile_indx_start)):
        # length is the end index minus start index
        dif = smile_indx_end[k] - smile_indx_start[k]
        smile_dur_list.append(dif)
    # sum all the list elements to get total smile duration
    smile_dur = sum(smile_dur_list) / FPS
    # calculate the smile frequency
    total_smile_frequency = len(smile_indx_start)
    # total video duration in seconds
    video_dur = (my_data.shape[0] / FPS)
    # 60 seconds = 1 min
    smile_rate = total_smile_frequency / (video_dur / 60)
    # see definitions above in colab
    smile_prop = (smile_dur) / video_dur

    # print("Video Duration: ", video_dur,"\n")
    # print("Smile indexes start: ", smile_indx_start)
    # print("Smile indexes end: ",smile_indx_end,"\n")

    # print("Smile frequency: ", total_smile_frequency)
    # print('Smile rate = ', smile_rate)
    # print('Smile prop = ', smile_prop)
    # print("Smile duration: ", smile_dur,"\n")

    # LOOKFACE CALCULATIONS
    total_lf_frequency = 0
    lf_rate = 0
    lf_prop = 0
    lf_dur = 0
    # select the lookface column from the dataframe
    lf_loop = my_data["look_face"]
    # create empty list for lf indexes
    lf_indx = []
    # if lookface == 1 append it to the empty list that we created
    for d in range(len(lf_loop)):
        if my_data["look_face"][d] == 1:
            lf_indx.append(d)

    # select the starting indexes of the lf window
    lf_indx_start = []

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(lf_indx) > 0:
        # append the very first element since it is left out in the for loop below
        lf_indx_start.append(lf_indx[0])
    # if the difference between two lf indexes is larger than 1 it means that a new
    # lookface window has started
    for e in range(len(lf_indx)):
        if (lf_indx[e] - lf_indx[e - 1]) > 1:
            # append the starting lf index
            lf_indx_start.append(lf_indx[e])

    lf_indx_end = []
    for f in range(len(lf_indx)):
        if (lf_indx[f] - lf_indx[f - 1]) > 1:
            # append the ending index of lf
            lf_indx_end.append(lf_indx[f - 1])

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(lf_indx) > 0:
        # append the very last element since it is left out in the for loop above
        lf_indx_end.append(lf_indx[-1])

    # calculate the total lf duration in the video
    lf_dur_list = []
    for lf_i in range(len(lf_indx_start)):
        # length is the end index minus start index
        dif_lf = lf_indx_end[lf_i] - lf_indx_start[lf_i]
        lf_dur_list.append(dif_lf)
    # sum all the list elements to get total lf duration
    lf_dur = sum(lf_dur_list) / FPS
    # calculate the lf frequency
    total_lf_frequency = len(lf_indx_start)
    # 60 seconds = 1 min
    lf_rate = total_lf_frequency / (video_dur / 60)
    # see definitions above in colab
    lf_prop = (lf_dur) / video_dur

    # print("Lookface indexes start: ", lf_indx_start)
    # print("Lookface indexes end: ",lf_indx_end,"\n")

    # print("Lookface frequency: ", total_lf_frequency)
    # print('Lookface rate = ', lf_rate)
    # print('Lookface prop = ', lf_prop)
    # print("Lookface duration: ", lf_dur,"\n")

    # VOCAL CALCULATIONS
    total_vocal_frequency = 0
    vocal_rate = 0
    vocal_prop = 0
    vocal_dur = 0
    # select the lookface column from the dataframe
    vocal_loop = my_data["vocal"]
    # create empty list for vocal indexes
    vocal_indx = []
    # if lookface == 1 append it to the empty list that we created
    for v_i in range(len(vocal_loop)):
        if my_data["vocal"][v_i] == 1:
            vocal_indx.append(v_i)

    # select the starting indexes of the vocal window
    vocal_indx_start = []

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(vocal_indx) > 0:
        # append the very first element since it is left out in the for loop below
        vocal_indx_start.append(vocal_indx[0])

    # if the difference between two vocal indexes is larger than 1 it means that a new
    # lookface window has started
    for ee in range(len(vocal_indx)):
        if (vocal_indx[ee] - vocal_indx[ee - 1]) > 1:
            # append the starting vocal index
            vocal_indx_start.append(vocal_indx[ee])

    vocal_indx_end = []
    for ff in range(len(vocal_indx)):
        if (vocal_indx[ff] - vocal_indx[ff - 1]) > 1:
            # append the ending index of vocal
            vocal_indx_end.append(vocal_indx[ff - 1])

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(vocal_indx) > 0:
        # append the very last element since it is left out in the for loop above
        vocal_indx_end.append(vocal_indx[-1])

    # calculate the total vocal duration in the video
    vocal_dur_list = []
    for vocal_i in range(len(vocal_indx_start)):
        # length is the end index minus start index
        dif_vocal = vocal_indx_end[vocal_i] - vocal_indx_start[vocal_i]
        vocal_dur_list.append(dif_vocal)
    # sum all the list elements to get total vocal duration
    vocal_dur = sum(vocal_dur_list) / FPS
    # calculate the vocal frequency
    total_vocal_frequency = len(vocal_indx_start)
    # 60 seconds = 1 min
    vocal_rate = total_vocal_frequency / (video_dur / 60)
    # see definitions above in colab
    vocal_prop = (vocal_dur) / video_dur

    # print("Vocal indexes start: ", vocal_indx_start)
    # print("Vocal indexes end: ",vocal_indx_end,"\n")
    # print("Vocal frequency: ", total_vocal_frequency)
    # print('Vocal rate = ', vocal_rate)
    # print('Vocal prop = ', vocal_prop)
    # print("Vocal duration: ", vocal_dur,"\n")

    # LOOK OBJECT CALCULATIONS
    total_lo_frequency = 0
    lo_rate = 0
    lo_prop = 0
    lo_dur = 0
    # select the lookface column from the dataframe
    lo_loop = my_data["look_object"]
    # create empty list for lo indexes
    lo_indx = []
    # if lookface == 1 append it to the empty list that we created
    for lo_i in range(len(lo_loop)):
        if my_data["look_object"][lo_i] == 1:
            lo_indx.append(lo_i)

    # select the starting indexes of the lo window
    lo_indx_start = []

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(lo_indx) > 0:
        # append the very first element since it is left out in the for loop below
        lo_indx_start.append(lo_indx[0])

    # if the difference between two lo indexes is larger than 1 it means that a new
    # lookface window has started
    for lo_ee in range(len(lo_indx)):
        if (lo_indx[lo_ee] - lo_indx[lo_ee - 1]) > 1:
            # append the starting lo index
            lo_indx_start.append(lo_indx[lo_ee])

    lo_indx_end = []
    for lo_ff in range(len(lo_indx)):
        if (lo_indx[lo_ff] - lo_indx[lo_ff - 1]) > 1:
            # append the ending index of lo
            lo_indx_end.append(lo_indx[lo_ff - 1])

    # when there are no lf events (zero length) avoid appending element as this
    # will crash the code
    if len(lo_indx) > 0:
        # append the very last element since it is left out in the for loop above
        lo_indx_end.append(lo_indx[-1])

    # calculate the total lo duration in the video
    lo_dur_list = []
    for lo_i in range(len(lo_indx_start)):
        # length is the end index minus start index
        dif_lo = lo_indx_end[lo_i] - lo_indx_start[lo_i]
        lo_dur_list.append(dif_lo)
    # sum all the list elements to get total lo duration
    lo_dur = sum(lo_dur_list) / FPS
    # calculate the lo frequency
    total_lo_frequency = len(lo_indx_start)
    # 60 seconds = 1 min
    lo_rate = total_lo_frequency / (video_dur / 60)
    # see definitions above in colab
    lo_prop = (lo_dur) / video_dur
    # print("Look Object indexes start: ", lo_indx_start)
    # print("Look Object indexes end: ",lo_indx_end,"\n")
    # print("Look Object frequency: ", total_lo_frequency)
    # print('Look Object rate = ', lo_rate)
    # print('Look Object prop = ', lo_prop)
    # print("Look Object duration: ", lo_dur,"\n")

    # SOCIAL SMILE CALCULATIONS
    total_social_smile_frequency = 0
    social_smile_rate = 0
    social_smile_prop = 0
    social_smile_dur = 0
    # extend the smile start window for -2 seconds (60 frames)
    for g in range(len(smile_indx_start)):
        smile_indx_start[g] = smile_indx_start[g] - social_frames
    # extend the smile end window for +2 seconds (60 frames)
    for h in range(len(smile_indx_end)):
        smile_indx_end[h] = smile_indx_end[h] + social_frames

    # Create empty lists to store social smile start and end indexes
    social_smile_indexes_start = []
    social_smile_indexes_end = []
    # concatenated for loop to compare lookface START and smile

    ###
    # if the smile events occur more than lookface than outer loop is smile
    for i in range(len(lf_indx_start)):
        for j in range(len(smile_indx_start)):
            # if lookface is in the range of smile start - smile end append the smile index
            # to the empty list
            if lf_indx_start[i] in range(smile_indx_start[j],
                                         smile_indx_end[j] + 1):  # +1 allows the ending index to be inclusive
                # append the start that was found
                social_smile_indexes_start.append(smile_indx_start[j])
                # append the corresponding end to the start
                social_smile_indexes_end.append(smile_indx_end[j])

    # concatenated for loop to compare lookface END and smile
    for i in range(len(lf_indx_end)):
        for j in range(len(smile_indx_end)):
            if lf_indx_end[i] in range(smile_indx_start[j],
                                       smile_indx_end[j] + 1):  # +1 allows the ending index to be inclusive
                # append the corresponding start to the end
                social_smile_indexes_start.append(smile_indx_start[j])
                # append the end that was found
                social_smile_indexes_end.append(smile_indx_end[j])
    ###

    ###
    # else if the lookface events occur more than smile than outer loop is lookface
    for i in range(len(lf_indx_start)):
        for j in range(len(smile_indx_start)):
            if smile_indx_start[j] in range(lf_indx_start[i], lf_indx_end[i] + 1):
                # append the start that was found
                social_smile_indexes_start.append(smile_indx_start[j])
                # append the corresponding end to the start
                social_smile_indexes_end.append(smile_indx_end[j])
                # concatenated for loop to compare lookface END and smile

    for i in range(len(lf_indx_end)):
        for j in range(len(smile_indx_end)):
            if smile_indx_end[j] in range(lf_indx_start[i],
                                          lf_indx_end[i] + 1):  # +1 allows the ending index to be inclusive
                # append the corresponding start to the end
                social_smile_indexes_start.append(smile_indx_start[j])
                # append the end that was found
                social_smile_indexes_end.append(smile_indx_end[j])
    ###

    unique_social_smile_indexes_start = []
    # values are repeated due to the nature of the for loop thus get unique values
    unique_starts = set(social_smile_indexes_start)

    # convert the list {} to array [] (ps: set returns list and we need array for calculations)
    for number in unique_starts:
        unique_social_smile_indexes_start.append(number)
    unique_social_smile_indexes_start.sort()

    unique_social_smile_indexes_end = []
    # values are repeated due to the nature of the for loop thus get unique values
    unique_ends = set(social_smile_indexes_end)
    # convert the list {} to array []
    for number_end in unique_ends:
        unique_social_smile_indexes_end.append(number_end)
    unique_social_smile_indexes_end.sort()

    # add 60 to start indexes to fix to original
    for g in range(len(unique_social_smile_indexes_start)):
        unique_social_smile_indexes_start[g] = unique_social_smile_indexes_start[g] + social_frames
    # subtract 60 from end indexes to fix to original
    for h in range(len(unique_social_smile_indexes_end)):
        unique_social_smile_indexes_end[h] = unique_social_smile_indexes_end[h] - social_frames

    # calculate the total smile duration in the video
    social_smile_dur_list = []
    for ks in range(len(unique_social_smile_indexes_start)):
        # length is the end index minus start index
        dif_ss = unique_social_smile_indexes_end[ks] - unique_social_smile_indexes_start[ks]
        social_smile_dur_list.append(dif_ss)
    # sum all the list elements to get total smile duration
    social_smile_dur = sum(social_smile_dur_list) / FPS

    # calculate the smile frequency
    total_social_smile_frequency = len(unique_social_smile_indexes_start)
    # 60 seconds = 1 min
    social_smile_rate = total_social_smile_frequency / (video_dur / 60)
    # see definitions above in colab
    social_smile_prop = (social_smile_dur) / video_dur

    # print("Social Smile frequency: ", total_social_smile_frequency)
    # print('Social Smile rate = ', social_smile_rate)
    # print('Social Smile prop = ', social_smile_prop)
    # print("Social Smile duration: ", social_smile_dur,"\n")
    # print("Social smile indexes start: ",unique_social_smile_indexes_start)
    # print("Social smile indexes end: ",unique_social_smile_indexes_end)

    # SOCIAL VOCAL CALCULATIONS
    total_social_vocal_frequency = 0
    social_vocal_rate = 0
    social_vocal_prop = 0
    social_vocal_dur = 0

    # print("Vocal indexes start: ", vocal_indx_start)
    # print("Vocal indexes end: ", vocal_indx_end,"\n")
    # extend the vocal start window for -2 seconds (60 frames)
    for g in range(len(vocal_indx_start)):
        vocal_indx_start[g] = vocal_indx_start[g] - social_frames
    # extend the vocal end window for +2 seconds (60 frames)
    for h in range(len(vocal_indx_end)):
        vocal_indx_end[h] = vocal_indx_end[h] + social_frames

    # Create empty lists to store social vocal start and end indexes
    social_vocal_indexes_start = []
    social_vocal_indexes_end = []
    # concatenated for loop to compare lookface START and vocal

    ###
    # if the vocal events occur more than lookface than outer loop is vocal
    for i in range(len(lf_indx_start)):
        for j in range(len(vocal_indx_start)):
            # if lookface is in the range of vocal start - vocal end append the vocal index
            # to the empty list
            if lf_indx_start[i] in range(vocal_indx_start[j],
                                         vocal_indx_end[j] + 1):  # +1 allows the ending index to be inclusive
                # append the start that was found
                social_vocal_indexes_start.append(vocal_indx_start[j])
                # append the corresponding end to the start
                social_vocal_indexes_end.append(vocal_indx_end[j])

    # concatenated for loop to compare lookface END and vocal
    for i in range(len(lf_indx_end)):
        for j in range(len(vocal_indx_end)):
            if lf_indx_end[i] in range(vocal_indx_start[j],
                                       vocal_indx_end[j] + 1):  # +1 allows the ending index to be inclusive
                # append the corresponding start to the end
                social_vocal_indexes_start.append(vocal_indx_start[j])
                # append the end that was found
                social_vocal_indexes_end.append(vocal_indx_end[j])

    ###
    # else if the lookface events occur more than vocal than outer loop is lookface
    for i in range(len(lf_indx_start)):
        for j in range(len(vocal_indx_start)):
            if vocal_indx_start[j] in range(lf_indx_start[i], lf_indx_end[i] + 1):
                # append the start that was found
                social_vocal_indexes_start.append(vocal_indx_start[j])
                # append the corresponding end to the start
                social_vocal_indexes_end.append(vocal_indx_end[j])
                # concatenated for loop to compare lookface END and vocal

    for i in range(len(lf_indx_end)):
        for j in range(len(vocal_indx_end)):
            if vocal_indx_end[j] in range(lf_indx_start[i],
                                          lf_indx_end[i] + 1):  # +1 allows the ending index to be inclusive
                # append the corresponding start to the end
                social_vocal_indexes_start.append(vocal_indx_start[j])
                # append the end that was found
                social_vocal_indexes_end.append(vocal_indx_end[j])
    ###

    unique_social_vocal_indexes_start = []
    # values are repeated due to the nature of the for loop thus get unique values
    unique_starts = set(social_vocal_indexes_start)

    # convert the list {} to array [] (ps: set returns list and we need array for calculations)
    for number in unique_starts:
        unique_social_vocal_indexes_start.append(number)
    unique_social_vocal_indexes_start.sort()

    unique_social_vocal_indexes_end = []
    # values are repeated due to the nature of the for loop thus get unique values
    unique_ends = set(social_vocal_indexes_end)
    # convert the list {} to array []
    for number_end in unique_ends:
        unique_social_vocal_indexes_end.append(number_end)
    unique_social_vocal_indexes_end.sort()

    # add 60 to start indexes to fix to original
    for g in range(len(unique_social_vocal_indexes_start)):
        unique_social_vocal_indexes_start[g] = unique_social_vocal_indexes_start[g] + social_frames
    # subtract 60 from end indexes to fix to original
    for h in range(len(unique_social_vocal_indexes_end)):
        unique_social_vocal_indexes_end[h] = unique_social_vocal_indexes_end[h] - social_frames

    # calculate the total vocal duration in the video
    social_vocal_dur_list = []
    for ks in range(len(unique_social_vocal_indexes_start)):
        # length is the end index minus start index
        dif_ss = unique_social_vocal_indexes_end[ks] - unique_social_vocal_indexes_start[ks]
        social_vocal_dur_list.append(dif_ss)
    # sum all the list elements to get total vocal duration
    social_vocal_dur = sum(social_vocal_dur_list) / FPS
    # calculate the vocal frequency
    total_social_vocal_frequency = len(unique_social_vocal_indexes_start)
    # 60 seconds = 1 min
    social_vocal_rate = total_social_vocal_frequency / (video_dur / 60)
    # see definitions above in colab
    social_vocal_prop = (social_vocal_dur) / video_dur

    # print("Social Vocal frequency: ", total_social_vocal_frequency)
    # print('Social Vocal rate = ', social_vocal_rate)
    # print('Social Vocal prop = ', social_vocal_prop)
    # print("Social vocal indexes start: ",unique_social_vocal_indexes_start)
    # print("Social vocal indexes end: ",unique_social_vocal_indexes_end)

    #############################
    current_filename = ntpath.basename(in_files)
    #print(current_filename)
    raw_data_elem = ({'filename': current_filename, 'total_smile_fq': total_smile_frequency, 'smile_rate': smile_rate,
                      'total_smile_duration': smile_dur, 'smile_prop': smile_prop,
                      'total_lookface_fq': total_lf_frequency, 'lookface_rate': lf_rate,
                      'total_lookface_dur': lf_dur, 'lookface_prop': lf_prop,
                      'total_lookobject_fq': total_lo_frequency, 'lookobject_rate': lo_rate,
                      'total_lookobject_dur': lo_dur, 'lookobject_prop': lo_prop,
                      'total_vocal_fq': total_vocal_frequency, 'vocal_rate': vocal_rate,
                      'total_vocal_duration': vocal_dur, 'vocal_prop': vocal_prop,
                      'total_social_smile_fq': total_social_smile_frequency, 'social_smile_rate': social_smile_rate,
                      'total_social_smile_duration': social_smile_dur, 'social_smile_prop': social_smile_prop,
                      'total_social_vocal_fq': total_social_vocal_frequency, 'social_vocal_rate': social_vocal_rate,
                      'total_social_vocal_duration': social_vocal_dur, 'social_vocal_prop': social_vocal_prop})

    raw_data.append(raw_data_elem)

df = pd.DataFrame(raw_data, columns=['filename',
                                     'total_smile_fq', 'smile_rate', 'total_smile_duration', 'smile_prop',
                                     'total_lookface_fq', 'lookface_rate', 'total_lookface_dur', 'lookface_prop',
                                     'total_lookobject_fq', 'lookobject_rate', 'total_lookobject_dur',
                                     'lookobject_prop',
                                     'total_vocal_fq', 'vocal_rate', 'total_vocal_duration', 'vocal_prop',
                                     'total_social_smile_fq', 'social_smile_rate', 'total_social_smile_duration',
                                     'social_smile_prop', 'total_social_vocal_fq', 'social_vocal_rate',
                                     'total_social_vocal_duration',
                                     'social_vocal_prop'])

df = pd.merge(aggregate_gt_df, df, on='filename', how='inner')
df.to_csv(OUTDIR + 'hj_{}_{}_Aggregated.csv'.format(split, version), index=False)
print('Saved to hj_{}_{}_Aggregated.csv'.format(split, version))