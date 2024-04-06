import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

sys.path.append("..")

# Function to read data from file
def read_data_from_file(file_path):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            timestamp_str, values_str = line.split(':', 1)
            timestamp = tuple(map(int, timestamp_str.strip('()').split(', ')))
            values = ast.literal_eval(values_str.strip())
            data[timestamp] = values
    return data

# Function to process the data
def process_data(raw_data):
    records = []
    for timestamp, values in raw_data.items():
        ts = pd.Timestamp(year=2024, month=4, day=8, hour=timestamp[0], minute=timestamp[1], second=timestamp[2])
        for person_id, value in values.items():
            records.append({
                "Timestamp": ts,
                "Person": person_id,
                "Value": value
            })
    return pd.DataFrame.from_records(records)

emotion_raw_data = read_data_from_file('../Facial_Emotion_Recognition/ERM_Results/emotion_dict.txt')
speech_raw_data = read_data_from_file('../Facial_Emotion_Recognition/ERM_Results/speech_dict.txt')

emotion_df = process_data(emotion_raw_data)
speech_df = process_data(speech_raw_data)

# Convert emotion and speech to numeric values for plotting
value_to_num = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
emotion_df['Value_Num'] = emotion_df['Value'].map(value_to_num)
speech_df['Value_Num'] = speech_df['Value'].map(value_to_num)

average_emotion = emotion_df['Value_Num'].mean()
average_speech = speech_df['Value_Num'].mean()

def determine_metric_text(average_value):
    if average_value > 0:
        return 'Positive'
    elif average_value < 0:
        return 'Negative'
    else:
        return 'Neutral'

metric1_text = determine_metric_text(average_emotion)
metric2_text = determine_metric_text(average_speech)

# GUI starts here
st.title('Edvisor Dashboard')
st.header('Overview')

col1, col2 = st.columns(2)
col1.metric("Average Emotion", metric1_text)
col2.metric("Average Speech", metric2_text)

with col1:
    # Plotting Emotion Over Time
    st.subheader('Emotion Over Time')
    fig1, ax1 = plt.subplots()
    for person_id, group_df in emotion_df.groupby('Person'):
        group_df = group_df.sort_values('Timestamp')
        ax1.plot(group_df['Timestamp'], group_df['Value_Num'], marker='o', label=f'Person {person_id}')
    ax1.set_ylabel('Emotion Level')
    ax1.set_title('Emotion Over Time')
    ax1.legend()
    st.pyplot(fig1)

with col2:
    # Plotting Speech Over Time
    st.subheader('Speech Over Time')
    fig2, ax2 = plt.subplots()
    ax2.plot(speech_df['Timestamp'], speech_df['Value_Num'], marker='o', color='blue', label='Speech')
    ax2.set_ylabel('Speech Level')
    ax2.set_title('Speech Over Time')
    ax2.legend()
    st.pyplot(fig2)

# Overall Analysis
st.subheader('Overall Analysis')
analysis_fig, analysis_ax = plt.subplots()

# Average emotion and speech level for each person
for person_id in emotion_df['Person'].unique():
    emotion = emotion_df[emotion_df['Person'] == person_id]['Value_Num'].mean()
    speech = speech_df['Value_Num'].mean()
    analysis_ax.scatter(speech, emotion, label=f'Person {person_id}')

analysis_ax.axhline(0, color='grey', linestyle='--')
analysis_ax.axvline(0, color='grey', linestyle='--')
analysis_ax.set_xlabel('Average Speech Level')
analysis_ax.set_ylabel('Average Emotion Level')
analysis_ax.set_title('Overall Analysis')
analysis_ax.grid(True)
analysis_ax.legend()
st.pyplot(analysis_fig)

emotion_df.rename(columns={'Value': 'Emotion_Value'}, inplace=True)
speech_df.rename(columns={'Value': 'Speech_Value'}, inplace=True)

combined_df = pd.concat([emotion_df, speech_df], axis=1)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
st.subheader('Overall Data')
st.dataframe(combined_df)

