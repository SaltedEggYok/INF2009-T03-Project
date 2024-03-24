import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

sys.path.append("..")
#from Facial_Emotion_Recognition.emotion_detector import detect_emotion
#from SER.Emotion_Voice_Detection_Model import SER

# function to read the txt file 
def read_emotion_data_from_file(file_path):
    emotion_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  
            if not line:  
                continue
            timestamp_str, emotions_str = line.split(':', 1)
            timestamp = tuple(map(int, timestamp_str.strip('()').split(', ')))
            emotions = ast.literal_eval(emotions_str.strip())
            emotion_data[timestamp] = emotions
    return emotion_data

# placeholder (see how it looks like)
def generate_sample_data():
    np.random.seed(0)
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2021', periods=100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'value': np.random.rand(100)
    })
    return data

# Converting of dict to df (HW here)
def process_emotion_data(raw_emotion_data):
    records = []
    for timestamp, emotions in raw_emotion_data.items():
        ts = pd.Timestamp(year=2021, month=1, day=1, hour=timestamp[0], minute=timestamp[1], second=timestamp[2])
        for person_id, emotion in emotions.items():
            records.append({
                "Timestamp": ts,
                "Person": person_id,
                "Emotion": emotion
            })
    
    return pd.DataFrame.from_records(records)

# Using example data
file_path = '../Facial_Emotion_Recognition/ERM_Results/emotion_dict.txt' 
raw_emotion_data = read_emotion_data_from_file(file_path)

def get_emotion_data():
    # Change this to put our data
    timestamps = pd.date_range('2021-01-01', periods=10, freq='15S')
    averaged_emotions = np.random.uniform(low=0, high=1, size=10)
    return dict(zip(timestamps.strftime('%H:%M:%S'), averaged_emotions))

emotion_df = process_emotion_data(raw_emotion_data)

emotion_to_num = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
emotion_df['Emotion'] = emotion_df['Emotion'].map(emotion_to_num)

# fake data for the facial and speech graph
data = generate_sample_data()
emotion_data = get_emotion_data()

# dict to a df
df_emotion = pd.DataFrame(list(emotion_data.items()), columns=['Timestamp', 'Averaged Emotion'])


# GUI STARTS HERE
metric1, metric2 = 'Happy', 'Excited'

st.title('Edvisor Dashboard')
st.header('Overview')

# metrics at the top
col1, col2 = st.columns(2)
col1.metric("Emotion", metric1)
col2.metric("Speech", metric2)

# Charts
st.subheader('Charts')
fig_col1, fig_col2 = st.columns(2)

# First chart maybe show the emotion? 
with fig_col1:
    fig, ax = plt.subplots()
    pivot_table = data.pivot_table(index='category', columns='date', values='value', aggfunc=np.mean)
    cax = ax.matshow(pivot_table, cmap='viridis')
    plt.title('Emotion Chart')
    st.pyplot(fig)

# Second chart show speech?
with fig_col2:
    fig, ax = plt.subplots()
    ax.hist(data['value'], bins=15)
    plt.title('Speech Chart')
    st.pyplot(fig)

st.subheader('Emotion Over Time')
fig, ax = plt.subplots(figsize=(10, 4))

# Plotting a line for each person 
for person_id, group_df in emotion_df.groupby('Person'):
    # Sort by timestamp to plot it properlyy
    group_df = group_df.sort_values('Timestamp')  
    ax.plot(group_df['Timestamp'], group_df['Emotion'], marker='o', label=f'Person {person_id}')

all_timestamps = pd.to_datetime(emotion_df['Timestamp'].sort_values().unique())
ax.set_xticks(all_timestamps)
ax.set_xticklabels([ts.strftime('%H:%M:%S') for ts in all_timestamps], rotation=45, ha='right')

plt.ylabel('Emotion Level')
plt.xlabel('Time')
plt.legend(title='Person ID')
plt.title('Emotion Over Time')
plt.tight_layout()
st.pyplot(fig)

st.subheader('Processed Emotion Data')
st.dataframe(emotion_df)




