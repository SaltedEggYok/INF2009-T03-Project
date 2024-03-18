import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
#from Facial_Emotion_Recognition.emotion_detector import detect_emotion
#from SER.Emotion_Voice_Detection_Model import SER


# placeholder (see how it looks like)
def generate_sample_data():
    np.random.seed(0)
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2021', periods=100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'value': np.random.rand(100)
    })
    return data

def get_emotion_data():
    # Change this to put our data
    timestamps = pd.date_range('2021-01-01', periods=10, freq='15S')
    averaged_emotions = np.random.uniform(low=0, high=1, size=10)
    return dict(zip(timestamps.strftime('%H:%M:%S'), averaged_emotions))

data = generate_sample_data()
emotion_data = get_emotion_data()

# dict to a df
df_emotion = pd.DataFrame(list(emotion_data.items()), columns=['Timestamp', 'Averaged Emotion'])

# Placeholders
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
fig, ax = plt.subplots()
ax.plot(df_emotion['Timestamp'], df_emotion['Averaged Emotion'], marker='o')
plt.xticks(rotation=45)
plt.title('Overall Chart')
plt.tight_layout()
st.pyplot(fig)

# Displaying of DF
st.subheader('Overall result?')
st.dataframe(data)





