# Imports
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# COLORCODES
# Background Blue: #24A0ED
# Lighter Blue: #a7d9f7
# Darker Blue: #003571


#####################
### PAGE SETTINGS ###
#####################
# Configure Fav Icon and Title
favicon = 'images/favicon.ico'
st.set_page_config(
                page_title='Fall Detector',
                page_icon = favicon
                )

def _max_width_():
    max_width_str = f"max-width: 900px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
_max_width_()

# Use Local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style/style.css")


####################
### PAGE LAYOUT ####
####################
# Header Section
image = Image.open('images/header.png')
st.image(image,
        use_column_width=True,
        unsafe_allow_html=True,
        )



### OUR SOLUTION ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>OUR SOLUTION</h1>", unsafe_allow_html=True)
st.markdown(
"""
### Identify the fall and when this happens a notification is sent to a dedicated person - either a family member or a carer.
Unordered List:
- First
- Second
    - thirdt
    - Fourth
- Third
""")

### ZETEOH ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>OUR PARTNER: ZETEOH</h1>", unsafe_allow_html=True)
st.markdown(
"""
### Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
Unordered List:
- First
- Second
- Third
""")

st.markdown("<a style='color: white' href='https://www.zeteoh.com/'>>>zeteoh<<</a>", unsafe_allow_html=True)



### BEIND THE SCENCES ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>BEHIND THE SCENCES</h1>", unsafe_allow_html=True)

st.markdown(
"""
### Fall Detection Process Flow
1. Sensor Data is sent to Deep Learning [DL] Model
2. DL Model predicts: Fall / No Fall
3. In case of fall: Selected User are getting a notification on their smartphone
""")


### EXAMPLE DATA ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>EXAMPLE DATA</h1>", unsafe_allow_html=True)



### HOW IT WORKS ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>HOW IT WORKS</h1>", unsafe_allow_html=True)
col1, col2 = st.beta_columns([3, 1])
data = np.random.randn(10, 1)
col1.subheader("A wide column with a chart")
col1.line_chart(data)
col2.subheader("A narrow column with the data")
col2.write(data)


### FAQ ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>FAQ</h1>", unsafe_allow_html=True)

with st.beta_expander("Q : What happened when someone fall?"):
    st.write("""
        Good question! The system will send to notification to the register users.
    """)

with st.beta_expander("Q : What happened when the user didn't fall but somehow the algorithm think they fell?"):
    st.write("""
        We plan to have a pop-up messsage to check if the user actually fall or not. If the user do not response in
        5 minutes the system will send the notification! Otherwise it will discard the notification.
    """)

with st.beta_expander("Q : What kind of model did you use to develop the algorithm?"):
    st.write("""
        Maybe you miss the part above about how our algorithm works but no worry! We use RNN - LSTM (Long Short Term Memory)
        to develop the algorithm.
    """)

with st.beta_expander("Q : What is your accuracy on detecting fall pattern?"):
    st.write("""
        One of the most challenging thing in deep learning is lack of data. It is also one of our
        obstacle in this project. With our limited data, our accuray is 100 percent on the fall data with 17 percent on false alarm.
    """)

with st.beta_expander("Q : Can your algorithm know the different between the user falling and the phone falling?"):
    st.write("""
        Currently, no. The reason is that we have only user fall data to deveop an algorithm on.
        To do so, we need the data of the phone falling data to feed to our algorithm so that it can differenciate
        between the user falling and phone falling.
    """)

### ABOUT US ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>ABOUT US</h1>", unsafe_allow_html=True)
st.markdown(
"""
### Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
Team Members:""")
st.markdown(
"""
- Miri
    - I am cool dude
    - Connect with me here
""")
miri_image = Image.open('images/miri.jpg')
st.image(miri_image,
        width=300,
        unsafe_allow_html=True,
        )
st.markdown(
"""
- Jin
    - I am cool dude
    - Connect with me here
""")
jin_image = Image.open('images/jin.jpg')
st.image(jin_image,
        width=200,
        unsafe_allow_html=True,
        )
st.markdown(
"""
- Sven
    - I am cool dude
    - Connect with me here
""")

code = '''def thank_you():
   print("Thank you for visiting us!")'''
st.code(code, language='python')
