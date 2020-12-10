# Imports
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import altair as alt

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

#####################
##### GET DATA ######
#####################
@st.cache(suppress_st_warning=True)
def get_data():
    df = pd.read_excel('fallguys/data/demo_data.xlsx')
    return df

df = get_data()


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
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>PROBLEM</h1>", unsafe_allow_html=True)
st.markdown(
"""
###
- Fall can be a critical problem for the elderly social group.
- If the fall remains unnoticed, it can have significant impact on the health and lifestyle of the person.
- It’s a widespread problem with global elderly population rising due to declining fertility rate and increased longevity.
    - 23.3% of Japan’s population is over 65
    - Up to 50% of nursing home residents suffer from falls every year
    - 23% patients older than 65 suffer a trauma related-death after a fall

""")
#IMAGE/

charts = 'images/image1.png'
st.image(charts,
        width=750,
        unsafe_allow_html=True,
        )

charts1 = 'images/image2.png'
st.image(charts1,
        width=750,
        unsafe_allow_html=True,
        )

charts2 = 'images/image3.png'
st.image(charts2,
        width=750,
        unsafe_allow_html=True,
        )


### OUR SOLUTION ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>SOLUTION</h1>", unsafe_allow_html=True)
st.markdown(
    """
    We created a powerful algorithm that can identify and distinguish an event of fall among other activities of the user.
    """)

process = 'images/process.png'
st.image(process,
        use_column_width=True,
        unsafe_allow_html=True,
        )


### ZETEOH ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>DATA</h1>", unsafe_allow_html=True)
st.markdown(

"""

- We partnered with Zeteoh, a japanese company that offers options for augmented data hub to it's clients
- Detecting physical activities through smartphones
""")

st.markdown("<a style='color: white' href='https://www.zeteoh.com/'>Website: zeteoh</a>", unsafe_allow_html=True)

acceler = 'images/acceler2.png'
st.image(acceler,
        use_column_width=True,
        unsafe_allow_html=True,
        )


eda = 'images/eda.png'
st.image(eda,
        use_column_width=True,
        unsafe_allow_html=True,
        )

# st.markdown(
# """
# ### HOW
# -
# - Falling data
#     - We used falling data to expand their product and identify falling pattern..
#     - Final solution/product would be integrated to the app...
#     - why smart phone

# """)

### BEHIND THE SCENCES ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>BEHIND THE SCENES</h1>", unsafe_allow_html=True)

st.markdown(
"""
### Fall Detection Process Flow
1. Sensor Data is sent to Deep Learning [DL] Model
2. DL Model predicts: Fall / No Fall
3. In case of fall: Selected User are getting a notification on their smartphone


- How does this work?
    - Using accelerometer data to identify falling pattern
    - we used deep learning model .. to train and test our approach
        - what is a deep learning model
        - what model did we use (LSTM)
        - why LSTM
        - what results did we get

    - what were our main challenges?
        1. data preprocessing
        - insert image EDA
        - show data sample
        2. true positives/false negatives trade-off


""")

### PRODUCT ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>PRODUCT</h1>", unsafe_allow_html=True)
st.markdown(

"""

Our product is a powerful deep learning model, that has X accuracy in detecting an event of fall, using input form the sensor data.
    - Augmented data hub
    - Detecting physical activities through smartphones
""")



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

#################################### START CHART #####################################
counter = 0.01
mask = df['x_values_cum'] < counter
x = df[mask]['x_values_cum']
y1 = df[mask]['acc_x']
y2 = df[mask]['acc_y']
y3 = df[mask]['acc_z']


base = alt.Chart(df[mask]).properties(width=900, height=450).encode(alt.X('x_values_cum',axis=alt.Axis(title='Time in seconds')),
                                                                    alt.Y('acc_x' + ':Q',axis=alt.Axis(title='Accelerometer')))
base.configure_legend(
    strokeColor='gray',
    fillColor='#EEEEEE',
    padding=10,
    cornerRadius=10,
    orient='top-right'
)

# Chart Style
line_x_color = '#fd1916' # red
line_y_color = '#24af93' # green
line_z_color = '#003571' # blue
line_width = 2

fig = alt.layer(
    base.mark_line(color=line_x_color, size=line_width).encode(y='acc_x'),
    base.mark_line(color=line_y_color, size=line_width).encode(y='acc_y'),
    base.mark_line(color=line_z_color, size=line_width).encode(y='acc_z')
).configure(background='#a7d9f7')


line_chart = st.altair_chart(fig)

st.write('Status:')
progress_bar = st.progress(0)
activity_text = st.empty()
st.text("")
#Append more random data to the chart using add_rows
def animate():
    for i in range(1, 50):

        # Update progress bar.
        progress_bar.progress(i + 4)

        # Mask, values between previous timestamp and next timestamp (+1 second)
        mask = (df['x_values_cum'] >= counter) & (df['x_values_cum'] < counter + i)
        # mask = df['x_values_cum'] <= time_stamp
        line_chart.add_rows(df[mask])

        # Update status text.
        activity_text.markdown(f"<h2 style='text-align: left; color: white; text-shadow: 1.5px 1.5px #003571;padding:0'>Activity: {df[mask]['activity_name_map'].tail(1).iloc[0]}</h1>", unsafe_allow_html=True)

        # Refresh for x-amount of seconds
        time.sleep(0.3)

    st.balloons()


# Button
if st.button('Run Prediction 👩‍💻'):
    st.balloons()
    animate()

#################################### END CHART #####################################


### EXAMPLE DATA ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>EXAMPLE DATA</h1>", unsafe_allow_html=True)

# Display Dataframe
df_display = pd.read_excel('fallguys/data/demo_data.xlsx')
df_display_demo = df_display[["acc_x","acc_y","acc_z","activity_name"]].copy()
df_display_demo = df_display_demo.loc[0:30,:]
df = pd.DataFrame(df_display_demo)
st.dataframe(df)


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
    - I am cool cat lady
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
    - I fall for you.
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
