# Imports
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


#####################
##### GET DATA ######
#####################
# @st.cache(suppress_st_warning=True)
def get_data():
    df = pd.read_pickle('data/X_train.pkl')
    return df

df = get_data()
x = [i for i in range(1,99)]
acc_x = df['acc_x'][:99]
acc_y  = df['acc_y'][:99]
acc_z  = df['acc_z'][:99]



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

def init():  # give a clean slate to start
    acc_x_line.set_ydata([np.nan] * len(x))
    acc_y_line.set_ydata([np.nan] * len(x))
    acc_z_line.set_ydata([np.nan] * len(x))


def animate():  # update the y values (every 1000ms)
    for j in range(len(x)):
        progress_bar.progress(j+1)
        # Update status text.
        status_text.text(
            f'Duration: {j+1} seconds')
        acc_x_line.set_xdata(x[:j])
        acc_x_line.set_ydata(acc_x[:j])

        acc_y_line.set_xdata(x[:j])
        acc_y_line.set_ydata(acc_y[:j])

        acc_z_line.set_xdata(x[:j])
        acc_z_line.set_ydata(acc_z[:j])

        the_plot.pyplot(plt)
        prediction = random.choice(status)
        prediction_text.header(f'PREDICTION: {prediction}')

        time.sleep(1)

fig, ax = plt.subplots()
max_x = len(x)
max_rand = 10
ax.set_ylim(-15, max_rand)
acc_x_line, = ax.plot(x, np.random.randint(0, max_rand, max_x),color='#e0565f',linewidth=1)
acc_y_line, = ax.plot(x, np.random.randint(0, max_rand, max_x),color='#e0565f',linewidth=1)
acc_z_line, = ax.plot(x, np.random.randint(0, max_rand, max_x),color='#e0565f',linewidth=1)


# Set Styling for Plot
fig.patch.set_facecolor('#24A0ED')
ax.set_facecolor('#a7d9f7')
ax.grid(color='#24A0ED')
ax.tick_params(color='white', labelcolor='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#202020')


# Prediction Output
prediction_text = st.empty()

#Plot the Chart
the_plot = st.pyplot(plt)
status = ['fall', 'no_fall']
status_text = st.empty()
progress_bar = st.progress(0)

# Button
if st.button('Run Prediction 👩‍💻'):
    st.balloons()
    init()
    animate()




### EXAMPLE DATA ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>EXAMPLE DATA</h1>", unsafe_allow_html=True)

# Display Dataframe
df_display = pd.read_excel('data/demo_data.xlsx')
df = pd.DataFrame(df_display)
st.dataframe(df)




### HOW IT WORKS ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>HOW IT WORKS</h1>", unsafe_allow_html=True)
col1, col2 = st.beta_columns([3, 1])
data = np.random.randn(10, 1)
col1.subheader("A wide column with a chart")
col1.line_chart(data)
col2.subheader("A narrow column with the data")
col2.write(data)





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



### FAQ ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>FAQ</h1>", unsafe_allow_html=True)

with st.beta_expander("Questions 1"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

with st.beta_expander("Questions 2"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

with st.beta_expander("Questions 3"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

with st.beta_expander("Questions 4"):
    st.write("""
        The chart above shows some numbers I picked for you.
        I rolled actual dice for these, so they're *guaranteed* to
        be random.
    """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

### ABOUT US ###
st.markdown("---")
st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>ABOUT US</h1>", unsafe_allow_html=True)
st.markdown(
"""
### Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
Team Members:
- Miri
    - I am cool dude
    - Connect with me here
- Jin
    - I am cool dude
    - Connect with me here
- Sven
    - I am cool dude
    - Connect with me here
""")

code = '''def thank_you():
   print("Thank you for visiting us!")'''
st.code(code, language='python')
