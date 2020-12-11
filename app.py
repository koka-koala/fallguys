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
import plotly.graph_objects as go

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


@st.cache
def load_data_population():
    df = pd.read_excel('fallguys/data/Japan-1950-2020.xlsx')
    return df

df_pop = load_data_population()

####################
### PAGE LAYOUT ####
####################
# Header Section
# Slide 1
with st.beta_container():
    image = Image.open('images/header.png')
    st.image(image,
            use_column_width=True,
            unsafe_allow_html=True,
            )


### OUR SOLUTION ###
st.markdown("---")

with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>PROBLEM</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    ###
    - Fall can be a critical problem for the elderly demographic group.
    - If the fall remains unnoticed, it can have significant impact on the health and lifestyle of the person.
    - It’s a widespread problem with global elderly population rising due to a declining fertility rate and increased longevity.
        - Up to 50% of nursing home residents suffer from falls every year
        - 23% patients older than 65 suffer a trauma-related death after a fall
        - 28.5% of Japan’s population is over 65 in 2020

    """)
    # #IMAGE/
    # st.write('Population of Japan by age and sex in 2015')
    # charts1 = 'images/image2.png' #current situation
    # st.image(charts1,
    #         width=500,
    #         unsafe_allow_html=True,
    #         caption='Source: https://voxeu.org/article/japan-s-age-wave-challenges-and-solutions'
    #         )


    year = st.slider('Select Year:', 1950, 2020)
    st.text("")
    mask = df_pop['Year'] == year

    y = df_pop[mask]['Age']
    x1 = df_pop[mask]['M']
    x2_label = df_pop[mask]['F']
    x2 = df_pop[mask]['F'] * -1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y,
        x=x1,
        name='Male',
        orientation='h',
        marker=dict(
            color='rgba(0, 53, 113, 0.6)',
            line=dict(color='rgba(0, 53, 113, 1.0)', width=3)
        )
    ))
    fig.add_trace(go.Bar(
        y=y,
        x=x2,

        name='Female',
        orientation='h',
        marker=dict(
            color='rgba(255, 100, 151, 0.6)',
            line=dict(color='rgba(255, 100, 151, 1.0)', width=3)
        )
    ))

    fig.update_layout(
        width=650,
        height=300,
        margin=dict(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=40, #top margin
            ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=f'<b>Population Pyramid Japan in {year}</b>',
        font=dict(
            color="white"
            ),
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Age Group',
            titlefont_size=16,
            tickfont_size=14,
            showgrid=False,
            gridwidth=0.2,
            gridcolor='#d3d3d3',
        ),
        xaxis=dict(
            title='Population in Mio',
            titlefont_size=16,
            tickfont_size=14,
            showgrid=False,
            gridwidth=0.2,
            gridcolor='white',
            tickvals=[-4000000, -2000000,0,2000000,4000000],
            ticktext=["4M", "2M","0","2M","4M"],
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='relative',
        bargap=0.0, # gap between bars of adjacent location coordinates.
        bargroupgap=0 # gap between bars of the same location coordinate.
    )
    st.plotly_chart(fig)
    st.markdown('*Datasource: https://www.populationpyramid.net/japan/*')

### OUR SOLUTION ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>SOLUTION</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        We created a powerful algorithm that can identify and distinguish an event of fall among other activities of the user.
        """)
    st.markdown(
        """
        The aim for the algorithm is to continuosly track our users' every day activity, and in the case of a fall event, send a notification to a dedicated contact for immediate assistance.
        """)

    process = 'images/process.png'
    st.image(process,
            use_column_width=True,
            unsafe_allow_html=True,
            )

### DATA ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>DATA</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    - Our solution relies on accelerometer sensor data, collected by smartphones.
    - Most smartphones contain built-in accelerometers and can support apps that use its data.
    - We partnered with <a style='color: white; text-decoration: underline' href='https://www.zeteoh.com/' target="_blank">Zeteoh</a>,
    a Japanese company that deploys deep learning models on small devices.
    - One of their products focuses on detecting physical activities on smartphone for gaming, insurance and healthcare industries.
    """, unsafe_allow_html=True)

    acceler = 'images/acceler2.png'
    with st.beta_expander("How does Accelerometer Sensor data look like?"):
        st.image(acceler,
                use_column_width=True,
                width=700,
                unsafe_allow_html=True,
                caption='X, Y and Z axes correspond to the motion of your phone in three-dimensional space.'
                )


### BEHIND THE SCENCES ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>MODEL</h1>", unsafe_allow_html=True)

    st.markdown(
    """
    ### Model : Deep learning

    Type : Long-Short Term Memory (LSTM)

    - Can process entire sequences of data and appropriate for time series
    - Tested best in the evaluation stage

    Metrics/Result detecting fall event → Fall 100% : Others 17%

    Challenge : Limited dataset
    """)

st.markdown("---")
with st.beta_container():

    st.markdown(
    """
    ### Visualization of Data Stream
    """)

    #################################### START CHART #####################################
    counter = 0.01
    mask = df['x_values_cum'] < counter
    x = df[mask]['x_values_cum']
    y1 = df[mask]['acc_x']
    y2 = df[mask]['acc_y']
    y3 = df[mask]['acc_z']


    base = alt.Chart(df[mask]).properties(width=600, height=300).encode(alt.X('x_values_cum',axis=alt.Axis(title='Time in seconds')),
                                                                        alt.Y('acc_x' + ':Q',axis=alt.Axis(title='Accelerometer'))).properties(title='')
    base.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-right'
    )

    # Chart Style
    line_x_color = '#DC143C' # red
    line_y_color = '#17705e' # green
    line_z_color = '#003571' # blue
    line_width = 2

    fig = alt.layer(
        base.mark_line(color=line_x_color, size=line_width).encode(y='acc_x'),
        base.mark_line(color=line_y_color, size=line_width).encode(y='acc_y'),
        base.mark_line(color=line_z_color, size=line_width).encode(y='acc_z')
    ).configure(background='#a7d9f7')


    line_chart = st.altair_chart(fig)


    col1_header, col2_header, col3_header = st.beta_columns(3)
    with col1_header:
        st.subheader("**What the accelerometer measures:**")
    with col2_header:
        st.subheader("**What the user is doing:**")
    with col3_header:
        st.subheader("**What our model predicts:**")

    col1_display, col2_display, col3_display = st.beta_columns(3)
    with col1_display:
        col1_display = st.empty()
    with col2_display:
        col2_display = st.empty()
    with col3_display:
        col3_display = st.empty()

    col1_display1, col2_display1, col3_display1 = st.beta_columns(3)
    with col1_display1:
        col1_display1 = st.empty()
    with col2_display1:
        col2_display1 = st.empty()
    with col3_display1:
        col3_display1 = st.empty()

    col1_display2, col2_display2, col3_display2 = st.beta_columns(3)
    with col1_display2:
        col1_display2 = st.empty()
    with col2_display2:
        col2_display2 = st.empty()
    with col3_display2:
        col3_display2 = st.empty()


    # Load Icons to display in animation
    downstairs = Image.open('icons/stairs_down.png')
    upstairs = Image.open('icons/stairs_up.png')
    cycling = Image.open('icons/cycling.png')
    falling = Image.open('icons/fall.png')
    check = Image.open('icons/check.png')
    warning = Image.open('icons/warning.png')


    # st.write('Status:')
    # progress_bar = st.progress(0)
    activity_text = st.empty()
    st.text("")
    #Append more random data to the chart using add_rows
    def animate():
        for i in range(1, 52):

            # Update progress bar.
            # progress_bar.progress(i + 48)

            # Mask, values between previous timestamp and next timestamp (+1 second)
            mask = (df['x_values_cum'] >= counter) & (df['x_values_cum'] < counter + i)
            # mask = df['x_values_cum'] <= time_stamp
            line_chart.add_rows(df[mask])

            acc_x_data = df[mask]['acc_x'].tail(1).iloc[0]
            acc_y_data = df[mask]['acc_y'].tail(1).iloc[0]
            acc_z_data = df[mask]['acc_z'].tail(1).iloc[0]

            activity = df[mask]['activity_name_map'].tail(1).iloc[0]
            prediction = df[mask]['prediction'].tail(1).iloc[0]

            # Update Sensor Data
            col1_display.markdown(f"<font color={line_x_color}>x-axis: {acc_x_data}</font>",unsafe_allow_html=True)

            # Update activity
            col2_display.markdown(f"{activity}")

            # Update Sensor Data
            col3_display.markdown(f"{prediction}")


            ### ROW 1 ###
            # Update Sensor Data
            col1_display1.markdown(f"<font color={line_y_color}>y-axis: {acc_y_data}</font>",unsafe_allow_html=True)


            ### ROW 2 ###
            col1_display2.markdown(f"<font color={line_z_color}>z-axis: {acc_z_data}</font>",unsafe_allow_html=True)

            # Function to disc
            def display_icon_activity(activity, columnnumber):
                if activity == 'upstairs':
                    activity = upstairs
                if activity == 'downstairs':
                    activity = downstairs
                if activity == 'cycling':
                    activity = cycling
                if activity == 'falling':
                    activity = falling
                if activity == 'fall':
                    activity = warning
                if activity == 'no fall':
                    activity = check
                if columnnumber == 2:
                    icon = col2_display1.image(activity,
                                    use_column_width=False,
                                    unsafe_allow_html=True,
                                    width=25,
                                    height=25)
                if columnnumber == 3:
                    icon = col3_display1.image(activity,
                                    use_column_width=False,
                                    unsafe_allow_html=True,
                                    width=25,
                                    height=25)
                return icon

            if activity == 'Climbing Stairs':
                display_icon_activity('upstairs',2)
            if activity == 'Going Down Stairs':
                display_icon_activity('downstairs',2)
            if activity == 'Cycling':
                display_icon_activity('cycling',2)
            if activity == 'Falling':
                display_icon_activity('falling',2)
            if prediction == 'FALL':
                display_icon_activity('fall',3)
                col3_display2.markdown('<span style="color:#DC143C;font-weight: bold;">Notfication sent!</span>',unsafe_allow_html=True)
            if prediction == 'No Fall':
                display_icon_activity('no fall',3)

            # Refresh for x-amount of seconds
            time.sleep(0.3)


    # Button
    if st.button('Show Data Stream'):
        animate()

#################################### END CHART ###################


### PRODUCT ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>DEPLOYMENT</h1>", unsafe_allow_html=True)

    st.markdown(

    """
    Our model was designed to work best on smartphones, but can also be adapted and integrated to other
    environments that can provide accelerometer sensor data input.
    """)
    image_deploy = Image.open('images/deploy.png')
    st.image(image_deploy,
            width=500,
            unsafe_allow_html=True,
            )


### ABOUT US ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>ABOUT US</h1>", unsafe_allow_html=True)
    # Setting up card template
    def contact_card(
        name="Name",
        title="Python Expert",
        photo_url="https://filmshotfreezer.files.wordpress.com/2011/07/untitled-1.jpg",
        banner_url="https://snap-photos.s3.amazonaws.com/img-thumbs/960w/RQ2Z75PQIN.jpg",
        contact_url="",
        description=""
        ):
        return f"""
            <div class="card">
                <div class="photo" style="background-image: url({photo_url});"></div>
                <div class="banner" style="background-image: url({banner_url});"></div>
                <ul>
                    <li><b>{name}</b></li>
                    <li>{title}</li>
                </ul>
                <a href="{contact_url}" target="_blank">
                    <button class="contact" id="main-button">click to get in touch</button>
                </a>
                <div class="description">{description}</div>
            </div>"""
    # Information for the team. Please use photos hosted externally, e.g. google drive
    # Convert google drive link to useable link:
    # https://stackoverflow.com/questions/10311092/displaying-files-e-g-images-stored-in-google-drive-on-a-website
    team = [
        {
            "name": "Jin Khokasai",
            "title": "Snake lover :) ",
            "description": "“When all else fails... reboot”",
            "photo_url": "https://drive.google.com/uc?export=view&id=1YFOd_5XyN6LHFytORIuMwidBSFUq8wGQ",
            "banner_url": "https://drive.google.com/uc?export=view&id=1DCM6WGnNrSVnEBNJs0VzC9PfkbH7CIxK",
            "contact_url": "https://www.linkedin.com/in/jin-khokasai/"
        },
        {
            "name": "Miri Nikolic",
            "title": "Survival Expert",
            "description": "“180 days quarantine challenge”",
            "photo_url": "https://drive.google.com/uc?export=view&id=1d_DFuSR4BkG4pgEeL_kcLtBvTUsGQ5xc",
            "banner_url": "https://drive.google.com/uc?export=view&id=1DCM6WGnNrSVnEBNJs0VzC9PfkbH7CIxK",
            "contact_url": "https://www.linkedin.com/in/mirnikolic/"
        },
        {
            "name": "Sven Bosau",
            "title": "Data Science Enthusiast",
            "description": "“Life is short, use Python”",
            "photo_url": "https://drive.google.com/uc?export=view&id=1tZhtkdr-VKPgEc7Oo_bUfSeh47CYDCRV",
            "banner_url": "https://drive.google.com/uc?export=view&id=1DCM6WGnNrSVnEBNJs0VzC9PfkbH7CIxK",
            "contact_url": "https://www.linkedin.com/in/sven-bosau-092837201/"
        }
    ]
    # Inserting the cards in the page
    st.markdown(f'<div class="cards">{"".join([contact_card(**member) for member in team])}</div>', unsafe_allow_html=True)

### FAQ ###
st.markdown("---")
with st.beta_container():
    st.markdown("<h1 style='text-align: center; color: white; text-shadow: 1.5px 1.5px #003571'>FAQs</h1>", unsafe_allow_html=True)

    with st.beta_expander("Q : What happens when someone falls?"):
        st.write("""
            Good question! Our model traces users' everyday activities and can tell if that someone fell. In that case,
            a notification can be send to a dedicated contact to request assistance.
        """)

    with st.beta_expander("Q : What happens when the user did not fall but somehow the algorithm think they fell?"):
        st.write("""
            We plan to have a pop-up messsage to check if the user actually fell or not. If the user does not respond in
            e.g. 5 minutes the system will send the notification! Otherwise it will discard the notification.
        """)

    with st.beta_expander("Q : What kind of model did you use to develop the algorithm?"):
        st.write("""
            Maybe you miss the part above, We use RNN (Recurring Neuron Network) - LSTM (Long Short Term Memory)
            to develop the algorithm.
        """)

    with st.beta_expander("Q : What is your accuracy on detecting fall pattern?"):
        st.write("""
            One of the most challenging things in deep learning is lack of data. This was also one of our
            obstacles in this project. With our limited data, our accuracy is 100 percent detecting actual fall data and 17 percent being false alarm.
        """)

    with st.beta_expander("Q : Can your algorithm know the difference between the user falling and the phone falling?"):
        st.write("""
            Currently, no. This is because we only have user fall data to deveop the algorithm with.
            To be able to detect phone falling, we need specific data to distinguish phones from users.
        """)
    with st.beta_expander("Q : What did you find challenging in this project?"):
        st.write("""
            We had limited time and limited data to work on. A bigger (dataset) is always better for model accuracy.
        """)



