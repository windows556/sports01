import streamlit as st
from multiapp import MultiApp
from data_app import home, intro, model
import json
import requests

app = MultiApp()

# Add all your application here
# app.add_app("Home", home.app)
app.add_app("Introduction", intro.app)
app.add_app("Model", model.app)
# The main app
app.run()


# st.markdown("""
# # Multi-Page App

# This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed by [Praneel Nihar](https://medium.com/@u.praneel.nihar). 
# Also check out his [Medium article](https://medium.com/@u.praneel.nihar/building-multi-page-web-app-using-streamlit-7a40d55fa5b4).

# """)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

# lottie_coding = load_lottiefile("lottiefile.json")  # replace link to local lottie file
# lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

# st_lottie(
#     lottie_coding,
#     speed=1,
#     reverse=False,
#     loop=True,
#     quality="low", # medium ; high
#     renderer="svg", # canvas
#     height=None,
#     width=None,
#     key=None,
# )

# st_lottie(
#     lottie_hello,
#     speed=1,
#     reverse=False,
#     loop=True,
#     quality="low", # medium ; high
#     renderer="svg", # canvas
#     height=None,
#     width=None,
#     key=None,
# )
