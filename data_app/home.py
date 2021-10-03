import streamlit as st

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
        <style>
        body {
        # background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
    print("background")
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def app():
    st.title('Home')

    st.write('This is the `home page` of this multi-page app.')

    st.write('In this app, we will be building a simple classification model using the Iris dataset.')

    # set_png_as_page_bg('data_app/background.jpg')
