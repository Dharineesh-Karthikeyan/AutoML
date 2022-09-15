import streamlit as st
import numpy as np
#from PIL import Image

# Custom imports
from multipage import MultiPage
# import your pages here
from Pages import clf, intro, reg

# Create an instance of the app
st.set_page_config(
    page_title="AutoML Learning App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# An AutoML app"
    }
)
app = MultiPage()

# Title of the main page
#display = Image.open('Logo.png')
#display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
#col1, col2 = st.beta_columns(2)
#col1.image(display, width=400)

# Add all your application here
app.add_page("Introduction", intro.app)
app.add_page("Classification", clf.app)
app.add_page("Regression", reg.app)

# The main app
app.run()
