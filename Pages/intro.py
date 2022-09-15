import streamlit as st


def app():
    title_cols1, title_cols2, title_cols3 = st.columns(3)
    with title_cols2:
        st.title("AutoML App")
    st.markdown("---------")
    st.markdown("Welcome to ML Learning App for beginners !")
    st.write("Hey there. This is a AutoML app for beginners to learn and practice ML algorithms and also fully automate their ML projects.")
    st.write("There are a few different models to select from and also performs tasks like classification and regression with ease with just some button clicks")
    st.write(
        "Select from the dropdown box to your left to start working with a ML task")
