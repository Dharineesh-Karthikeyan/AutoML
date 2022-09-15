import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import time
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt


def app():
    title_cols1, title_cols2, title_cols3 = st.columns(3)
    with title_cols2:
        st.title("  Regression")
    st.markdown("---------")

    st.sidebar.markdown(''' To go to any particular part of the process, click here:
- [Dataset Selection](#1-dataset-selection)
- [Target Variable and Feature Selection](#2-target-variable-and-feature-selection)
- [Train-test-split and Pre-processing data](#3-train-test-split-and-pre-processing-data)
- [Classifiers](#4-selecting-the-classifer)
- [Hyperparameters Tuning](#5-hyperparameters-tuning)
- [Metrics](#metrics)
''', unsafe_allow_html=True)

    @st.cache(persist=True)
    def load_data(file):
        data = pd.read_csv('SampleData/'+file)
        return data

    @st.cache(persist=True)
    def data_split(df, target, split, normalize_by):
        y = df[target]
        x = df.drop(columns=[target])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=split, random_state=0)
        x_train_before = x_train.copy()

        # -- Label Encoders --
        label_enc = []
        if(df[target].dtype == "object"):
            le_target = LabelEncoder(
            ).fit(y_train)
            y_train = le_target.transform(
                y_train)
            y_test = le_target.transform(
                y_test)

        for col in df.columns:
            if(df[col].dtype == "object" and col != target):
                globals()["le_"+col] = LabelEncoder().fit(x.loc[:, col])
                label_enc.append("le_"+col)

        for col in df.columns:
            for le in range(len(label_enc)):
                if(col in label_enc[le]):
                    x_train.loc[:, col] = globals()[label_enc[le]
                                                    ].transform(x_train.loc[:, col])
                    x_test.loc[:, col] = globals()[label_enc[le]
                                                   ].transform(x_test.loc[:, col])

        if normalize_by == "MinMax Scalar":
            normalizer = MinMaxScaler().fit(x_train)
            x_train = normalizer.transform(x_train)
            x_test = normalizer.transform(x_test)

        else:
            normalizer = StandardScaler().fit(x_train)
            x_train = normalizer.transform(x_train)
            x_test = normalizer.transform(x_test)

        return x_train_before, x_train, x_test, y_train, y_test, normalizer

    # -- Dataset Selection --
    st.markdown(
        "<h3 style='text-align:left; color: #FF6863;'>1) Dataset Selection</h3>", unsafe_allow_html=True)
    upload_status = False
    upload_file = st.file_uploader("Upload CSV Data", type=["csv"])
    if upload_file is not None:
        file_details = {"filename": upload_file.name, "filetype": upload_file.type,
                        "filesize": upload_file.size}
        st.write(file_details)
        df = pd.read_csv(upload_file)
        st.markdown("This Dataset has been selected...")
        upload_status = True

    if upload_status == False:
        st.markdown("OR")
        file = None
        sample_db = st.selectbox(
            "Select from one of the sample datasets", ("None", "Car Sales", "Real Estate"), help="You can select any of the sample datasets to try the app out !!")
        if sample_db == "None":
            file = None
            st.markdown("No Dataset is selected...")
        elif sample_db == "Car Sales":
            file = 'car sales.csv'
            st.markdown("This Dataset has been selected...")
        elif sample_db == "Real Estate":
            file = 'real estate.csv'
            st.markdown("This Dataset has been selected...")
        if(file != None):
            df = load_data(file)

    # Loading the data and feature selection
    if(upload_status == True or file != None):
        cols_name = df.columns
        if st.checkbox("Show the data", False):
            st.subheader("A Look At The Data")
            with st.spinner('Working on it...Please wait'):
                st.write(df)
            st.info('This is the whole data')
        st.markdown(
            """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        # -- Target Selection --
        st.markdown(
            "<h3 style='text-align:left; color: #FF6863;'>2) Target Variable and Feature Selection</h3>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align:left; color: #e75480;'>Target Variable</h4>", unsafe_allow_html=True)
        st.markdown(
            "Select the column which is the target variable (to be predicted)")
        target = st.selectbox("Target Variable Column",
                              cols_name, index=len(cols_name)-1)
        st.warning(
            "The target column must be numerical for regression task")

        # -- Feature Selection --
        st.markdown(
            "<h4 style='text-align:left; color: #e75480;'>Feature Selection</h4>", unsafe_allow_html=True)
        st.markdown("Select all the columns you want to predictor variable")
        all = st.checkbox("Select all columns")
        predictors = cols_name.copy()
        if all == False:
            selected_features = st.multiselect(
                "Select all the required features", predictors)
            if len(selected_features) != 0:
                selected_features.append(target)
                df = df[selected_features]
                with st.spinner('Working on it...Please wait'):
                    st.write(df)
                    st.info('This is the feature selected data')
                    st.info(
                        'NOTE: The target column is placed as the last column by default')
        else:
            selected_features = cols_name
        st.markdown(
            """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        # -- Train-test split --
        st.markdown(
            "<h3 style='text-align:left; color: #FF6863;'>3) Train-test split and Pre-processing data</h3>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align:left; color: #e75480;'>Train-test split</h4>", unsafe_allow_html=True)
        split = st.number_input(
            "Train-test split value", 0.1, 0.9, step=0.05, key='split')

        # -- Pre-processing --
        st.markdown(
            "<h4 style='text-align:left; color: #e75480;'>Pre-processing</h4>", unsafe_allow_html=True)
        st.markdown("Label Encoding and Normalization is applied to the data")
        st.markdown(
            "Label Encoding is performed here since OneHotEncoding tends to increase the number of features.\n")
        st.markdown("")
        normalize_by = st.selectbox(
            "Select one of the normalization methods", ("MinMax Scalar", "Standard Scalar"))
        # -- Train-test split Continues --
        x_train_before, x_train, x_test, y_train, y_test, normalizer = data_split(
            df, target, split, normalize_by)
        class_names = df[target].unique()
        # -- Pre-Processing Continues --
        if st.checkbox("Show data after pre-processing", False):
            before_pre, after_pre = st.columns(2)
            with st.spinner('Working on it...Please wait'):
                with before_pre:
                    st.subheader("Before Pre-Processing")
                    st.write(x_train_before)
                with after_pre:
                    st.subheader("After Pre-Processing")
                    st.write(x_train)
            st.info('Note: This is only X_train data')
        st.markdown(
            """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        # -- Classifiers --
        st.markdown(
            "<h3 style='text-align:left; color: #FF6863;'>4) Selecting the Regression</h3>", unsafe_allow_html=True)
        regressor = st.selectbox(
            "Regression", ("Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression"))

        # -- Linear Regression --
        if regressor == "Linear Regression":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            st.write(
                "Linear Regression has very few parameters, nothing major. \nIf you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Metrics</h4>", unsafe_allow_html=True)
            metrics_list = st.multiselect(
                "Choose metrics", ('R2-Score', 'Mean-Absolute Error', 'Mean Squared Error'))
            # -- Prediction --
            prediction = st.checkbox("Would you like to add an own input for the model to predict?",
                                     help="The model performance is measured by the test set results. You can add this input as an additional value for the model to predict on !")
            if prediction:
                pred_data = []
                for pred_col in df.columns:
                    if(df[pred_col].dtype == "object" and pred_col != target):
                        value = st.selectbox(
                            label=pred_col, options=df[pred_col].unique())
                        le = LabelEncoder().fit(x_train_before[pred_col])
                        value = le.transform(
                            np.array(value).reshape(1, -1))
                        pred_data.append(value)

                    if(df[pred_col].dtype != "object" and pred_col != target):
                        value = st.number_input(label=pred_col)
                        pred_data.append(value)
                if df[target].dtype == "object":
                    le_target = LabelEncoder().fit(df[target])
                    inv_needed = True
                else:
                    inv_needed = False

                pred_data = normalizer.transform(
                    np.array(pred_data).reshape(1, -1))
                flag = True
            else:
                flag = False

            classify = st.button("Classify", key='classify')
            if classify:
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>Linear Regression Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = LinearRegression()
                my_bar.progress(25)
                model.fit(x_train, y_train)
                my_bar.progress(50)
                accuracy = model.score(x_test, y_test)
                my_bar.progress(75)
                y_pred = model.predict(x_test)
                if flag:
                    pred = model.predict(pred_data)
                time.sleep(0.5)
                my_bar.progress(100)

                # -- LR Final Metrics --
                metric1, metric2, metric3 = st.columns(3)
                if 'R2-Score' in metrics_list:
                    with metric1:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>R2-Score</h4>", unsafe_allow_html=True)
                        st.write(r2_score(y_test, y_pred).round(2))
                if 'Mean-Absolute Error' in metrics_list:
                    with metric2:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Absolute Error</h4>", unsafe_allow_html=True)
                        st.write(mean_absolute_error(y_test, y_pred).round(2))
                if 'Mean Squared Error' in metrics_list:
                    with metric3:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Squared Error</h4>", unsafe_allow_html=True)
                        st.write(mean_squared_error(y_test, y_pred).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])
                st.success('Finished processing')

        # -- Ridge Regression --
        if regressor == "Ridge Regression":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            alpha = st.number_input(
                "Alpha", 1.0, 100.0, step=0.1, value=1.0, key='alpha')
            max_iter = st.slider(
                "Maximum Iterations", 1000, 5000, key='max_iter')
            solver = st.radio(
                "Solver", ("auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"), key='solver')
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Metrics</h4>", unsafe_allow_html=True)
            metrics_list = st.multiselect(
                "Choose metrics", ('R2-Score',
                                   'Mean-Absolute Error', 'Mean Squared Error'))

            # -- Prediction --
            prediction = st.checkbox("Would you like to add an own input for the model to predict?",
                                     help="The model performance is measured by the test set results. You can add this input as an additional value for the model to predict on !")
            if prediction:
                pred_data = []
                for pred_col in df.columns:
                    if(df[pred_col].dtype == "object" and pred_col != target):
                        value = st.selectbox(
                            label=pred_col, options=df[pred_col].unique())
                        le = LabelEncoder().fit(x_train_before[pred_col])
                        value = le.transform(
                            np.array(value).reshape(1, -1))
                        pred_data.append(value)

                    if(df[pred_col].dtype != "object" and pred_col != target):
                        value = st.number_input(label=pred_col)
                        pred_data.append(value)

                if df[target].dtype == "object":
                    le_target = LabelEncoder().fit(df[target])
                    inv_needed = True
                else:
                    inv_needed = False

                pred_data = normalizer.transform(
                    np.array(pred_data).reshape(1, -1))

                flag = True
            else:
                flag = False

            if st.button("Predict", key='classify'):
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>Ridge Regression Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = Ridge(max_iter=max_iter, alpha=alpha, solver=solver)
                my_bar.progress(25)
                model.fit(x_train, y_train)
                my_bar.progress(50)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                my_bar.progress(75)
                if flag:
                    pred = model.predict(pred_data)
                time.sleep(0.5)
                my_bar.progress(100)
                # -- Ridge Final Metrics --
                metric1, metric2, metric3 = st.columns(3)
                if 'R2-Score' in metrics_list:
                    with metric1:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>R2-Score</h4>", unsafe_allow_html=True)
                        st.write(r2_score(y_test, y_pred).round(2))
                if 'Mean-Absolute Error' in metrics_list:
                    with metric2:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Absolute Error</h4>", unsafe_allow_html=True)
                        st.write(mean_absolute_error(y_test, y_pred).round(2))
                if 'Mean Squared Error' in metrics_list:
                    with metric3:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Squared Error</h4>", unsafe_allow_html=True)
                        st.write(mean_squared_error(y_test, y_pred).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])
                st.success('Finished processing')

        # -- Lasso Regression --
        if regressor == "Lasso Regression":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            alpha = st.number_input(
                "Alpha", 1.0, 100.0, step=0.1, value=1.0, key='alpha')
            max_iter = st.slider(
                "Maximum Iterations", 1000, 5000, key='max_iter')
            selection = st.radio(
                "Selection", ("cyclic", "random"), key='selection')
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Metrics</h4>", unsafe_allow_html=True)
            metrics_list = st.multiselect(
                "Choose metrics to plot", ('R2-Score', 'Mean-Absolute Error', 'Mean Squared Error'))

            # -- Prediction --
            prediction = st.checkbox("Would you like to add an own input for the model to predict?",
                                     help="The model performance is measured by the test set results. You can add this input as an additional value for the model to predict on !")
            if prediction:
                pred_data = []
                for pred_col in df.columns:
                    if(df[pred_col].dtype == "object" and pred_col != target):
                        value = st.selectbox(
                            label=pred_col, options=df[pred_col].unique())
                        le = LabelEncoder().fit(x_train_before[pred_col])
                        value = le.transform(
                            np.array(value).reshape(1, -1))
                        pred_data.append(value)

                    if(df[pred_col].dtype != "object" and pred_col != target):
                        value = st.number_input(label=pred_col)
                        pred_data.append(value)

                if df[target].dtype == "object":
                    le_target = LabelEncoder().fit(df[target])
                    inv_needed = True
                else:
                    inv_needed = False

                pred_data = normalizer.transform(
                    np.array(pred_data).reshape(1, -1))
                flag = True
            else:
                flag = False

            if st.button("Predict", key='classify'):
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>Lasso Regression Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = Lasso(
                    alpha=alpha, max_iter=max_iter, selection=selection)
                my_bar.progress(25)
                model.fit(x_train, y_train)
                my_bar.progress(50)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                my_bar.progress(75)
                if flag:
                    pred = model.predict(pred_data)
                time.sleep(0.5)
                my_bar.progress(100)

                # -- Lasso Final Metrics --
                metric1, metric2, metric3 = st.columns(3)
                if 'R2-Score' in metrics_list:
                    with metric1:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>R2-Score</h4>", unsafe_allow_html=True)
                        st.write(r2_score(y_test, y_pred).round(2))
                if 'Mean-Absolute Error' in metrics_list:
                    with metric2:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Absolute Error</h4>", unsafe_allow_html=True)
                        st.write(mean_absolute_error(y_test, y_pred).round(2))
                if 'Mean Squared Error' in metrics_list:
                    with metric3:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Squared Error</h4>", unsafe_allow_html=True)
                        st.write(mean_squared_error(y_test, y_pred).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])

                st.success('Finished processing')

        # -- Elastic Net Regression --
        if regressor == "Elastic Net Regression":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            alpha = st.number_input(
                "Alpha", 1.0, 100.0, step=0.1, value=1.0, key='alpha')
            l1_ratio = st.number_input(
                "L1 Ratio", 0.0, 1.0, step=0.1, value=0.5, key='l1_ratio')
            max_iter = st.slider(
                "Maximum Iterations", 1000, 5000, key='max_iter')

            selection = st.radio(
                "Selection", ("cyclic", "random"), key='selection')
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Metrics</h4>", unsafe_allow_html=True)
            metrics_list = st.multiselect(
                "Choose metrics to plot", ('R2-Score', 'Mean-Absolute Error', 'Mean Squared Error'))

            # -- Prediction --
            prediction = st.checkbox("Would you like to add an own input for the model to predict?",
                                     help="The model performance is measured by the test set results. You can add this input as an additional value for the model to predict on !")
            if prediction:
                pred_data = []
                for pred_col in df.columns:
                    if(df[pred_col].dtype == "object" and pred_col != target):
                        value = st.selectbox(
                            label=pred_col, options=df[pred_col].unique())
                        le = LabelEncoder().fit(x_train_before[pred_col])
                        value = le.transform(
                            np.array(value).reshape(1, -1))
                        pred_data.append(value)

                    if(df[pred_col].dtype != "object" and pred_col != target):
                        value = st.number_input(label=pred_col)
                        pred_data.append(value)

                if df[target].dtype == "object":
                    le_target = LabelEncoder().fit(df[target])
                    inv_needed = True
                else:
                    inv_needed = False

                pred_data = normalizer.transform(
                    np.array(pred_data).reshape(1, -1))
                flag = True
            else:
                flag = False

            if st.button("Predict", key='classify'):
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>ElasticNet Regression Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = ElasticNet(
                    alpha=alpha, max_iter=max_iter, selection=selection, l1_ratio=l1_ratio)
                my_bar.progress(25)
                model.fit(x_train, y_train)
                my_bar.progress(50)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                my_bar.progress(75)
                if flag:
                    pred = model.predict(pred_data)
                time.sleep(0.5)
                my_bar.progress(100)

                # -- Lasso Final Metrics --
                metric1, metric2, metric3 = st.columns(3)
                if 'R2-Score' in metrics_list:
                    with metric1:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>R2-Score</h4>", unsafe_allow_html=True)
                        st.write(r2_score(y_test, y_pred).round(2))
                if 'Mean-Absolute Error' in metrics_list:
                    with metric2:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Absolute Error</h4>", unsafe_allow_html=True)
                        st.write(mean_absolute_error(y_test, y_pred).round(2))
                if 'Mean Squared Error' in metrics_list:
                    with metric3:
                        st.markdown(
                            "<h4 style='text-align:left; color: #e75480;'>Mean-Squared Error</h4>", unsafe_allow_html=True)
                        st.write(mean_squared_error(y_test, y_pred).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])

                st.success('Finished processing')


# app()
