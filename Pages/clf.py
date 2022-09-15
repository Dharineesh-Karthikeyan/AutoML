import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt


def app():
    title_cols1, title_cols2, title_cols3 = st.columns(3)
    with title_cols2:
        st.title("  Classification")
    st.markdown("---------")

    st.sidebar.markdown(''' To go to any particular part of the process, click here: 
- [Dataset Selection](#1-dataset-selection)
- [Target Variable and Feature Selection](#2-target-variable-and-feature-selection)
- [Train-test-split and Pre-processing data](#3-train-test-split-and-pre-processing-data)
- [Classifiers](#4-selecting-the-classifer)
- [Hyperparameters Tuning](#5-hyperparameters-tuning)
- [Plotting](#plotting)
- [Results](#accuracy)
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
                globals()["le_"+col] = LabelEncoder().fit(x_train.loc[:, col])
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

    def plot_metrics(metrics_list):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            fig = pp_matrix_from_data(y_test, y_pred)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            #fig, ax = plt.subplots()
            #fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
            #auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
            #ax.plot(fpr, tpr, label="data 1, auc="+str(auc))
            # ax.legend(loc=4)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

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
            "Select from one of the sample datasets", ("None", "Heart Disease", "Smart Grid Stability"), help="You can select any of the sample datasets to try the app out !!")
        if sample_db == "None":
            file = None
            st.markdown("No Dataset is selected...")
        elif sample_db == "Heart Disease":
            file = 'heart.csv'
            st.markdown("This Dataset has been selected...")
        elif sample_db == "Smart Grid Stability":
            file = 'smart_grid_stability.csv'
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
            "The target column must be categorical for classification task")

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
            "<h3 style='text-align:left; color: #FF6863;'>4) Selecting the classifer</h3>", unsafe_allow_html=True)
        classifier = st.selectbox(
            "Classifier", ("Logistic Regression", "Support Vector Machine (SVM)", "Random Forest"))

        # -- Logistic Regression --
        if classifier == "Logistic Regression":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            C = st.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0, key='C_LR')
            max_iter = st.slider(
                "Maximum number of iterations", 100, 500, key='max_iter')
            solver_ = st.selectbox(
                "Solver", ("newton-cg", "lbfgs", "liblinear", "sag", "saga"), index=1, help="For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss.‘liblinear’ is limited to one-versus-rest schemes.")
            if solver_ == "newton-cg":
                st.warning("Supported penalties are - l2,none")
            if solver_ == "lbfgs":
                st.warning("Supported penalties are - l2,none")
            if solver_ == "liblinear":
                st.warning("Supported penalties are - l1,l2")
            if solver_ == "sag":
                st.warning("Supported penalties are - l2,none")
            if solver_ == "saga":
                st.warning("Supported penalties are - elasticnet,l1,l2,none")
            penalty_ = st.selectbox(
                "Penalty", ("none", "l1", "l2", "elasticnet"), index=2)
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")

            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Plotting</h4>", unsafe_allow_html=True)
            metrics_list = st.multiselect(
                "Choose metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
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
                    "<h3 style='text-align:left; color: #FF6863;'>Logistic Regression Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = LogisticRegression(
                    C=C, max_iter=max_iter, penalty=penalty_, solver=solver_)
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
                metric1, metric2, metric3, metric4 = st.columns(4)
                with metric1:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Accuracy</h4>", unsafe_allow_html=True)
                    st.write(accuracy.round(2))
                with metric2:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Precision</h4>", unsafe_allow_html=True)
                    st.write(precision_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric3:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Recall</h4>", unsafe_allow_html=True)
                    st.write(recall_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric4:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>F1-Score</h4>", unsafe_allow_html=True)
                    st.write(f1_score(y_test, y_pred,
                             labels=class_names).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])
                #fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
                #auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
                plot_metrics(metrics_list)
                st.success('Finished processing')

        # -- Support Vector Machine --
        if classifier == "Support Vector Machine (SVM)":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            C = st.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, value=1.0, key='C')
            kernel = st.radio(
                "Kernel", ("rbf", "linear", "poly", "sigmoid"), key='kernel')
            if kernel == "poly":
                st.warning("Only for kernel polynomial")
                deg = st.number_input(
                    "Degree ?", 0, 10, step=1, value=3, key='deg')
            else:
                deg = 3
            gamma = st.radio(
                "Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Plotting</h4>", unsafe_allow_html=True)
            metrics = st.multiselect(
                "Choose metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

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

            if st.button("Classify", key='classify'):
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>Support Vector Machine (SVM) Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = SVC(C=C, kernel=kernel, degree=deg, gamma=gamma)
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
                # -- SVM Final Metrics --
                metric1, metric2, metric3, metric4 = st.columns(4)
                with metric1:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Accuracy</h4>", unsafe_allow_html=True)
                    st.write(accuracy.round(2))
                with metric2:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Precision</h4>", unsafe_allow_html=True)
                    st.write(precision_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric3:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Recall</h4>", unsafe_allow_html=True)
                    st.write(recall_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric4:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>F1-Score</h4>", unsafe_allow_html=True)
                    st.write(f1_score(y_test, y_pred,
                             labels=class_names).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])

                plot_metrics(metrics)
                st.success('Finished processing')

        # -- Random Forest --
        if classifier == "Random Forest":
            st.markdown(
                """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:left; color: #FF6863;'>5) Hyperparameters Tuning</h3>", unsafe_allow_html=True)
            n_estimators = st.number_input(
                "The number of trees in the forest", 0, 5000, step=10, value=100, key='n_estimators')
            criterion_ = st.radio(
                "Criterion", ("gini", "entropy"), key='criterion_')
            max_depth = st.number_input(
                "The maximum depth of the tree", 1, 100, step=1, key='max_depth')
            max_ft = st.radio(
                "Max Features", ("auto", "sqrt", "log2"), key='max_ft')
            bootstrap = st.radio(
                "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            if bootstrap == 'True':
                st.warning("Only if bootstrap is true")
                oob_sc = st.radio(
                    "Use out-of-bag samples to estimate generalization score?", ('True', 'False'), index=1, key='oob_sc')
            st.write(
                "If you want to know more about the parameters, or other parameters available - check out this [link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
            st.markdown(
                "<h4 style='text-align:left; color: #FF6863;'>Plotting</h4>", unsafe_allow_html=True)
            metrics = st.multiselect(
                "Choose metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

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

            if st.button("Classify", key='classify'):
                st.markdown(
                    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='text-align:left; color: #FF6863;'>Random Forest Results</h3>", unsafe_allow_html=True)
                my_bar = st.progress(0)
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, criterion=criterion_, max_features=max_ft, bootstrap=bootstrap, oob_score=oob_sc, n_jobs=-1)
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

                # -- LR Final Metrics --
                metric1, metric2, metric3, metric4 = st.columns(4)
                with metric1:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Accuracy</h4>", unsafe_allow_html=True)
                    st.write(accuracy.round(2))
                with metric2:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Precision</h4>", unsafe_allow_html=True)
                    st.write(precision_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric3:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Recall</h4>", unsafe_allow_html=True)
                    st.write(recall_score(y_test, y_pred,
                             labels=class_names).round(2))
                with metric4:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>F1-Score</h4>", unsafe_allow_html=True)
                    st.write(f1_score(y_test, y_pred,
                             labels=class_names).round(2))
                if flag:
                    st.markdown(
                        "<h4 style='text-align:left; color: #e75480;'>Prediction for the given values is:</h4>", unsafe_allow_html=True)
                    if inv_needed:
                        pred = le_target.inverse_transform(pred)
                        st.write(pred[0])
                    else:
                        st.write(pred[0])

                plot_metrics(metrics)
                st.success('Finished processing')


# app()
