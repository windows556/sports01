import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn_lvq import GlvqModel
from sklearn_lvq.utils import plot2d

# st.set_page_config(layout="wide")
def app():

    df = pd.read_csv("./data/EPL.csv")
    filtered = []
    filtered_column_list = []

    normalize = True

    # st.dataframe(dataframe.style.highlight_max(axis=0))
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    # nb_ppc = 100
    # print('GLVQ:')
    # toy_data = np.append(
    #     np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=nb_ppc),
    #     np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=nb_ppc), axis=0)
    # toy_label = np.append(np.zeros(nb_ppc), np.ones(nb_ppc), axis=0)

    # glvq = GlvqModel()
    # glvq.fit(toy_data, toy_label)
    # # st.pyplot(plot2d(glvq, toy_data, toy_label, 1, 'glvq'))
    # fig = plot2d(glvq, toy_data, toy_label, 1, 'glvq')
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot(fig)
    # # st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

    # print('classification accuracy:', glvq.score(toy_data, toy_label))

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("# Uploaded Data")

        filtered_column_list = st.sidebar.multiselect("Select columns", options=list(df.columns.tolist()), default=list(df.columns.tolist()))

        st.dataframe(df)
    else:
        st.write("# EPL Data")

        column_explanation = {'Column Name':['POS', 'M', 'W', 'D', 'L', 'G', 'GA', 'PTS', 'xG',
                                            'NPxG', 'xGA', 'NPxGA', 'NPxGD', 'PPDA', 'OPPDA', 'DC', 'ODC',
                                            'xPTS', 'G-xG', 'GA-xGA', 'PTS-xPTS', 'G/xG', 'GA/xGA', 'PTS/xPTS',
                                            'Shots pg', 'Possession', 'Pass', 'AerialsWon', 'Tackles pg',
                                            'Interceptions pg', 'Fouls pg', 'Offsides pg', 'Shots OT pg',
                                            'Dribbles pg', 'Offensive Fouled pg', 'Rating', 'NextYear'],
                            'Explanation':["the standing of championship in the year", "the number of matches of the team", "the number of matches won",
                                            "the number of matches draw", "the number of matches lost", "the number of goals scored by team",
                                            'the number of goals scored against team', 'the number of points won by team', 'the expected goals of team',
                                            'the number of non–penalties expected goals of team', 'the number of expected goals against team', 'the number of non–penalties expected goals against team',
                                            'difference between for and against non–penalties goals for the team', 'passes allowed per defensive action in the opposition half', 'opponent passes allowed per defensive action in the opposition half',
                                            'passes completed within an estimated 20 yards of goal (crosses excluded)', 'opponent passes completed within an estimated 20 yards of goal (crosses excluded)', 'expected points of team',
                                            'difference between goals and expected goals for the team', 'difference between goals against and expected goals against the team', 'difference between team points and expected points',
                                            'ratio between goals and expected goals', 'ratio between goals against and expected goals against', 'ratio between points and expected points',
                                            'shots per game', 'overall percentage of team ball possession', 'overall percentage of team completed passes',
                                            'aerials won by team per game', 'tackles by team per game', 'interceptions by team per game',
                                            'fouls of team per game', 'offsides of team per game', 'shots on target by team per game',
                                            'dribbles by team per game', 'fouled of team per game', 'rating in whoscored.com',
                                            'the team is going to have a better or worse season than the previous year in terms of points won']}

        if st.checkbox("Show Column Explanation"):
            col_labels = go.Figure(data=[go.Table(columnwidth = [80,400], header=dict(values=['Column Name', 'Explanation']),
                    cells=dict(values=[column_explanation['Column Name'], column_explanation['Explanation']]))
                        ])
            # col_labels.update_layout(margin=dict(margin_autoexpand=5))
            col_labels.update_layout(margin=dict(b=2, l=2, r=2, t=2))
            st.write(col_labels)

        filtered = st.sidebar.multiselect("Select columns", options=list(df.columns.tolist()[2:-1]), default=list(df.columns.tolist())[2:-1])
        filtered_column_list = ["Team", "Year"] + filtered + ["NextYear"]
        if st.checkbox("Show Data", value = True):
            st.write("Data Dimension: " + str(df.shape[0]) + " rows and " + str(df.shape[1]) + " columns")
            st.dataframe(df[filtered_column_list])

        normalize = st.checkbox('normalizing data', value = True)

    # st.header("Correlation Chart")
    # fig, ax = plt.subplots()
    # # sns.heatmap(df[filtered].corr(), ax=ax)
    # plt.matshow(df.corr())
    # # plt.show()
    # st.write(fig)

    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="datafile.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df[filtered_column_list]), unsafe_allow_html=True)

    # fig_lvq = plot2d(glvq_lvq, X, Y, 1, 'glvq')
    # st.pyplot(fig_lvq)
    # st.markdown("""
    #             <style>
    #             .big-font {
    #                 font-size:300px !important;
    #             }
    #             </style>
    #             """, unsafe_allow_html=True)
    # st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

    with st.expander("Random Forest", expanded = True):
        # Random Forest
        st.write("# Random Forest")

        # filtered = st.multiselect("Filter columns", options=list(df.columns.drop(["Team", "Year", "NextYear"])), default=list(df.columns.drop(["Team", "Year", "NextYear"])))
        # filtered = st.multiselect("Select columns", options=list(df.columns.tolist()[2:-1]), default=list(df.columns.tolist())[2:-1])

        # col1, col2, col3, col4, col5, col6 = st.columns((1,1,1,1,1,1))
        # rf_n_estimators = col1.number_input("n_estimators", min_value = 1, value = 50)
        # rf_criterion = col2.selectbox("criterion", options = ["gini", "entropy"])
        # rf_max_depth = col3.number_input("max_depth", min_value = 2)
        # rf_min_samples_split = col4.number_input("min_samples_split", min_value = 2)
        # rf_min_samples_leaf = col5.number_input("min_samples_leaf", min_value = 1)
        # rf_max_features = col6.selectbox("max_features", options = ["auto", "sqrt", "log2"], index = 0)
        col1, col2, col3 = st.columns((1,1,1))
        rf_n_estimators = col1.number_input("n_estimators", min_value = 1, value = 50)
        rf_criterion = col2.selectbox("criterion", options = ["gini", "entropy"])
        rf_max_depth = col3.number_input("max_depth", min_value = 2)
        col1, col2, col3 = st.columns((1,1,1))
        rf_min_samples_split = col1.number_input("min_samples_split", min_value = 2)
        rf_min_samples_leaf = col2.number_input("min_samples_leaf", min_value = 1)
        rf_max_features = col3.selectbox("max_features", options = ["auto", "sqrt", "log2"], index = 0)

        # filtered_column_list = ["Team", "Year"] + filtered + ["NextYear"]
        df_random_forest = df[filtered_column_list].copy()

        # normalization
        scaler = StandardScaler()
        
        X = df_random_forest.loc[df["Year"] != 2014,filtered].to_numpy()
        if normalize:
            X = scaler.fit_transform(df_random_forest.loc[df["Year"] != 2014,filtered].to_numpy())

        Y = df_random_forest.loc[df_random_forest["Year"] != 2014,:].iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        classifier = RandomForestClassifier(n_estimators = rf_n_estimators, criterion = rf_criterion, max_depth = rf_max_depth, 
                                            min_samples_split = rf_min_samples_split, min_samples_leaf = rf_min_samples_leaf, max_features = rf_max_features,
                                            random_state=np.random.seed(0))
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        st.write("# Classification Report")
        st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

        acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
        st.write("Mean is", acc.mean())

        importances = pd.Series(classifier.feature_importances_, index=df.loc[df["Year"] != 2014,filtered].columns)
        fig, ax = plt.subplots()
        chart = importances.plot.bar(ax=ax)
        st.bar_chart(importances.sort_values(ascending = False))

    with st.expander("SVM"):
        st.write("# C-Support Vector Classification")

        # col1, col2, col3, col4, col5 = st.columns((1,1,1,1,1))
        # svc_C = col1.number_input("C", min_value = 0, value = 1)
        # svc_kernel = col2.selectbox("kernel", options = ["linear", "poly", "rbf", "sigmoid", "precomputed"], index = 2)
        # svc_degree = 3
        # if svc_kernel == "poly":
        #     svc_degree = col3.number_input("degree", value = 3)
        # svc_gamma = col4.selectbox("gamma", options = ["scale", "auto"], index = 0)
        # svc_coef0 = col5.number_input("coef0", value = 0)
        col1, col2, col3 = st.columns((1,1,1))
        svc_C = col1.number_input("C", min_value = 0, value = 1)
        svc_kernel = col2.selectbox("kernel", options = ["linear", "poly", "rbf", "sigmoid", "precomputed"], index = 2)
        svc_degree = 3
        if svc_kernel == "poly":
            svc_degree = col3.number_input("degree", value = 3)
        col1, col2, col3 = st.columns((1,1,1))
        svc_gamma = col1.selectbox("gamma", options = ["scale", "auto"], index = 0)
        svc_coef0 = col2.number_input("coef0", value = 0)

        df_svc = df[filtered_column_list].copy()

        # normalization
        scaler = StandardScaler()

        X = scaler.fit_transform(df_svc.loc[df_svc["Year"] != 2014,filtered].to_numpy())
        if normalize:
            X = scaler.fit_transform(df_svc.loc[df_svc["Year"] != 2014,filtered].to_numpy())

        Y = df_svc.loc[df_svc["Year"] != 2014,:].iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)

        svc = SVC(C = svc_C, kernel = svc_kernel, degree = svc_degree, gamma = svc_gamma, coef0 = svc_coef0)
        svc.fit(X_train, y_train)
        predictions = svc.predict(X_test)

        st.write("# Classification Report")
        st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

        acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
        st.write("Mean is", acc.mean())

    with st.expander("k-nearest neighbors"):
        st.write("# k-nearest neighbors")

        # col1, col2, col3, col4, col5, col6 = st.columns((1,1,1,1,1,1))
        # knn_n_neighbors = col1.number_input("n_neighbors", min_value = 1, value = 2)
        # knn_weights = col2.selectbox("weights", options = ["uniform", "distance"], index = 0)
        # knn_algorithm = col3.selectbox("algorithm", options = ["auto", "ball_tree", "kd_tree", "brute"], index = 0)
        # knn_leaf_size = col4.number_input("leaf_size", min_value = 1, value = 10)
        # knn_p = col5.number_input("p", min_value = 1, value = 2)
        # knn_metric = col6.selectbox("metric", options = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"], index = 3)
        col1, col2, col3 = st.columns((1,1,1))
        knn_n_neighbors = col1.number_input("n_neighbors", min_value = 1, value = 2)
        knn_weights = col2.selectbox("weights", options = ["uniform", "distance"], index = 0)
        knn_algorithm = col3.selectbox("algorithm", options = ["auto", "ball_tree", "kd_tree", "brute"], index = 0)
        col1, col2, col3 = st.columns((1,1,1))
        knn_leaf_size = col1.number_input("leaf_size", min_value = 1, value = 10)
        knn_p = col2.number_input("p", min_value = 1, value = 2)
        knn_metric = col3.selectbox("metric", options = ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobis"], index = 3)

        df_knn = df[filtered_column_list].copy()

        # normalization
        scaler = StandardScaler()
        
        X = df_knn.loc[df["Year"] != 2014,filtered].to_numpy()
        if normalize:
            X = scaler.fit_transform(df_knn.loc[df["Year"] != 2014,filtered].to_numpy())

        Y = df_knn.loc[df_knn["Year"] != 2014,:].iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        knn = KNeighborsClassifier(n_neighbors = int(knn_n_neighbors), weights = knn_weights, algorithm = knn_algorithm, leaf_size = knn_leaf_size, p = knn_p, metric = knn_metric)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

        st.write("# Classification Report")
        st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

        acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
        st.write("Mean is", acc.mean())

    with st.expander("Gaussian Naive Bayes"):
        st.write("# Gaussian Naive Bayes")

        col1, col2, col3 = st.columns((1,1,1))
        nb_var_smoothing = col1.number_input("var_smoothing", value = 0.00001, step = 0.000001)

        df_nb = df[filtered_column_list].copy()

        # normalization
        scaler = StandardScaler()
        
        X = df_nb.loc[df["Year"] != 2014,filtered].to_numpy()
        if normalize:
            X = scaler.fit_transform(df_nb.loc[df["Year"] != 2014,filtered].to_numpy())

        Y = df_nb.loc[df_nb["Year"] != 2014,:].iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        nb = GaussianNB(var_smoothing = nb_var_smoothing)
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_test)

        st.write("# Classification Report")
        st.write(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())

        acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
        st.write("Mean is", acc.mean())

    with st.expander("Learning Vector Quantization (LVQ2.1)"):
        st.write("# Learning Vector Quantization (LVQ2.1)")

        col1, col2, col3 = st.columns((1,1,1))
        lvq21_prototypes_per_class = col1.number_input("prototypes_per_class", min_value = 1, value = 1)
        lvq21_max_iter = col2.number_input("max_iter", min_value = 100, value = 2500)
        
        st.write("# Classification Report")
        df_lvq = df[filtered_column_list].copy()
        X = df_lvq.loc[df["Year"] != 2014,filtered].to_numpy()
        Y = df_lvq.loc[df_lvq["Year"] != 2014,:].iloc[:, -1].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        glvq_lvq = GlvqModel(prototypes_per_class=lvq21_prototypes_per_class, max_iter=lvq21_max_iter)
        glvq_lvq.fit(X_train, y_train)
        pred = glvq_lvq.predict(X_test)
        print(pred)
        print(y_test)
        st.write(pd.DataFrame(classification_report(y_test, pred, output_dict=True)).transpose())

        acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
        st.write("Mean is", acc.mean())