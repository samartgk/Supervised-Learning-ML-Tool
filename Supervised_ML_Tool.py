########################################################################################################################
#####                                              By : ALPHA                                                      #####
########################################################################################################################
import streamlit as st
import random
import warnings
import plotly.graph_objects as go
import gc
import math
import pandas as pd
import numpy as np
import base64
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from timeit import default_timer as timer
from PIL import Image
gc.collect()
start = timer()


##################################################### FUNCTIONS ########################################################


def Visualisations(graph_type, x_data, y_data):
    if graph_type == 'Line Chart':
        st.markdown("<h1 style='text-align: center; color: white; font-size: 2em;'>Train-Test Split</h1>",
                    unsafe_allow_html=True)
        labels = ['X', 'Y']
        values = [x_data, y_data]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textfont=dict(size=16)
        )])

        fig.update_layout(
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            legend=dict(
                font=dict(
                    size=25
                ),
                x=0.75,
                y=0.5,
                xanchor='right',
                yanchor='middle'
            )
        )

        st.plotly_chart(fig, use_container_width=True)


def Perform_Std_Scalar(db, columns_to_scale):
    scaler = StandardScaler()
    for col in columns_to_scale:
        db[col] = scaler.fit_transform(db[[col]])
    return db


def page_1():
    st.markdown("<h1 style='text-align: center; color: red; font-size: 3em;'>Machine Learning</h1>",
                unsafe_allow_html=True)
    try:
        Input_File = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'xls'])
        file_extension = Input_File.name.split(".")[-1].lower()
    except:
        st.stop()

    if file_extension == "csv":
        input_df = pd.read_csv(Input_File)
    elif file_extension in ["xls", "xlsx"]:
        input_df = pd.read_excel(Input_File)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")

    input_df = Raw_Data_Cleanup(input_df)
    input_df = Identification(input_df)
    final_train, final_test = Model_Process(input_df)
    calculated_variable = 'Calculated_' + str(target_column)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Expand to visualize Train Dataset"):
            st.markdown("<h1 style='text-align: center; color: white; font-size: 2em;'>TRAIN</h1>",
                        unsafe_allow_html=True)
            with st.container():
                st.markdown("<h1 style='text-align: center; color: white; font-size: 1em;'>Graph</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    .bokeh { width: 100% !important; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                data = pd.DataFrame({
                    'True Values': final_train[target_column],
                    'Calculated Values': final_train[calculated_variable]
                })

                true_values = list(final_test[target_column])
                predicted_values = list(final_test[calculated_variable])

                df = pd.DataFrame({
                    "x": list(range(len(true_values))),
                    "true": true_values,
                    "predicted": predicted_values
                })

                fig_train = go.Figure()

                fig_train.add_trace(
                    go.Scatter(
                        x=df["x"],
                        y=df["true"],
                        mode="markers",
                        name="True Values",
                        marker=dict(size=8, color="red", opacity=0.7),
                        customdata=df[["true", "predicted"]],  # Include both true and predicted for hover
                        hovertemplate="True: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
                    )
                )

                fig_train.add_trace(
                    go.Scatter(
                        x=df["x"],
                        y=df["predicted"],
                        mode="markers",
                        name="Predicted Values",
                        marker=dict(size=8, color="green", opacity=0.7),
                        customdata=df[["true", "predicted"]],  # Include both true and predicted for hover
                        hovertemplate="True: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
                    )
                )

                fig_train.update_layout(
                    title="True vs. Predicted Values",
                    xaxis_title="Data Point Index",
                    yaxis_title="Values",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    showlegend=True,
                    hovermode="x"
                )


                st.plotly_chart(fig_train, use_container_width=True, key="true_vs_predicted_plot_for_Train")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: white; font-size: 1em;'>Dataset</h1>",
                            unsafe_allow_html=True)
                st.dataframe(final_train)

    with col2:
        with st.expander("Expand to visualize Test Dataset"):
            st.markdown("<h1 style='text-align: center; color: white; font-size: 2em;'>TEST</h1>",
                        unsafe_allow_html=True)
            with st.container():
                st.markdown("<h1 style='text-align: center; color: white; font-size: 1em;'>Graph</h1>",
                            unsafe_allow_html=True)
                st.markdown(
                    """
                    <style>
                    .bokeh { width: 100% !important; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                data = pd.DataFrame({
                    'True Values': final_test[target_column],
                    'Calculated Values': final_test[calculated_variable]
                })

                true_values = list(final_test[target_column])
                predicted_values = list(final_test[calculated_variable])

                df = pd.DataFrame({
                    "x": list(range(len(true_values))),
                    "true": true_values,
                    "predicted": predicted_values
                })

                fig_test = go.Figure()

                fig_test.add_trace(
                    go.Scatter(
                        x=df["x"],
                        y=df["true"],
                        mode="markers",
                        name="True Values",
                        marker=dict(size=8, color="red", opacity=0.7),
                        customdata=df[["true", "predicted"]],  # Include both true and predicted for hover
                        hovertemplate="True: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
                    )
                )

                fig_test.add_trace(
                    go.Scatter(
                        x=df["x"],
                        y=df["predicted"],
                        mode="markers",
                        name="Predicted Values",
                        marker=dict(size=8, color="green", opacity=0.7),
                        customdata=df[["true", "predicted"]],  # Include both true and predicted for hover
                        hovertemplate="True: %{customdata[0]}<br>Predicted: %{customdata[1]}<extra></extra>"
                    )
                )

                fig_test.update_layout(
                    title="True vs. Predicted Values",
                    xaxis_title="Data Point Index",
                    yaxis_title="Values",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    showlegend=True,
                    hovermode="x"
                )

                st.plotly_chart(fig_test, use_container_width=True, key="true_vs_predicted_plot_for_Test")
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("<h1 style='text-align: center; color: white; font-size: 1em;'>Dataset</h1>",
                            unsafe_allow_html=True)
                st.dataframe(final_train)

    st.markdown("<h1 style='text-align: center; color: red; font-size: 3em;'>THANK YOU</h1>",
                unsafe_allow_html=True)


def intro_page():
    st.title("Welcome to Centralised Machine Learning Tool!")

    st.write("A simple tool to analyse and get insights into the dataset provided and the various models"
             " that can be implemented and tuned as needed. This also gives the various possible outcomes that can"
             " be done from one dataset.")

    st.header("What this does:")
    st.write(
        """
        * Provides multiple models.
        * Allows you to change parameters.
        * Offers 2 different train and test split methods.
        """
    )

    st.header("How to use it:")
    st.write(
        """
        1.  Navigate to the tool using the sidebar.
        2.  Follow the instructions on the page.
        3.  Enjoy!
        """
    )

    st.header("About the Developer:")
    st.write(
        """
        This app was created by ALPHA. If you have any suggestions or requests please write to samartgk.10@gmail.com
        """
    )
    image = Image.open(r"my_photo.jpg")
    resized_image = image.resize((100, 100))
    st.image(resized_image, caption="Developer", use_container_width=False)


def Set_Background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def Raw_Data_Cleanup(df):
    Unnameds1 = []
    for column in df.columns:
        if -1 != column.find('Unnamed'):
            Unnameds1.append(column)
    df.drop(Unnameds1, axis=1, inplace=True, errors='ignore')
    cols_with_inf = [col for col in df.columns if df[col].isin([float('inf'), -float('inf')]).any()]
    df.drop(cols_with_inf, axis=1, inplace=True)
    # df.dropna(axis=1, how='any', inplace=True)
    return df


def Regression_Results(a, b):
    total = len(a)
    if total == 0:
        st.write("EMPTY DATASET")
        return
    performance = mean_absolute_percentage_error(a, b)
    st.markdown(f"<h1 style='text-align: center; color: yellow; font-size: 2em;'>Model Accuracy = {round(100 - (performance*100),2)}%</h1>",
                unsafe_allow_html=True)
    with st.expander("Expand to view Detailed Results"):
        st.write("Mean absolute Percentage error : " + str(performance))
        error = mean_absolute_error(a, b)
        st.write("Mean absolute error : " + str(error))
        MSE = np.square(np.subtract(a, b)).mean()
        st.write("Mean square error : " + str(MSE))
        RMSE = math.sqrt(MSE)
        st.write("Root Mean Square Error:" + str(RMSE))
        r2 = r2_score(a, b)
        st.write("R2 Score:" + str(r2))
    return


def Classification_Results(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        st.write("EMPTY DATASET")
        return
    try:
        accuracy = accuracy_score(y_true, y_pred)
        st.markdown(
            f"<h1 style='text-align: center; color: yellow; font-size: 2em;'>Model Accuracy = {round(accuracy * 100, 2)}%</h1>",
            unsafe_allow_html=True)
    except:
        st.write("Accuracy cannot be done")

    with st.expander("Expand to view Detailed Results"):
        try:
            accuracy = accuracy_score(y_true, y_pred)
            st.write("Accuracy: " + str(accuracy))
        except:
            st.write("Accuracy cannot be done")

        try:
            precision = precision_score(y_true, y_pred)
            st.write("Precision: " + str(precision))
        except:
            st.write("Precision cannot be done")

        try:
            recall = recall_score(y_true, y_pred)
            st.write("Recall: " + str(recall))
        except:
            st.write("Recall cannot be done")

        try:
            f1 = f1_score(y_true, y_pred)
            st.write("F1 Score: " + str(f1))
        except:
            st.write("f1_score cannot be done")

        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            st.write("ROC AUC: " + str(roc_auc))
        except:
            st.write("roc_auc cannot be done")

    return


def ML_Predictions(db, problem_type, model):
    Y = db[[target_column]]
    X = db[[x for x in db.columns if x not in Y.columns]]

    split_type = st.selectbox("Choose how to split the data for training and testing", ('Built-In Train-Test Split',
                                                                                        'Grouping based on 1 Column'))

    if split_type == 'Built-In Train-Test Split':
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=st.slider('Choose the test size', 0.1, 0.9,
                                                                                key="test_size_slider"),
                                                            random_state=st.slider('Choose the random state for split',
                                                                                   0, 100,
                                                                                   key="random_state_slider"))

        train = pd.concat([x_train, y_train], axis=1)
        test = pd.concat([x_test, y_test], axis=1)

    else:
        random.seed(st.slider('Choose the random state for split', 0, 100, key="random_seed_slider"))
        selected_column = st.selectbox("Choose the column to group by :", db.columns)
        uniquelist = list(db[selected_column].unique())
        numbers_to_select = int(len(uniquelist) * st.slider('Choose the test size', 0.2, 0.9,
                                                            key="test_number_slider"))
        if numbers_to_select < 2:
            numbers_to_select = 2
        test_list = random.sample(uniquelist, numbers_to_select)
        train_list = [x for x in uniquelist if x not in test_list]
        train = db[db[selected_column].isin(train_list)]
        test = db[db[selected_column].isin(test_list)]
        y_test = test[[target_column]]
        y_train = train[[target_column]]
        x_train = train.drop(target_column, axis=1)
        x_test = test.drop(target_column, axis=1)

    st.markdown("<h1 style='text-align: center; color: white; font-size: 2em;'>Train-Test Split</h1>",
                unsafe_allow_html=True)
    labels = ['X_Train', 'X_Test', 'Y_Train', 'Y_Test']
    values = [len(x_train), len(x_test), len(y_train), len(y_test)]
    explode = (0.1, 0.1, 0.1, 0.1)

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        pull=explode,
        textfont=dict(size=16)
    )])

    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend=dict(
            font=dict(
                size=25
            ),
            x=0.75,
            y=0.5,
            xanchor='right',
            yanchor='middle'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    avoid_list = [target_column]
    column_list = [x for x in db.columns if x not in avoid_list]
    feature_list = []
    corr_number = st.slider('Choose the correlation Value for feature selection', 0.0, 1.0, key="correlation_slider")
    for x in column_list:
        if pd.api.types.is_numeric_dtype(train[x]):
            corr1 = train[x].corr(train[target_column])
            if abs(corr1) > corr_number:
                feature_list.append(x)

    if len(feature_list) > 0:
        st.write('The selected features are :', feature_list)
    else:
        st.write('No features selected. Please change correlation Value')
        st.stop()

    x_train = train[feature_list]

    y_train = train[target_column]

    x_test = test[feature_list]

    y_test = test[target_column]

    if st.button("PREDICT", key="centered-button"):
        try:
            model.fit(x_train, y_train)
            y_pred_train = model.predict(x_train)
            y_pred = model.predict(x_test)
        except:
            st.write('Please change above values as current values are incorrect.')
            st.stop()
    else:
        st.stop()

    calculated_variable = 'Calculated_' + str(target_column)
    train[calculated_variable] = y_pred_train
    train['Calculated_Diff'] = abs(train[target_column] - train[calculated_variable])

    test[calculated_variable] = y_pred
    test['Calculated_Diff'] = abs(test[target_column] - test[calculated_variable])

    st.markdown(
        f"<h1 style='text-align: center; color: white; font-size: 2em;'>Performance of Model : </h1>",
        unsafe_allow_html=True)
    if problem_type == 'Classification':
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<h1 style='text-align: center; color: white; font-size: 1em;'>Train Data</h1>",
                unsafe_allow_html=True)
            Classification_Results(y_train, y_pred_train)
        with col2:
            st.markdown(
                f"<h1 style='text-align: center; color: white; font-size: 1em;'>Test Data</h1>",
                unsafe_allow_html=True)
            Classification_Results(y_test, y_pred)

    elif problem_type == 'Regression':
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<h1 style='text-align: center; color: white; font-size: 1em;'>Train Data</h1>",
                unsafe_allow_html=True)
            Regression_Results(y_train, y_pred_train)
        with col2:
            st.markdown(
                f"<h1 style='text-align: center; color: white; font-size: 1em;'>Test Data</h1>",
                unsafe_allow_html=True)
            Regression_Results(y_test, y_pred)

    else:
        st.error('Try Again')

    return train, test


def Identification(db):
    global target_column

    if 'show_full_dataframe' not in st.session_state:
        st.session_state.show_full_dataframe = False

    if st.button("Display Entire DataSet"):
        st.session_state.show_full_dataframe = True

    if st.session_state.show_full_dataframe:
        st.dataframe(db)
        if st.button("Close DataSet View"):
            st.session_state.show_full_dataframe = False

    string_columns = [col for col in db.columns if db[col].apply(lambda x: isinstance(x, str)).any()]
    numeric_columns = [x for x in db.columns if x not in string_columns]
    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f'Click to view the {len(numeric_columns)} columns with numeric values '):
            st.write(numeric_columns)
            if 'show_num_dataframe' not in st.session_state:
                st.session_state.show_num_dataframe = False
            if st.button("View Numeric columns only "):
                st.session_state.show_num_dataframe = True
            if st.session_state.show_num_dataframe:
                st.dataframe(db[numeric_columns])
                if st.button("Close Numeric-only View"):
                    st.session_state.show_num_dataframe = False
    with col2:
        with st.expander(f'Click to view the {len(string_columns)} columns with String values '):
            st.write(string_columns)
            if 'show_string_dataframe' not in st.session_state:
                st.session_state.show_string_dataframe = False
            if st.button("View String columns only "):
                st.session_state.show_string_dataframe = True
            if st.session_state.show_string_dataframe:
                st.dataframe(db[string_columns])
                if st.button("Close String-only View"):
                    st.session_state.show_string_dataframe = False

    target_column = st.selectbox("Choose the column to predict", db.columns)

    process_string_columns = st.selectbox("Do you want to Select certain columns", ("NO", "YES"))
    if process_string_columns == 'YES':
        column_list = db.columns.tolist()
        selected_items = st.multiselect("Select columns : ", column_list)
        if target_column not in selected_items:
            selected_items.append(target_column)
        db = db[selected_items]
    else:
        process_string_columns = st.selectbox("Do you want to Drop any columns", ("NO", "YES"))
        if process_string_columns == 'YES':
            column_list = db.columns.tolist()
            column_list.remove(target_column)
            selected_items = st.multiselect("Select columns to Drop: ", column_list)
            db.drop(selected_items, axis=1, inplace=True)

    string_columns = [col for col in db.columns if db[col].apply(lambda x: isinstance(x, str)).any()]
    process_string_columns = st.selectbox("Do you want to Encode any columns", ("NO", "YES"))
    if process_string_columns == 'YES':
        selected_items = st.multiselect("Select columns to encode: ", string_columns)
        db = Label_Encoding(db, selected_items)

    process_string_columns = st.selectbox("Do you want to perform Standard Scaler on certain columns", ("NO", "YES"))
    if process_string_columns == 'YES':
        selected_items = st.multiselect("Select columns to Scale: ", db.columns.tolist())
        db = Perform_Std_Scalar(db, selected_items)

    if 'show_new_full_dataframe' not in st.session_state:
        st.session_state.show_new_full_dataframe = False

    if st.button("Display current DataSet after processing"):
        st.session_state.show_new_full_dataframe = True

    if st.session_state.show_new_full_dataframe:
        st.dataframe(db)
        if st.button("Close View"):
            st.session_state.show_new_full_dataframe = False

    return db


def Label_Encoding(db, columns_to_encode):
    label_encoder = preprocessing.LabelEncoder()
    for col in columns_to_encode:
        db[col] = label_encoder.fit_transform(db[col])
    return db


def Model_Process(db):
    problem_type = st.selectbox("Choose the type of problem", ("Classification", "Regression"))
    if problem_type == 'Classification':
        selected_model = st.selectbox("Choose the type of model you want to proceed with",
                                      ("Logistic", "Decision Tree Classifier", 'Random Forest Classifier',
                                       'XGB Classifier',
                                       'LightGBM Classifier', 'SVM Classifier', 'Naive Bayes Classifier', 'KNN Classifier'))
        if selected_model == 'Logistic':
            model = LogisticRegression(random_state=42)

        elif selected_model == 'XGB Classifier':
            model = XGBClassifier()

        elif selected_model == 'Decision Tree Classifier':
            model = DecisionTreeClassifier(max_depth=None, random_state=42)

        elif selected_model == 'Random Forest Classifier':
            model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

        elif selected_model == 'LightGBM Classifier':
            model = LGBMClassifier()

        elif selected_model == 'SVM Classifier':
            model = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)

        elif selected_model == 'Naive Bayes Classifier':
            model = GaussianNB()

        elif selected_model == 'KNN Classifier':
            n_neighbors = st.slider('Choose the number of clusters', 2, 15, key="kn_neighbors")
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        else:
            st.error("Please select a valid Classification model.")
    elif problem_type == 'Regression':
        selected_model = st.selectbox("Choose the type of model you want to proceed with",
                                      ("Linear", "Decision Tree Regressor", 'Random Forest Regressor',
                                       'XGB Regressor', 'LightGBM Regressor', 'SVM Regressor'))

        if selected_model == 'XGB Regressor':
            model = XGBRegressor(objective='reg:squarederror',
                                 enable_categorical='True',
                                 learning_rate=0.1,
                                 max_depth=6,
                                 alpha=10,
                                 n_estimators=150,
                                 num_parallel_tree=7,
                                 random_state=20)

        elif selected_model == 'Linear':
            model = LinearRegression()

        elif selected_model == 'Decision Tree Regressor':
            model = DecisionTreeRegressor(max_depth=None, random_state=42)

        elif selected_model == 'Random Forest Regressor':
            model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)

        elif selected_model == 'LightGBM Regressor':
            model = LGBMRegressor(objective='mape',
                                  num_leaves=400,
                                  learning_rate=0.01,
                                  n_estimators=500,
                                  tree_learner='feature',
                                  random_state=64,
                                  subsample=1,
                                  subsample_freq=1,
                                  reg_alpha=0.1)

        elif selected_model == 'SVM Regressor':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            st.error("Please select a valid Regression model.")
    else:
        st.error("Improper Selection.")

    st.write(model)
    final_train, final_test = ML_Predictions(db, problem_type, model)
    return final_train, final_test


######################################################## MAIN BLOCK ####################################################

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    Set_Background(r"new.png")
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            display: block;
            margin: 0 auto;
            width: 350px;  /* Adjust width as needed */
            height: 50px; /* Adjust height as needed */
            font-size: 20px; /* Adjust font size as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "ML Tool"])

    if page == "Introduction":
        intro_page()
    elif page == "ML Tool":
        page_1()



