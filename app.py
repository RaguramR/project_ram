import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
d_set = pd.read_csv('breastDataset.csv')

d_set.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

d_set['diagnosis'] = LabelEncoder().fit_transform(d_set['diagnosis'])

feature = d_set.corr()['diagnosis'].abs().sort_values()[23:]

X_train, X_test, Y_train, Y_test = train_test_split(
    d_set[feature.index].iloc[:, :-1], d_set['diagnosis'], test_size=0.3, random_state=42)

mms.fit_transform(X_train)
mms.fit_transform(X_test)


svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)

ch = []

st.markdown("<h1 style='text-align: center;'>Breast Cancer Detection</h1>",
            unsafe_allow_html=True)
nav = st.sidebar.radio("Navigation", ['Input Section'])

if nav == 'Input Section':

    st.subheader("Please Enter the given values")
    col1, col2 = st.columns(2)
    radius_mean = col1.number_input(
        "Radius Mean", min_value=-1000.0000, format="%.4f")

    area_worst = col2.number_input(
        "Area worst", min_value=-1000.0000, format="%.4f")

    perimeter_mean = col1.number_input(
        "Parameter Mean", min_value=-1000.0000, format="%.4f")

    radius_worst = col2.number_input(
        "Radius worst", min_value=-1000.0000, format="%.4f")

    concave_points_mean = col1.number_input(
        "Concave Points Mean", min_value=-1000.0000, format="%.4f")

    perimeter_worst = col1.number_input(
        "Perimeter Worst", min_value=-1000.0000, format="%.4f")

    concave_points_worst = col2.number_input(
        "Concave Points Worst", min_value=-1000.0000, format="%.4f")

    if st.button('submit'):
        if -1000.0000 not in ch:

            ch.append(radius_mean)
            ch.append(area_worst)
            ch.append(perimeter_mean)
            ch.append(radius_worst)
            ch.append(concave_points_mean)
            ch.append(perimeter_worst)
            ch.append(concave_points_worst)
            st.info(f"{ch}")

            st.info('Submitted')

            val = np.array(ch).reshape(1, -1)
            mms.fit_transform(val)
            val = svm.predict(val)
            if val == 0:
                st.success('Not Cancer')
            elif val == 1:
                st.error('Cancer')
        else:
            st.warning('Input are Empty')
