#Importing Packages
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             r2_score)

import imageio as iio
 
# read in images
MLBImage = iio.imread("mlbLogos.jpg")

#Importing and Fixing all Data
FinalBattingData = pd.read_pickle("FinalBattingData.pkl")
FinalBattingData2022 = pd.read_pickle("FinalBattingData2022.pkl")
FinalPitchingData = pd.read_pickle("FinalPitchingData.pkl")
FinalPitchingData2022 = pd.read_pickle("FinalPitchingData2022.pkl")
FinalCombinedData = pd.read_csv("FINALCOMBINEDDATA.csv")
FinalCombinedData2022 = pd.read_csv("FINALCOMBINEDDATA2022.csv")
FinalCombinedData.fillna(value = 0, inplace = True)
FinalCombinedData2022.fillna(value = 0, inplace = True)
GroupedFinalCombinedDataSize = FinalCombinedData.groupby(by = ['team_ID_y', 'year_ID', 'WinPercent']).size()
GroupedFinalCombinedDataMean = FinalCombinedData.groupby(by = ['team_ID_y', 'year_ID', 'WinPercent']).mean()
GroupedFinalCombinedDataMean = GroupedFinalCombinedDataMean[GroupedFinalCombinedDataSize > 10]
GroupedFinalCombinedDataSize2022 = FinalCombinedData2022.groupby(by = ['team_ID_y', 'year_ID', 'WinPercent']).size()
GroupedFinalCombinedDataMean2022 = FinalCombinedData2022.groupby(by = ['team_ID_y', 'year_ID', 'WinPercent']).mean()
GroupedFinalCombinedDataMean2022 = GroupedFinalCombinedDataMean2022[GroupedFinalCombinedDataSize2022 > 10]
GroupedFinalCombinedDataMean.reset_index(inplace = True)
GroupedFinalCombinedDataMean2022.reset_index(inplace = True)


#BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL BATTING MODEL
XBatting = FinalBattingData[['PA', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'SH', 'SF', 'GDP',
                     'SB', 'CS', 'BA', 'OPS']]

yBatting = FinalBattingData['WAR']

X_train, X_test, y_train, y_test = train_test_split(XBatting, yBatting, test_size=0.2, random_state=101)

lmBatting = LinearRegression()

lmBatting.fit(X_train, y_train)

predictionsBatting = lmBatting.predict(X_test)

XBatting2022 = FinalBattingData2022[['PA', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'SH', 'SF', 'GDP',
                     'SB', 'CS', 'BA', 'OPS']]

yBatting2022 = FinalBattingData2022['WAR']

predictionsBatting2022 = lmBatting.predict(XBatting2022)

FinalBattingData2022['predicted_WAR'] = predictionsBatting2022

BattingPredictionsDF = pd.DataFrame(data = FinalBattingData2022[['Name_x','WAR', 'predicted_WAR']])


#PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL PITCHING MODEL
Xpitching = FinalPitchingData[['W', 'L', 'SV', 'IP', 'H', 'ER', 'BB', 'SO',
       'HR', 'HBP', 'ERA', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS',
       'PO', 'BF', 'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip',
       'SO9', 'SO/W']]

ypitching = FinalPitchingData['WAR']

X_train, X_test, y_train, y_test = train_test_split(Xpitching, ypitching, test_size=0.2, random_state=101)

lmPitching = LinearRegression()

lmPitching.fit(X_train, y_train)

predictionsPitching = lmPitching.predict(X_test)

XPitching2022 = FinalPitchingData2022[['W', 'L', 'SV', 'IP', 'H', 'ER', 'BB', 'SO',
       'HR', 'HBP', 'ERA', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS',
       'PO', 'BF', 'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip',
       'SO9', 'SO/W']]


yPitching2022 = FinalPitchingData2022['WAR']

predictionsPitching2022 = lmPitching.predict(XPitching2022)

FinalPitchingData2022['predicted_WAR'] = predictionsPitching2022

PitchingPredictionsDF = pd.DataFrame(data = FinalPitchingData2022[['Name_x','WAR', 'predicted_WAR']])





#TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL TEAM MODEL 
Xteam = GroupedFinalCombinedDataMean[['WAR', 'GS', 'W',
       'L', 'SV', 'IP', 'H', 'ER', 'BB', 'SO', 'HR', 'HBP', 'ERA', 'AB', '2B',
       '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS', 'PO', 'BF', 'Str', 'StL', 'StS',
       'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W', 'PA', 'R', 'H.1',
       '2B.1', '3B.1', 'HR.1', 'RBI', 'BB.1', 'IBB.1', 'SO.1', 'HBP.1', 'SH',
       'SF.1', 'GDP.1', 'SB.1', 'CS.1', 'BA', 'OPS']]

yteam = GroupedFinalCombinedDataMean['WinPercent']

X_train, X_test, y_train, y_test = train_test_split(Xteam, yteam, test_size=0.2, random_state=101)

lmTeam = LinearRegression()

lmTeam.fit(X_train, y_train)

predictionsTeam = lmTeam.predict(X_test)

Xteam2022 = GroupedFinalCombinedDataMean2022[['WAR', 'GS', 'W',
       'L', 'SV', 'IP', 'H', 'ER', 'BB', 'SO', 'HR', 'HBP', 'ERA', 'AB', '2B',
       '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS', 'PO', 'BF', 'Str', 'StL', 'StS',
       'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W', 'PA', 'R', 'H.1',
       '2B.1', '3B.1', 'HR.1', 'RBI', 'BB.1', 'IBB.1', 'SO.1', 'HBP.1', 'SH',
       'SF.1', 'GDP.1', 'SB.1', 'CS.1', 'BA', 'OPS']]

yteam2022 = GroupedFinalCombinedDataMean2022['WinPercent']

predictionsTeam2022 = lmTeam.predict(Xteam2022)

GroupedFinalCombinedDataMean2022['predicted_WinPercent'] = predictionsTeam2022

TeamPredictions2022 = pd.DataFrame(data = GroupedFinalCombinedDataMean2022[['team_ID_y','WinPercent', 'predicted_WinPercent']])

# Coefficients

Teamcoeff_df = pd.DataFrame(data = lmTeam.coef_, index = Xteam.columns, columns = ['Coefficient'])


Battingcoeff_df = pd.DataFrame(data = lmBatting.coef_, index = XBatting.columns, columns = ['Coefficient'])


Pitchingcoeff_df = pd.DataFrame(data = lmPitching.coef_, index = Xpitching.columns, columns = ['Coefficient'])

BattingStats2022 = FinalBattingData2022[['Name_x', 'Age', 'PA', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB',
       'SO', 'HBP', 'SH', 'SF', 'GDP', 'SB', 'CS', 'BA', 'OPS',
       'year_ID','team_ID_y']]

PitchingStats2022 = FinalPitchingData2022[['Name_x', 'Age', 'GS', 'W', 'L', 'SV', 'IP', 'H', 'ER', 'BB', 'SO',
       'HR', 'HBP', 'ERA', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS',
       'PO', 'BF', 'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip',
       'SO9', 'SO/W','year_ID', 'team_ID_y']]




# Create a page dropdown 
page = st.sidebar.selectbox("Choose your page", ["Home", "Data Coefficients", "Batting Stats", "Pitching Stats", "BattingWAR Prediction", "PitchingWAR Prediction", "Team WAR Prediction"]) 
if page == "Home":
    st.write("# Welcome to my application!")
    st.markdown("""This application utilizes Major League Baseball Data ranging from 2010-2022 in prediction modeling! **Select a page from the sidebar** to learn more about the modeling and its predictions!
    """)
    st.image(MLBImage)
    st.write("To see even more about the project, view it on my [Github](https://github.com/gavinpastreich)!")

elif page == "BattingWAR Prediction":
    st.write("# Batting WAR Predictions")
    text_search = st.text_input("Search by Batter", value="")
    input = BattingPredictionsDF["Name_x"].str.contains(text_search)
    df_search = BattingPredictionsDF[input]
    df_search4 = BattingStats2022[input]

    if text_search:
        st.write(df_search)
        st.write(df_search4)





elif page == "PitchingWAR Prediction":
    st.write("# Pitching WAR Predictions")
    text_search2 = st.text_input("Search by Pitcher", value="")
    input2 = PitchingPredictionsDF["Name_x"].str.contains(text_search2)
    df_search2 = PitchingPredictionsDF[input2]
    df_search3 = PitchingStats2022[input2]

    if text_search2:
        st.write(df_search2)
        st.write(df_search3)

elif page == "Team WAR Prediction":
    st.write("# Predicted Team Win Percents")
    TeamPredictions2022

elif page == "Data Coefficients":
    st.write("# Coefficients of the models")
    col1, col2, col3 = st.columns(3)
    with col1:
      st.header("Batting Coefficients")
      Battingcoeff_df

    with col2:
      st.header("Pitching Coefficients")
      Pitchingcoeff_df

    with col3:
      st.header("Team Coefficients")
      Teamcoeff_df

elif page == "Batting Stats":
    st.write("# 2022 Batting Stats")
    BattingStats2022

elif page == "Pitching Stats":
    st.write("# 2022 Pitching Stats")
    PitchingStats2022
