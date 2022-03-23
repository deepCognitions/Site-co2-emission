from argparse import FileType
from heapq import heappop
from shutil import move
import streamlit as st
import shap
import pickle
import pandas as pd
import numpy as np
import base64
import joblib
import math
import streamlit.components.v1 as components
from predict import get_prediction
from PIL import Image
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

img_bg = Image.open('Img/seui_bg.jpg')
img_icons = Image.open('Img/seui_icons.png')
img_01 = Image.open('Img/seui_img01.jpg')
img_02 = Image.open('Img/seui_img02.jpg')



with open('Model/catb__final.joblib', 'rb') as f:
    catb = joblib.load(f)
with open('Notebook/facility.pkl', 'rb') as t:
    options_ftype = list(joblib.load(t))
#shap.initjs()      

catbe = shap.TreeExplainer(catb)
      

def explain_model_prediction(data,catbe):
        # Calculate Shap values
        shap_values = catbe.shap_values(data)
        p = shap.force_plot(catbe.expected_value, shap_values, data)
        return p, shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



st.set_page_config(page_title="Deep =>  Site's CO2 Emission evaluation",
                   page_icon="🏙️", layout="wide" )
                   
  

Features = ['energy_star_rating', 'cooling_degree_days', 'heating_degree_days', 'precipitation_inches',
       'snowfall_inches', 'snowdepth_inches','days_below_30f',
       'days_below_20f', 'days_below_10f', 'days_above_80f',
       'days_with_fog', 'facility_type', 'summer', 'rainy']


def main():
    


    
    st.markdown(
        """
        <style>
        .container {
            display: flex;
            background-color: #DCF6F5;
          
        }
        
        .logo-text {
            font-weight:600 !important;
            font-size:45px !important;
            color: #1D7AA7 !important;
            padding-top: 175px !important;
            padding-left: 75px !important;
            
        }
        .logo-img {
            float:right;
            width:50%;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('Img/seui_icons.png', "rb").read()).decode()}">
            <p class="logo-text">Site's CO2 Emission Evaluation</p>
        </div>
        """,
        unsafe_allow_html=True
    )
        
    with st.form('prediction_form'):
       

        st.header("Enter the input for following info:")

        col1, col2, = st.columns(2)
        
        with col1:
      
            inp = {}
            facility = st.selectbox("Facility type ", options=options_ftype)
            st.subheader("Temperature (in Farenheit)",)
            rainy = st.number_input("Rainy(May-Sep) :", value = 0.0, format="%f")
            summer = st.number_input("Summer(Jan-April) :", value = 0.0, format="%f")
            avg = st.number_input("Aveage(yearly) :",value = 0.0, format="%f")

            st.subheader("Snow/Rain (in inches)")
            snowfall = st.number_input("Snowfall :", value = 0.0, format="%f")
            snowdepth = st.number_input("Snow-depth :", value = 0.0, format="%f")
            precip = st.number_input("Precipitation :",value = 0.0, format="%f")
            
        with col2:
            energy = st.number_input("Energy Rating :", max_value = 100, step=1, value = 0, format="%d")
            st.subheader("Total Days")
            below30 = st.number_input("Days below 30F :", value = 0, format="%d")
            below20 = st.number_input("Days below 20F :", value = 0, format="%d")
            below10 = st.number_input("Days below 10F :", value = 0, format="%d")
            above80 = st.number_input("Days above 80F :",value = 0, format="%d")
            foggy = st.number_input("Foggy days :",value = 0, format="%d")

            st.subheader("Degree days")
            cooling = st.number_input("Cooling Deg days :", value = 0, format="%d")
            heating = st.number_input("Heating Deg days :", value = 0, format="%d")
        submit = st.form_submit_button("Estimate the CO2 emission !")


    if submit:
        inp["energy_star_rating"] = energy
        inp["cooling_degree_days"] = cooling
        inp["heating_degree_days"] = heating
        inp["precipitation_inches"] = precip
        inp["snowfall_inches"] = snowfall
        inp["snowdepth_inches"] = snowdepth
        inp["days_below_30f"] = below30
        inp["days_below_20f"] = below20
        inp["days_below_10f"] = below10
        inp["days_above_80f"] = above80
        inp["days_with_fog"] = foggy
        inp["facility_type"] = options_ftype.index(facility)
        inp["summer"] = summer
        inp["rainy"] = rainy

        df = pd.DataFrame.from_dict([inp])
        pred = get_prediction(data=df, model=catb)

        st.markdown("""<style> .big-font { font-family:sans-serif; color: #1D7AA7 ; font-size: 50px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">{round(pred, 2)} of CO2 is estimated.</p>', unsafe_allow_html=True)
        #st.write(f" => {pred} is predicted. <=")

        p, shap_values = explain_model_prediction(df,catbe)
        st.subheader('CO2 Interpretation')
        st_shap(p)
    
    

if __name__ == '__main__':
    main()