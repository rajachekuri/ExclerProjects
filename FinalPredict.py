# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:33:13 2023

@author: R5-PC
"""

import pickle
import streamlit as st
import numpy as np

# Load the Random Forest model using pickle
with open('C:/Users/Opstech Demo/Desktop/DataScience/ExcelrProjects/clusteringP288/trained_model.sav', 'rb') as file:
    model = pickle.load(file)

# Create a feature vector from the user inputs
def create_feature_vector():
    return np.array([
        birth_rate, co2_emissions, energy_usage, gdp, health_exp_gdp, infant_mortality,
        internet_usage, life_expectancy_female, life_expectancy_male, mobile_phone_usage,
        population_0_14, population_15_64, population_65_plus, population_total,
        population_urban, tourism_inbound, tourism_outbound
    ]).reshape(1, -1)

# Create a function to predict the cluster
def predict_cluster(user_input):
    cluster = model.predict(user_input)[0]
    return cluster

st.title('Cluster Prediction App')

# Create input fields for user input
birth_rate = st.number_input('Birth Rate', min_value=0.0, max_value=100.0)
co2_emissions = st.number_input('CO2 Emissions', min_value=0.0)
energy_usage = st.number_input('Energy Usage', min_value=0.0)
gdp = st.number_input('GDP', min_value=0.0)
health_exp_gdp = st.number_input('Health Exp % GDP', min_value=0.0, max_value=1.0)
infant_mortality = st.number_input('Infant Mortality Rate', min_value=0.0)
internet_usage = st.number_input('Internet Usage', min_value=0.0)
life_expectancy_female = st.number_input('Life Expectancy Female', min_value=0.0)
life_expectancy_male = st.number_input('Life Expectancy Male', min_value=0.0)
mobile_phone_usage = st.number_input('Mobile Phone Usage', min_value=0.0)
population_0_14 = st.number_input('Population 0-14', min_value=0.0)
population_15_64 = st.number_input('Population 15-64', min_value=0.0)
population_65_plus = st.number_input('Population 65+', min_value=0.0)
population_total = st.number_input('Population Total', min_value=0.0)
population_urban = st.number_input('Population Urban', min_value=0.0)
tourism_inbound = st.number_input('Tourism Inbound', min_value=0.0)
tourism_outbound = st.number_input('Tourism Outbound', min_value=0.0)

# Add a "Predict" button
if st.button("Predict"):
    user_input = create_feature_vector()
    cluster = predict_cluster(user_input)
    st.write(f'Predicted Cluster: {cluster}')
