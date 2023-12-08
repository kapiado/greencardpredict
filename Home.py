# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:46:34 2023

@author: katri
"""

import streamlit as st
import streamlit.components.v1 as components
# import numpy as np
import pandas as pd
# import plotly.express as px          # Plotly
# import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)


st.write("# Analysis of Factors Affecting U.S. Permanent Residency Using Data and Predictive Analytics")
image = Image.open('green card.jfif')

st.image(image,caption='Figure 1: Green Card Resident Image (Amazon)')




#df = pd.read_csv("PERM_Data_5_22_23.csv")

# Cases received vs. Years
# def cases(dataframe):
#     #uniqueyears = df["CASE_RECEIVED_YEAR"].unique() # Obtain years for x-axis
#     #years = list(uniqueyears) # Put into list
#     numcases= df["CASE_RECEIVED_YEAR"].value_counts().sort_index() # Number of cases per year
#     fig = px.bar(x = numcases.index, y = numcases, title='Cases Received from 2006-2021',height=500,color=numcases)
#     fig.update_xaxes(title='Year',type='category')
#     fig.update_yaxes(title='Total Received',type='linear')
#     st.plotly_chart(fig)


    
# def avgwait():

#     HtmlFile = open("AvgWaitingTime.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     components.html(source_code,height=600)
    
st.header('Background')

c = st.empty()
c.write('The immigration backlog is a result of the accumulation of immigration applications that have not been processed within a reasonable timeframe. \nIt is caused by increased demand, insufficient resources, complex procedures, and policy changes. Backlogs lead to delays in family reunification, economic impact, strain on resources, and uncertainty for individuals. In this dashboard, we aim to achieve transparency for individuals that are in the middle of the process and want to begin the process for residency. We will be  looking at historical data and factors that influence the wait times in this process, and provide an interface where users can predict their personal waiting time based on their demographics.')

#new line and research objectives content
st.write("")
st.header('Research Objectives')
st.write("1. Enhance **transparency**, **reliability**, and **predictability** in the immigration process for EB-2 future applicants.​")
st.write("2. Implement **Machine Learning** models leveraging historical data to forecast green card processing times, providing applicants with **estimated wait times** from eligible priority date to approval of green card.")
#st.subheader('Descriptive')
#st.write("We will create a novel analytical framework to address immigration issues and provide essential information for aspiring permanent residents in the US. These analytical frameworks will have a decision support system that people can leverage to make better decisions, with a data analytics page displaying trends and insights from immigration data, helping users understand the system and manage expectations.")
# st.subheader('Predictive')
# st.write("The predictive analytics page will allow users to input data and receive personalized estimates on their immigration timeline based on a variety of factors. Through this approach, the DSS will empower applicants and promote a transparent and efficient immigration process.")

#new line and team intro
st.write("")
st.header('The Team')
st.write('**Team Members:** Katrina Apiado, Mahek Karamchandani, Nika Mahdavi, & Boaz Nakhimovsky')
st.write('**Advisor:** Dr. German Serna & Dr. Liz Thompson')
st.write('**Sponsor:** Dr. Puneet Agarwal')
