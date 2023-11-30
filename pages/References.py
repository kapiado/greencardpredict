# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 23:03:51 2023

@author: katri
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(page_title="References")
st.markdown("# References")

path = "C:/Users/katri/multipage_app/HTML Files/"

# Data Sources
st.header("Data Source")
st.write("")
a = st.empty()
a.write("U.S. Department of Labor, https://www.dol.gov/agencies/eta/foreign-labor/performance")

# Home Page references
st.header("Home Page")
st.write("")
b = st.empty()
b.write('Galetti, Beth. “Amazon Urges Green Card Allocation to Help Immigrant Employees.” Amazon, Aug. 2022, www.aboutamazon.com/news/policy-news-views/amazon-urges-green-card-allocation-to-help-immigrant-employees.')

# 