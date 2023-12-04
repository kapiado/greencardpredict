# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:59:07 2023

@author: katri
"""

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Employee Profile")

st.markdown("# Employee Profile")

c = st.empty()
c.write("The Employee Profile includes key information about the demographic of employees in our dataset, who undergo the green card application process in the United States. This includes their education, salary, nationality, profession, major and layoff history. In the profile, we present visual representations of the data related to Immigration Backlog. We will explore the trend of immigration backlog overtime, in terms of applications received and waiting time for approval, for example. By examining these factors, we aim to gain insights into the patterns and changes within the immigration backlog, enabling a better understanding.")

path = "HTML Files/"


def WTvsYear():
    #HtmlFile = open(path+"AverageWaitingTimebyYear.html", 'r', encoding='utf-8') v1
    HtmlFile = open(path+"Avgwtperyr.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=500, width=700)

def CasesvsYear():
    #HtmlFile = open(path+"AverageWaitingTimebyYear.html", 'r', encoding='utf-8') v1
    HtmlFile = open(path+"Casesperyr.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=500,width=700)

def NAICSvsNumCases():
    #HtmlFile = open(path+"NAICSvsNumCases.html", 'r', encoding='utf-8') v1
    HtmlFile = open(path+"top10OccvsCases.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    # print(source_code)
    st.markdown(source_code, unsafe_allow_html=True)
    # components.html(source_code,height=500, width=1000)
 
def WTvsNAICS():
    HtmlFile = open(path+"AvgWTvsOcc.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    st.markdown(source_code,unsafe_allow_html=True)  

# def WTvsNumCases():
#     #HtmlFile = open(path+"AvgWTvsNumCasesperIndustry(RedLine).html", 'r', encoding='utf-8') #v1
#     HtmlFile = open(path+"AvgWTvsNumCasesperIndustry(RedLine) (2).html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     #components.html(source_code)
#     components.html(source_code,height=1000, width=1000)

    
def WTvsSalary():
    HtmlFile = open(path+"WTvsEmployeeSalary (3).html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=500)

def NumCasesvsSalary():
    HtmlFile = open(path+"NumCasesvsSalary (2).html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=500)

def unitofpay():
    HtmlFile = open(path+"unitofPaygraph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)
    
def nationality():
    HtmlFile = open(path+"WorldMapCases.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600,width=500)

def HighestEducation():
    HtmlFile = open(path+"HighestEducation.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600, width=1500)


# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Year","Industry","Nationality","Salary","Unit of Pay","Highest Education"])
tab1, tab2, tab3 = st.tabs(["Year","Industry","Nationality"])
# container = st.container(border=True)
with tab1:
    # Cache the dataframe so it's only loaded once
    #@st.cache_data
    # container.CasesvsYear()
    # container.WTvsYear()
    CasesvsYear()
    WTvsYear()

with tab2:
    NAICSvsNumCases()
    WTvsNAICS()

with tab3:
    nationality()
    
# with tab4:
#     NumCasesvsSalary()
#     WTvsSalary()
    
# with tab5:
#     unitofpay()
    
# with tab6:
#     HighestEducation()

   
    
    
