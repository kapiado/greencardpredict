# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:04:25 2023

@author: katri
"""

import streamlit as st
import streamlit.components.v1 as components

  
st.set_page_config(page_title="Employer Profile")
st.markdown("# Employer Profile")

path = "C:/Users/kapia/multipage_app/HTML Files/"

# def majorswait():
#     HtmlFile = open("major_graph.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     components.html(source_code,height=1300)

# def numemployeeswaitingtime():
#     HtmlFile = open("number_of_employee_and_waiting_time_bar_graph.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     print(source_code)
#     components.html(source_code,height=600)

a = st.empty()
a.write("The Employer Profile includes key information about the employers in our dataset, who assist in the green card application process in the United States. This consists of the experience required for a job position, number of employees per application, and location (U.S. state/territory).") #major requirements
     
def WTvsExpReq():
    HtmlFile = open(path+"WTvsExpReq (2).html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)
    
def NumCasesvsExpReq():
    HtmlFile = open(path+"CasesvsExp.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)
 
def NumCasesvsNumEmp():
    HtmlFile = open(path+"CasesvsNumEmp.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)

def WTvsNumEmp():
    HtmlFile = open(path+"WTvsNumEmp.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)

def NumCasesvsJobState():
    HtmlFile = open(path+"NumCasesvsJobState.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)
    
    
def WTvsJobState():
    HtmlFile = open(path+"WTvsJobState.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=800,width=1000)



#a = st.empty()
#a.write("Insert details on employer profile and how it is attained here")

tab1, tab2, tab3 = st.tabs(["Experience Required","Number of Employees","Employer Location"])

with tab1:
   a = st.empty()
   a.write("'Experience Required' identifies whether experience in the job offered by the employer is a requirement.")
   WTvsExpReq()
   #st.image("https://static.streamlit.io/examples/cat.jpg")

with tab2:
    a = st.empty()
    a.write("'Number of Employees' identifies the total number of employees employed by the employer.")
    NumCasesvsNumEmp()
    WTvsNumEmp()
   #st.image("https://static.streamlit.io/examples/dog.jpg")
   
with tab3:
    NumCasesvsJobState()
    WTvsJobState()
