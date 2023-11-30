# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:26:27 2023

@author: katri
"""

import streamlit as st
import streamlit.components.v1 as components


# st.set_page_config(
#     page_title="Background",
#     page_icon="👋",
# )
#st.markdown("# Overview")
st.set_page_config(page_title="Overview")
st.write("# Overview")

path = "C:/Users/kapia/multipage_app/HTML Files/"

def cases():
    HtmlFile = open(path+"CasesReceived.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=480)
    
def avgwait():    
    HtmlFile = open(path+"AvgWaitingTime.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)

cases()
avgwait()

a = st.empty()
a.write("The North American Industry Classification System (NAICS) is a standardized system used to classify business establishments based on their economic activity in Canada, Mexico, and the United States. It provides a hierarchical structure that groups businesses into various sectors, subsectors, industry groups, and industries. The NAICS code is a unique numerical identifier assigned to each business entity, allowing for consistent and comparable data collection and analysis across different industries and regions.")

# st.header("Columns Used")
# st.markdown(
#     """
#     - CASE_NO
#     - DECISION_DATE
#     - CASE_STATUS
#     - 2_NAICS
#     - PW_SOC_CODE
#     - PW_LEVEL_9089
#     - PW_AMOUNT_9089
#     - PW_UNIT_OF_PAY_9089
#     - JOB_INFO_WORK_CITY
#     - JOB_INFO_WORK_STATE
#     - COUNTRY_OF_CITIZENSHIP
#     - CLASS_OF_ADMISSION
#     - CASE_RECEIVED_DATE
#     - EMPLOYER_NUM_EMPLOYEES
#     - PW_DETERM_DATE
#     - PW_EXPIRE_DATE
#     - JOB_INFO_EDUCATION
#     - JOB_INFO_MAJOR
#     - JOB_INFO_TRAINING
#     - JOB_INFO_EXPERIENCE
#     - JOB_INFO_EXPERIENCE_NUM_MONTHS
#     - JOB_INFO_FOREIGN_ED
#     - JI_OFFERED_TO_SEC_J_FOREIGN_WORKER
#     - RECR_INFO_PROFESSIONAL_OCC
#     - RI_LAYOFF_IN_PAST_SIX_MONTHS
#     - FOREIGN_WORKER_INFO_EDUCATION
#     - FOREIGN_WORKER_INFO_MAJOR
#     - FW_INFO_YR_REL_EDU_COMPLETED
#     - FW_INFO_REQ_EXPERIENCE
#     - CASE_RECEIVED_YEAR
#     - CASE_RECEIVED_MONTH
#     """
#     )

# st.sidebar.success("Select a page above.")

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )