# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 00:37:41 2023

@author: katri
"""

import streamlit as st
import pandas as pd
import csv

st.title("Survey Form")
st.subheader("Enter details below:")

# unconditionally assign values to 'name' and 'phone'
st.session_state.name = st.session_state.get('name', 'Name')
st.session_state.anon = st.session_state.get('anon', 'Anonymous')



with st.form("form1", clear_on_submit=True):
    # name = st.radio(
    #     "Please select",
    #     ('Full Name', 'Anonymous'))
        
        
    # if name == 'Full Name':
    #     st.text_input('Enter your name')

    # else:
    #     name = st.text("Anonymous")


    # nameanon = st.radio("Full Name OR Anonymous",("Full Name","Anonymous"))
    # name = st.text_input()
    # if nameanon == "Anonymous":
    #     name = st.text_input(value="Anonymous")
    # else:
    #     name = st.text_input("Full Name")
    name = st.text_input("Full Name OR Anonymous")
    email = st.text_input("Email (leave blank if Anonymous)")
    message = st.text_area("Please share your feedback with us!")
    rating = st.slider("Rate the usability of this tool (1 = Not usable, 10 = Highly Usable):",min_value=1, max_value=10)
    
    submit = st.form_submit_button("Submit")
    
    if submit:
        # Append the submitted values to a CSV file
        with open('submissions.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name,email,message,rating])
        st.success('Form submitted successfully!')
        
def main():
    pass
    main()