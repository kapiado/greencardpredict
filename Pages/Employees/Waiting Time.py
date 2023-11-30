# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:34:35 2023

@author: katri
"""

import streamlit.components.v1 as components

def unitofpay():
    HtmlFile = open("unit_of_pay_bar_graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code,height=600)
    
unitofpay()
    