# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:16:53 2023

@author: kapia
"""

# Import libraries
import pandas as pd             # Pandas
import streamlit as st          # Streamlit
import matplotlib.pyplot as plt # Matplotlib
import seaborn as sns           # Seaborn

# Module to save and load Python objects to and from files
import pickle 
import joblib

# Package to implement Random Forest Model
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings('ignore')

st.title('Estimate Your Green Card Application Waiting Time') 

st.write("This app uses 12 inputs to predict the waiting time (months) for your green card application. Use the form below to get started!")
         # "a model built on the Palmer's Penguin's dataset. Use the form below" 

df2 = pd.read_csv("pages/11_30_23_Pred_Data_Final1.csv")
#print(df2)
df2.drop(['WAITING_TIMERANGE'], axis=1, inplace=True)

# Loading model and mapping pickle files
rf_pickle = open("pages/11_30_23_rf_model_final.pkl", 'rb') 
#rf_model = pd.read_pickle(r"pages/11_30_23_rf_model_final.sav")
# with open("pages/11_30_23_rf_model_final.sav", 'rb') as f:
#     rf_model = pickle.load(f)
rf_model = pickle.load(rf_pickle) 
rf_pickle.close() 
# Asking users to input their own data
# penguin_file = st.file_uploader('Upload your own penguin data to train the model') 

# Display an example dataset and prompt the user 
# to submit the data in the required format.
# st.write("Please ensure that your data adheres to this specific format:")

# Cache the dataframe so it's only loaded once
# @st.cache_data
# def load_data(filename):
#   df = pd.read_csv(filename)
#   return df

# data_format = load_data('dummieCodes.csv')
# st.dataframe(data_format, hide_index = True)

# Setting the default option to load our Decision Tree model
# if there is no penguin file uploaded by the user
# if penguin_file is None: 
#     # Load default dataset
#     # This dataset will be used later for plotting histograms
#     # in case if the user does not provide any data
#     default_df = pd.read_csv('dummieCodex.csv') 


# # If the file is provided, we need to clean it and train a model on it
# # similar to what we did in the Jupyter notebook
# else: 
#     # Load dataset as dataframe
#     penguin_df = pd.read_csv(penguin_file) 
#     # Dropping null values
#     penguin_df = penguin_df.dropna() 
#     # Output column for prediction
#     output = penguin_df['WAITING_TIMERANGE'] 
#     # Input features (excluding year column)
#     features = penguin_df[['NAICS_CODE','PW_LEVEL','PW_AMOUNT','WORK_STATE','COUNTRY_OF_CITIZENSHIP','EMPLOYER_NUM_EMPLOYEES','CLASS_OF_ADMISSION',
#             'JOB_EDUCATION','EXPERIENCE','EXPERIENCE_MONTHS','LAYOFF_IN_PAST_SIX_MONTHS','WORKER_EDUCATION']] 
#     # One-hot-encoding for categorical variables
#     features = pd.get_dummies(features) 
#     # Factorize output feature (convert from string to number)
#     output, unique_penguin_mapping = pd.factorize(output) 
#     # Data partitioning into training and testing 
#     train_X, test_X, train_y, test_y = train_test_split(features, output, test_size = 0.2, random_state = 1) 
#     # Defining prediction model
#     clf = DecisionTreeClassifier(random_state = 0)
#     # Fitting model on training data
#     clf.fit(train_X, train_y) 
#     # Making predictions on test set
#     y_pred = clf.predict(test_X) 
#     # Calculating F1-score of the model on test set
#     score = round(f1_score(y_pred, test_y, average = 'macro'), 2) 
#     st.write('We trained a Decision Tree model on these data,' 
#              ' it has an F1-score of {}! Use the ' 
#              'inputs below to try out the model.'.format(score))


    # Some code
with st.form(key='my_form'):
    codeOptions = ['11 - Agriculture, Forestry, Fishing and Hunting', 
                   '21 - Mining, Quarrying, and Oil and Gas Extraction', 
                   '22 - Utilities', 
                   '23 - Construction', 
                   '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)', 
                   '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)', 
                   '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)', 
                   '42 - Wholesale Trade',
                   '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)',
                   '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)', 
                   '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)', 
                   '49 - Transportation and Warehousing (Federal Government-Operated Postal Services, Couriers, Messengers, Warehousing Storage-Related Services)',
                   '51 - Information',
                   '52 - Finance and Insurance',
                   '53 - Real Estate and Rental and Leasing',
                   '54 - Professional, Scientific, and Technical Services',
                   '55 - Management of Companies and Enterprises',
                   '56 - Administrative and Support and Waste Management and Remediation Services',
                   '61 - Educational Services',
                   '62 - Health Care and Social Assistance',
                   '71 - Arts, Entertainment, and Recreation',
                   '72 - Accommodation and Food Services',
                   '81 - Other Services (except Public Administration)',
                   '92 - Public Administration']
    

# To get the selected value from the select box
    codeInfo = st.selectbox('NAICS Code', codeOptions, help="Select most appropriate Industry Code as found here https://www.census.gov/naics/?58967?yearbck=2022")

    # Initialize variables for NAICS Codes
# Assign numerical values based on specific strings in the NAICS Code
    education_options = [
"High School", "Associate's", "Bachelor's", "Doctorate", 
"Master's", "None", "Other"
]
    # was job education -- how to address this since this is the education required by the job
    educationInfo = st.selectbox('Highest Completed Education Level', options=education_options) 
    
    wage_levels = ["Level I", "Level II", "Level III", "Level IV"]
    wagelevelInfo = st.selectbox('Prevailing Wage Level', wage_levels, help = "Select most appropriate prevailing wage level")

    wageamountInfo = st.number_input('Prevailing Wage Amount', min_value = 0)
    admiclasses = [
"A-3", "A1/A2", "B-1", "B-2", "C-1", "C-3", "CW-1", "D-1", "D-2", 
"E-1", "E-2", "E-3", "EWI", "F-1", "F-2", "G-1", "G-4", "G-5", 
"H-1A", "H-1B", "H-1B1", "H-2A", "H-2B", "H-3", "H-4", "I", 
"J-1", "J-2", "K-4", "L-1", "L-2", "M-1", "M-2", "N", "Not in USA", 
"O-1", "O-2", "O-3", "P-1", "P-3", "P-4", "Parolee", "Q", 
"R-1", "R-2", "T-1", "T-2", "TD", "TN", "TPS", "U-1", "V-2", 
"VWB", "VWT"
]

    admiclassInfo = st.selectbox('Class of Admission', admiclasses)
    
    countryInfo = st.selectbox('Country of Citizenship', options=[
"ARGENTINA", "AUSTRALIA", "BANGLADESH", "BELARUS", "BRAZIL", 
"BULGARIA", "CANADA", "CHILE", "CHINA", "COLOMBIA", "ECUADOR", 
"EGYPT", "FRANCE", "GERMANY", "GREECE", "HONG KONG", "INDIA", 
"INDONESIA", "IRAN", "IRELAND", "ISRAEL", "ITALY", "JAPAN", 
"JORDAN", "LEBANON", "MALAYSIA", "MEXICO", "NEPAL", "NETHERLANDS", 
"NIGERIA", "PAKISTAN", "PERU", "PHILIPPINES", "POLAND", "ROMANIA", 
"RUSSIA", "SINGAPORE", "SOUTH AFRICA", "SOUTH KOREA", "SPAIN", 
"SRI LANKA", "SWEDEN", "SYRIA", "TAIWAN", "THAILAND", "TURKEY", 
"UKRAINE", "UNITED KINGDOM", "VENEZUELA", "VIETNAM"
])

    stateInfo = st.selectbox('U.S. Work State',
                                                  [
    'ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO', 'CONNECTICUT', 'DELAWARE',
    'DISTRICT OF COLUMBIA', 'FLORIDA', 'GEORGIA', 'GUAM', 'HAWAII', 'IDAHO', 'ILLINOIS', 'INDIANA',
    'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA', 'MAINE', 'MARSHALL ISLANDS', 'MARYLAND', 'MASSACHUSETTS',
    'MICHIGAN', 'MINNESOTA', 'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA', 'NEW HAMPSHIRE',
    'NEW JERSEY', 'NEW MEXICO', 'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'NORTHERN MARIANA ISLANDS',
    'OHIO', 'OKLAHOMA', 'OREGON', 'PENNSYLVANIA', 'PUERTO RICO', 'RHODE ISLAND', 'SOUTH CAROLINA',
    'SOUTH DAKOTA', 'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGIN ISLANDS', 'VIRGINIA', 'WASHINGTON',
    'WEST VIRGINIA', 'WISCONSIN', 'WYOMING'
], 
                             help = "Select the U.S. state of primary worksite")

    employeenumInfo = st.number_input('Number of Employees at Company', min_value = 0)


    jobeducation_options = [
"High School", "Associate's", "Bachelor's", "Doctorate", 
"Master's", "None", "Other"] 
    # was job education -- how to address this since this is the education required by the job
    jobeducationInfo = st.selectbox('Education Level Required by Job', options=jobeducation_options) 

    expInfo = st.radio('Do you have job/industry experience?', options=["Yes","No"])
    
    expmonthsInfo = st.number_input('Months of Experience', min_value = 0, help = "Input how many months of job experience you have")
    
    layoffInfo =  st.radio('Have you been affected from layoff(s) in the past six months?', options =["Yes","No"])



    submit = st.form_submit_button('Submit', args=(1,
                    [codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo]))
# columns
# 'NAICS_CODE', 'PW_LEVEL', 'PW_AMOUNT', 'WORK_STATE',
#        'COUNTRY_OF_CITIZENSHIP', 'EMPLOYER_NUM_EMPLOYEES',
#        'CLASS_OF_ADMISSION', 'JOB_EDUCATION', 'EXPERIENCE',
#        'EXPERIENCE_MONTHS', 'LAYOFF_IN_PAST_SIX_MONTHS', 'WORKER_EDUCATION',
#        'WAITING_TIMERANGE'



# NAICS_CODE,PW_LEVEL,PW_AMOUNT,WORK_STATE,COUNTRY_OF_CITIZENSHIP,EMPLOYER_NUM_EMPLOYEES,CLASS_OF_ADMISSION,JOB_EDUCATION,EXPERIENCE,EXPERIENCE_MONTHS,LAYOFF_IN_PAST_SIX_MONTHS,WORKER_EDUCATION,WAITING_TIMERANGE
# NAICS_CODE,PW_LEVEL,PW_AMOUNT,WORK_STATE,COUNTRY_OF_CITIZENSHIP,EMPLOYER_NUM_EMPLOYEES,CLASS_OF_ADMISSION,JOB_EDUCATION,EXPERIENCE,EXPERIENCE_MONTHS,LAYOFF_IN_PAST_SIX_MONTHS,WORKER_EDUCATION,WAITING_TIMERANGE
df3 = df2.copy()
df3.loc[len(df3)] = [codeInfo, wagelevelInfo, wageamountInfo, stateInfo, countryInfo, employeenumInfo,  admiclassInfo,  jobeducationInfo, expInfo, expmonthsInfo, layoffInfo, educationInfo]
# Create dummies for encode_df
cat_var = ['NAICS_CODE', 'PW_LEVEL','WORK_STATE','COUNTRY_OF_CITIZENSHIP','CLASS_OF_ADMISSION','JOB_EDUCATION','EXPERIENCE','LAYOFF_IN_PAST_SIX_MONTHS','WORKER_EDUCATION']
df3 = pd.get_dummies(df3, columns = cat_var)
# Extract encoded user data
user_encoded_df = df3.tail(1)

st.subheader("Predicting Waiting Time")


    # Using DT to predict() with encoded user data
new_prediction_rf = rf_model.predict(user_encoded_df)
new_prediction_prob_dt = rf_model.predict_proba(user_encoded_df).max()
# Show the predicted cost range on the app
st.write("Random Forest Prediction: {}".format(*new_prediction_rf))
st.write("Prediction Probability: {:.0%}".format(new_prediction_prob_rf))

# Showing additional items
st.subheader("Prediction Performance")
tab1 = st.tabs(["Feature Importance"])
# ONLY INCLUDE FEATURE IMPORTANCE
# with tab1:
#     st.image('dt_feature_imp.svg')


# # Dictionary to map category names to numerical values
# # code_mapping = {
# #     '11 - Agriculture, Forestry, Fishing and Hunting': 1,
# #     '21 - Mining, Quarrying, and Oil and Gas Extraction': 2,
# #     '22 - Utilities': 3,
# #     '23 - Construction': 4,
# #     '31 - Manufacturing (Food, Beverage, Tobacco, Apparel, Leather, Textiles)': 5,
# #     '32 - Manufacturing (Paper, Printing, Petroleum, Coal, Chemicals, Plastics, Rubber, Nonmetallic)': 6,
# #     '33 - Manufacturing (Primary Metals, Fabricated Metal, Machinery, Computer and Electronic Products, Electrical Equipment and Appliances, Transportations Equipment, Furniture, Miscellaneous Manufacturing)': 7,
# #     '42 - Wholesale Trade': 8,
# #     '44 - Retail Trade (Automotive Sales and Services, Home Furnishing and Improvement, Food and Beverage, Health and Personal Care, Clothing and Accessories, Gasoline Stations)': 9,
# #     '45 - Retail Trade (Sporting Goods, Hobbies, Books, Department Stores, General Merchandise Stores, Florists, Office Supplies, Pet Supplies, Art Dealers, Various Specialty Stores)': 10,
# #     '48 - Transportation and Warehousing (Air, Rail, Water, Truck, Transit, Pipeline, Scenic and Sightseeing Services, Transportation Support Activities)': 11,
# #     '49 - Transportation and Warehousing (Federal Government-Operated Postal Services, Couriers, Messengers, Warehousing Storage-Related Services)': 12,
# #     '51 - Information': 13,
# #     '52 - Finance and Insurance': 14,
# #     '53 - Real Estate and Rental and Leasing': 15,
# #     '54 - Professional, Scientific, and Technical Services': 16,
# #     '55 - Management of Companies and Enterprises': 17,
# #     '56 - Administrative and Support and Waste Management and Remediation Services': 18,
# #     '61 - Educational Services': 19,
# #     '62 - Health Care and Social Assistance': 20,
# #     '71 - Arts, Entertainment, and Recreation': 21,
# #     '72 - Accommodation and Food Services': 22,
# #     '81 - Other Services (except Public Administration)': 23,
# #     '92 - Public Administration': 24
# # }
# # # for codes
# # if codeInfo in code_mapping.keys():
# #     codeInfo = code_mapping[codeInfo]
# # else:
# #     st.error("Selected NAICS code not found in mapping.")
    
# # # Dictionary to map education levels
# # education_mapping = {
# #     "High School": 1,
# #     "Associate's": 2,
# #     "Bachelor's": 3,
# #     "Doctorate": 4,
# #     "Master's": 5,
# #     "None": 6,
# #     "Other": 7
# # }

# # if educationInfo in education_mapping:
# #     educationInfo = education_mapping[educationInfo]
# # else:
# #     st.error("Education not found in mapping.")
    
# # # Dictionary to map wage levels
# # wage_level_mapping = {
# #     "1": 1,
# #     "2": 2,
# #     "3": 3,
# #     "4": 4
# # }
# # if wagelevelInfo in wage_level_mapping:
# #     wagelevelInfo = wage_level_mapping[wagelevelInfo]
# # else:
# #     st.error("Wage level not found in mapping.")

# # admission_class_mapping = {
# #     "A-3": 1,
# #     "A1/A2": 2,
# #     "B-1": 3,
# #     "B-2": 4,
# #     "C-1": 5,
# #     "C-3": 6,
# #     "CW-1": 7,
# #     "D-1": 8,
# #     "D-2": 9,
# #     "E-1": 10,
# #     "E-2": 11,
# #     "E-3": 12,
# #     "EWI": 13,
# #     "F-1": 14,
# #     "F-2": 15,
# #     "G-1": 16,
# #     "G-4": 17,
# #     "G-5": 18,
# #     "H-1A": 19,
# #     "H-1B": 20,
# #     "H-1B1": 21,
# #     "H-2A": 22,
# #     "H-2B": 23,
# #     "H-3": 24,
# #     "H-4": 25,
# #     "I": 26,
# #     "J-1": 27,
# #     "J-2": 28,
# #     "K-4": 29,
# #     "L-1": 30,
# #     "L-2": 31,
# #     "M-1": 32,
# #     "M-2": 33,
# #     "N": 34,
# #     "Not in USA": 35,
# #     "O-1": 36,
# #     "O-2": 37,
# #     "O-3": 38,
# #     "P-1": 39,
# #     "P-3": 40,
# #     "P-4": 41,
# #     "Parolee": 42,
# #     "Q": 43,
# #     "R-1": 44,
# #     "R-2": 45,
# #     "T-1": 46,
# #     "T-2": 47,
# #     "TD": 48,
# #     "TN": 49,
# #     "TPS": 50,
# #     "U-1": 51,
# #     "V-2": 52,
# #     "VWB": 53,
# #     "VWT": 54
# # }
# # if admiclassInfo in admission_class_mapping:
# #     admiclassInfo = admission_class_mapping[admiclassInfo]
# # else:
# #     st.error("Admission class not found in mapping.")



# # #codeInfo, educationInfo, wagelevelInfo, wageamountInfo, admiclassInfo, countryInfo, stateInfo, jobeducationInfo, employeenumInfo,  expInfo, expmonthsInfo, layoffInfo
# # user_data = pd.DataFrame({
# #     'NAICS_CODE': [codeInfo],
# #     'WORKER_EDUCATION': [educationInfo],
# #     'PW_LEVEL': [wagelevelInfo],
# #     'PW_AMOUNT': [wageamountInfo],
# #     'CLASS_OF_ADMISSION': [admiclassInfo],
# #     'COUNTRY_OF_CITIZENSHIP': [countryInfo],
# #     'WORK_STATE': [stateInfo],
# #     'JOB_EDUCATION': [jobeducationInfo],
# #     'EMPLOYER_NUM_EMPLOYEES': [employeenumInfo],
# #     'EXPERIENCE': [expInfo],
# #     'EXPERIENCE_MONTHS': [expmonthsInfo],
# #     'LAYOFF_IN_PAST_SIX_MONTHS': [layoffInfo]})
    
# # # Assuming categorical columns are 'NAICS_CODE', 'EDUCATION_LEVEL', 'WAGE_LEVEL', etc.
# # encoded_columns = ['NAICS_CODE', 'PW_LEVEL', 'WORK_STATE',
# #        'COUNTRY_OF_CITIZENSHIP',
# #        'CLASS_OF_ADMISSION', 'JOB_EDUCATION', 'EXPERIENCE',
# #         'LAYOFF_IN_PAST_SIX_MONTHS', 'WORKER_EDUCATION']  # Update with all categorical columns
# # encoded_user_data = pd.get_dummies(user_data, columns=encoded_columns)

# # Make predictions
# # new_prediction_dt = dt_model.predict(encoded_user_data)
# # new_prediction_prob_dt = dt_model.predict_proba(encoded_user_data).max()

# default_df = pd.read_csv('dummieCodes.csv') 
# user_encoded_df = default_df.copy()

# # Combine the list of user data as a row to default_df

# encode_df.loc[len(encode_df)] = [codeInfo, 
#                                   educationInfo, wagelevelInfo, wageamountInfo, admiclassInfo, countryInfo, stateInfo, jobeducationInfo, employeenumInfo,  expInfo, expmonthsInfo, layoffInfo]


# # Create dummies for encode_df
# cat_var = ['NAICS_CODE', 'WORKER_EDUCATION','PW_LEVEL', 'CLASS_OF_ADMISSION',
#         'COUNTRY_OF_CITIZENSHIP', 'WORK_STATE',
#          'JOB_EDUCATION', 'EXPERIENCE',
#         'LAYOFF_IN_PAST_SIX_MONTHS', ]
# encode_dummy_df = pd.get_dummies(cat_var, columns = cat_var)
# # Extract encoded user data
# user_encoded_df = encode_dummy_df.tail(1)



# # Showing additional items
# st.subheader("Prediction Performance")
# tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])
# ONLY INCLUDE FEATURE IMPORTANCE
# with tab1:
#     st.image('dt_feature_imp.svg')
# with tab2:
#     st.image('dt_confusion_mat.svg')
# with tab3:
#     df = pd.read_csv('dt_class_report.csv', index_col = 0)
#     st.dataframe(df)

# else:
#     # Using RF to predict() with encoded user data
#     new_prediction_rf = rf_model.predict(user_encoded_df)
#     new_prediction_prob_rf = rf_model.predict_proba(user_encoded_df).max()
#     # Show the predicted cost range on the app
#     st.write("Random Forest Prediction: {}".format(*new_prediction_rf))
#     st.write("Prediction Probability: {:.0%}".format(new_prediction_prob_rf))


# new_prediction = clf.predict([[codeInfo, educationInfo, wagelevelInfo, wageamountInfo, admiclassInfo, countryInfo, stateInfo, jobeducationInfo, employeenumInfo,  expInfo, expmonthsInfo, layoffInfo]]) 

# # Map prediction with penguin species
# prediction_species = unique_penguin_mapping[new_prediction][0]
# st.subheader("Predicting Your Application Wait Time")
# st.write('We predict the waiting time of your application to be {} months'.format(prediction_species)) 

# # Adding histograms for continuous variables for model explanation
# st.write('Below are the histograms for each continuous variable '
#          'separated by penguin species. The vertical line '
#          'represents your inputted value.')

# fig, ax = plt.subplots()
# ax = sns.displot(x = penguin_df['PW_AMOUNT'], hue = penguin_df['WAITING_TIMERANGE'])
# plt.axvline(wageamountInfo)
# plt.title('Bill Length by Species')
# st.pyplot(ax)

# fig, ax = plt.subplots()
# ax = sns.displot(x = penguin_df['EMPLOYER_NUM_EMPLOYEES'], hue = penguin_df['species'])
# plt.axvline(employeenumInfo)
# plt.title('Bill Depth by Species')
# st.pyplot(ax)

# fig, ax = plt.subplots()
# ax = sns.displot(x = penguin_df['EXPERIENCE_MONTHS'], hue = penguin_df['species'])
# plt.axvline(expmonthsInfo)
# plt.title('Bill Flipper by Species')
# st.pyplot(ax)

# NOTE: sns.distplot() function accepts the data variable as an argument 
# and returns the plot with the density distribution