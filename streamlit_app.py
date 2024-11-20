import streamlit as st

st.title('💳 Credit risk checker')

st.info('👈 Please fill your profile in the sidebar')

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Could not find the number of physical cores")

statlog_german_credit_data = fetch_ucirepo(id=144) 
    
# Data (as pandas dataframes) 
features = statlog_german_credit_data.data.features 
targets = statlog_german_credit_data.data.targets -1

amount = features['Attribute5']

plt.hist(amount, bins=30)
plt.show()

import plotly.graph_objects as go

__version__ = "5.1"
def gauge(gVal, gTitle="", gMode='gauge+number', gSize="FULL", gTheme="Black",
          grLow=.29, grMid=.69, gcLow='#FF1708', gcMid='#FF9400', 
          gcHigh='#1B8720', xpLeft=0, xpRight=1, ypBot=0, ypTop=1, 
          arBot=None, arTop=1, pTheme="streamlit", cWidth=True, sFix=None):

    """Deploy Plotly gauge or indicator data visualization

    Keyword arguments:

    gVal -- gauge Value (required)
        Description:
            The value passed to this argument is displayed in
            the center of the visualization, drives the color & position
            of the gauge and is required to successfully call this function.
        Data Type:
            integer, float

    gTitle -- gauge Title (default '')
        Description:
            Adds a header displayed at the top
            of the visualization.
        Data Type:
            string

    gMode -- gauge Mode (default gauge+number)
        Description:
            Declares which type of visualization should
            be displayed.
        Options:
            gauge+number, gauge, number
        Data Type:
            string

    gSize -- gauge Size (default FULL)
        Description:
            Automatically resizes the gauge or indicator using 
            pre-defined values options.

            The size of visualization can also be customized by passing the 'CUST' value to
            the argument and assigning a decimal value from 0 to 1 to the following 
            arguments; xpLeft, xpRight, ypBot, and ypTop.
        Options:
            SML, MED, LRG, FULL, CUST
        Data Type:
            String

    grLow -- Low gauge Range (default 0.30)
        Description:
            Sets the bottom (lowest) percentile target group for the gauge value.  
            When the gauge Value (gVal) is less than the value assigned to this
            argument, the color assigned to the gcLow (Low gauge Color) argument
            is displayed.
        Data Type:
            integer, float

    grMid -- Middle gauge Range (default 0.70)
        Description:
            Sets the middle percentile target group for the gauge value.  When
            the gauge Value (gVal) is less than the value assigned to this argument,
            the color assigned to the gcMid (Middle gauge Color) argument is displayed.
            
            If the value assigned to the gVal argument is greater than or equal to
            the value assigned to the grMid argument, the color value assigned to
            gcHigh will then be displayed.
        Data Type:
            integer, float

    gcLow -- Low gauge Color (default #FF1708)
        Description:
            gauge color for bottom percentile target group. Default value
            is a hex code for red.  Argument excepts hex color codes and 
            there associated names.
        Data Type:
            string

    gcMid -- Middle gauge Color (default #FF9400)
        Description:
            gauge color for middle percentile target group. Default value
            is a hex code for orange.  Argument excepts hex color codes and 
            there associated names.
        Data Type:
            string

    gcHigh -- High gauge Color (default #1B8720)
        Description:
            gauge color for middle percentile target group. Default value
            is a hex code for green.  Argument excepts hex color codes and 
            there associated names.
        Data Type:
            string

    sFix -- gauge Value Suffix (default 0.0)
        Description:
            Adds a suffix (character) to the gauge value displayed in the
            center of the visualization.
            
            Assigning the '%' character to this argument will display the
            percentage symbol at the end of the value shown in the center
            of the visualization and convert the gauge value from a floating
            point integer so the value displays correctly as a percentage.
        Options:
            %
        Data Type:
            string

    xpLeft -- X-Axis Position 1 for Plot (default 0.0)
    xpRight --  X-Axis Position 2 for Plot (default 0.0)
    ypBot --  X-Axis Position 1 for Plot (default 0.0)
    ypTop --  X-Axis Position 2 for Plot (default 0.0)
    arBot -- Bottom Axis Range Value (default 0.0) 
    arTop --  Bottom Axis Range Value (default 0.0)
    pTheme -- Plot Theme (default 0.0)
    cWidth -- Container Width (default 0.0)
    """

    if sFix == "%":

        gaugeVal = round((gVal * 100), 1)
        top_axis_range = (arTop * 100)
        bottom_axis_range = arBot
        low_gauge_range = (grLow * 100)
        mid_gauge_range = (grMid * 100)

    else:

        gaugeVal = gVal
        top_axis_range = arTop
        bottom_axis_range = arBot
        low_gauge_range = grLow
        mid_gauge_range = grMid

    if gSize == "SML":
        x1, x2, y1, y2 =.25, .25, .75, 1
    elif gSize == "MED":
        x1, x2, y1, y2 = .50, .50, .50, 1
    elif gSize == "LRG":
        x1, x2, y1, y2 = .75, .75, .25, 1
    elif gSize == "FULL":
        x1, x2, y1, y2 = 0, 1, 0, 1
    elif gSize == "CUST":
        x1, x2, y1, y2 = xpLeft, xpRight, ypBot, ypTop   

    if gaugeVal <= low_gauge_range: 
        gaugeColor = gcLow
    elif gaugeVal >= low_gauge_range and gaugeVal <= mid_gauge_range:
        gaugeColor = gcMid
    else:
        gaugeColor = gcHigh

    fig1 = go.Figure(go.Indicator(
        mode = gMode,
        value = gaugeVal,
        domain = {'x': [x1, x2], 'y': [y1, y2]},
        number = {"suffix": sFix},
        title = {'text': gTitle},
        gauge = {
            'axis': {'range': [bottom_axis_range, top_axis_range]},
            'bar' : {'color': gaugeColor}
        }
    ))

    config = {'displayModeBar': False}
    fig1.update_traces(title_font_color=gTheme, selector=dict(type='indicator'))
    fig1.update_traces(number_font_color=gTheme, selector=dict(type='indicator'))
    fig1.update_traces(gauge_axis_tickfont_color=gTheme, selector=dict(type='indicator'))
    fig1.update_layout(margin_b=5)
    fig1.update_layout(margin_l=20)
    fig1.update_layout(margin_r=20)
    fig1.update_layout(margin_t=50)

    fig1.update_layout(margin_autoexpand=True)

    st.plotly_chart(
        fig1, 
        use_container_width=cWidth, 
        theme=pTheme, 
        **{'config':config}
    )

def get_probability(credentials_df):
    
# Fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    # Data (as pandas dataframes) 
    features = statlog_german_credit_data.data.features 
    targets = statlog_german_credit_data.data.targets -1
    


    df = pd.DataFrame(features)
    df['Attribute1'].replace({'A11': 1, 'A12': 2, 'A14': 3, 'A13': 4}, inplace=True)
    df['Attribute3'].replace({'A34': 5, 'A32': 3, 'A33': 4, 'A30': 1, 'A31': 2}, inplace=True)
    df['Attribute4'].replace({'A40': 1, 'A41': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10, 'A410': 11}, inplace=True)
    df['Attribute6'].replace({'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5}, inplace=True)
    df['Attribute7'].replace({'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5}, inplace=True)
    df['Attribute9'].replace({'A91': 1, 'A92': 2, 'A93': 3, 'A94': 4}, inplace=True)
    df['Attribute10'].replace({'A101': 1, 'A102': 2, 'A103': 3}, inplace=True)
    df['Attribute12'].replace({'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4}, inplace=True)
    df['Attribute14'].replace({'A141': 1, 'A142': 2, 'A143': 3}, inplace=True)
    df['Attribute15'].replace({'A151': 1, 'A152': 2, 'A153': 3}, inplace=True)
    df['Attribute17'].replace({'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4}, inplace=True)
    df['Attribute19'].replace({'A191': 0, 'A192': 1}, inplace=True)
    df['Attribute20'].replace({'A201': 0, 'A202': 1}, inplace=True)

    df_combined = pd.concat([features, targets], axis=1)


    # Apply the function to remove outliers
    df_cleaned = df_combined

    # Separate features and targets
    X = df_cleaned.drop(columns=['class'])
    y = df_cleaned['class']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance using SMOTE
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Convert the entire dataset to PyTorch tensors
    X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
    y_tensor = torch.tensor(y_resampled.values, dtype=torch.float32)

    # Define the Classifier class
    progress_text = 'Training Data...'
    my_bar = st.progress(0, text=progress_text)
    
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(X_tensor.shape[1], 18)
            self.fc2 = nn.Linear(18, 20)
            self.fc3 = nn.Linear(20, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.fc3(x))
            return x

    # Instantiate the model, criterion, and optimizer
    model = Classifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 2000
    train_losses = []
    train_accuracies = []

    for e in range(epochs):
        my_bar.progress(e/2000, text=progress_text)
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor).squeeze()
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Record training loss
        train_losses.append(loss.item())
        
        # Calculate training accuracy
        with torch.no_grad():
            train_pred = (outputs > 0.5).float()
            train_accuracy = (train_pred == y_tensor).sum().item() / y_tensor.size(0) * 100
            train_accuracies.append(train_accuracy)
    my_bar.empty()

    credentials_df['Attribute1'].replace({'A11': 1, 'A12': 2, 'A14': 3, 'A13': 4}, inplace=True)
    credentials_df['Attribute3'].replace({'A34': 5, 'A32': 3, 'A33': 4, 'A30': 1, 'A31': 2}, inplace=True)
    credentials_df['Attribute4'].replace({'A40': 1, 'A41': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10, 'A410': 11}, inplace=True)
    credentials_df['Attribute6'].replace({'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5}, inplace=True)
    credentials_df['Attribute7'].replace({'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5}, inplace=True)
    credentials_df['Attribute9'].replace({'A91': 1, 'A92': 2, 'A93': 3, 'A94': 4}, inplace=True)
    credentials_df['Attribute10'].replace({'A101': 1, 'A102': 2, 'A103': 3}, inplace=True)
    credentials_df['Attribute12'].replace({'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4}, inplace=True)
    credentials_df['Attribute14'].replace({'A141': 1, 'A142': 2, 'A143': 3}, inplace=True)
    credentials_df['Attribute15'].replace({'A151': 1, 'A152': 2, 'A153': 3}, inplace=True)
    credentials_df['Attribute17'].replace({'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4}, inplace=True)
    credentials_df['Attribute19'].replace({'A191': 0, 'A192': 1}, inplace=True)
    credentials_df['Attribute20'].replace({'A201': 0, 'A202': 1}, inplace=True)

    

    # Scale and predict
    credentials_scaled = scaler.transform(credentials_df)
    credentials_tensor = torch.tensor(credentials_scaled, dtype=torch.float32)
    model.eval()


    with torch.no_grad():
        probability = model(credentials_tensor).item()

    return probability*100

my_credentials = {
    'Attribute1': ['A13'],
    'Attribute2': [36],
    'Attribute3': ['A32'],
    'Attribute4': ['A46'],
    'Attribute5': [4500],
    'Attribute6': ['A63'],
    'Attribute7': ['A75'],
    'Attribute8': [0.0005],
    'Attribute9': ['A94'],
    'Attribute10': ['A103'],
    'Attribute11': [5],
    'Attribute12': ['A121'],
    'Attribute13': [22],
    'Attribute14': ['A141'],
    'Attribute15': ['A152'],
    'Attribute16': [3],
    'Attribute17': ['A172'],
    'Attribute18': [2],
    'Attribute19': ['A192'],
    'Attribute20': ['A202'],
}

my_credentials_df = pd.DataFrame(my_credentials)


with st.sidebar:
    st.header('Input Features')
    Attribute1 = st.selectbox('Status of existing checking account', ('less than £0', 'between £0 and £150', 'greater than £150', 'no existing checking account'),index=None, placeholder='Select one...' )
    Attribute2 = st.number_input("Credit Duration (in months)")
    Attribute3 = st.selectbox('Credit history', ('No existing (ongoing) credits anywhere', 'No existing (ongoing) credits in current bank', 'Existing credits being paid duly', 'Delay in paying credits in the past', 'Critical Account'),index=None, placeholder='Select one...')
    Attribute4 = st.selectbox('Purpose of Credit', ('Car (new)', 'Car(used)', 'Furniture/Equipment', 'Radio/television', 'Other appliances', 'Repairs', 'Education', 'Vacation', 'Retraining', 'Business', 'Other reason'),index=None, placeholder='Select one...')
    Attribute5 = st.number_input("Credit Amount (in GBP)")
    Attribute6 = st.selectbox('Balance in Savings account / Bonds', ('Less than £1000', 'Between £1000 and £5000', 'Between £5000 and £10000', 'Greater than £10000', 'No Savings Account'),index=None, placeholder='Select one...')
    Attribute7 = st.selectbox('Present Employment Since', ('Unemployed', '< 1 year', '< 4 years and >= 1 year', '< 7 years and >= 4 years', 'more than 7 years'),index=None, placeholder='Select one...')
    Attribute8 = st.number_input('Installment rate in percentage to disposable income', value=0)
    Attribute9 = st.selectbox('Personal status and Sex', ('Male - divorced/separated', 'Female - divorced/separated/married', 'Male - single', 'Male - married/widowed', 'Female - single'),index=None, placeholder='Select one...')
    Attribute10 = st.selectbox('Other debtors/guarantors', ('None', 'Co-applicant', 'Guarantor'),index=None, placeholder='Select one...')
    Attribute11 = st.number_input('Present residence since', 0, 4)
    Attribute12 = st.selectbox('Collateral', ('Real Estate', 'savings agreement/ life insurance', 'Car or other', 'No Collateral'),index=None, placeholder='Select one...')
    Attribute13 = st.number_input("Age (in years)",step=1)
    Attribute14 = st.selectbox('Other Installment plans', ('Bank', 'Stores', 'None'),index=None, placeholder='Select one...')
    Attribute15 = st.selectbox('Current housing', ('Rent', 'Own', 'other'),index=None, placeholder='Select one...')
    Attribute16 = st.slider('Number of existing credits in this bank', 0, 4)
    Attribute17 = st.selectbox('Skill level', ('unemployed/ unskilled  - non-resident', 'unskilled - resident', 'skilled employee / official', 'management/ self-employed/highly qualified employee/ officer'),index=None, placeholder='Select one...')
    Attribute18 = st.slider('Dependents', 0, 2)
    Attribute19 = st.selectbox('Telephone', ('Yes', 'No'),index=None, placeholder='Select one...')
    Attribute20 = st.selectbox('Foreign Worker', ('Yes', 'No'),index=None, placeholder='Select one...')

   
    input_df = pd.DataFrame({"Attribute1" : [Attribute1], "Attribute2" : [Attribute2], "Attribute3" : [Attribute3], "Attribute4" : [Attribute4], "Attribute5" : [Attribute5], "Attribute6" : [Attribute6], "Attribute7" : [Attribute7], "Attribute8" : [Attribute8], "Attribute9" : [Attribute9], "Attribute10" : [Attribute10], "Attribute11" : [Attribute11], "Attribute12" : [Attribute12], "Attribute13" : [Attribute13], "Attribute14" : [Attribute14], "Attribute15": [Attribute15], "Attribute16" : [Attribute16], "Attribute17" : [Attribute17], "Attribute18" : [Attribute18], "Attribute19" : [Attribute19], "Attribute20" : [Attribute20]})
    input_df['Attribute1'].replace({'less than £0': 'A11', 'between £0 and £150': 'A12', 'greater than £150': 'A13','no existing checking account': 'A14'}, inplace=True)
    input_df['Attribute3'].replace({'No existing (ongoing) credits anywhere': 'A30', 'No existing (ongoing) credits in current bank': 'A31', 'Existing credits being paid duly': 'A32', 'Delay in paying credits in the past': 'A33', 'Critical Account': 'A34'}, inplace=True)
    input_df['Attribute4'].replace({'Car (new)': 'A40', 'Car(used)': 'A41', 'Furniture/Equipment': 'A42', 'Radio/television': 'A43', 'Other appliances': 'A44', 'Repairs': 'A45', 'Education': 'A46', 'Vacation': 'A47', 'Retraining': 'A48', 'Business': 'A49', 'Other reason': 'A410'}, inplace=True)
    input_df['Attribute6'].replace({'Less than £1000': 'A61', 'Between £1000 and £5000': 'A62', 'Between £5000 and £10000': 'A63', 'Greater than £10000': 'A64', 'No Savings Account': 'A65'}, inplace=True)    
    input_df['Attribute7'].replace({'Unemployed': 'A71', '< 1 year': 'A72', '< 4 years and >= 1 year': 'A73', '< 7 years and >= 4 years': 'A74', 'more than 7 years': 'A75'}, inplace=True)
    input_df['Attribute9'].replace({'Male - divorced/separated': 'A91', 'Female - divorced/separated/married': 'A92', 'Male - single': 'A93', 'Male - married/widowed': 'A94', 'Female - single': 'A95'}, inplace=True)
    input_df['Attribute10'].replace({'None': 'A101', 'Co-applicant': 'A102', 'Guarantor': 'A103'}, inplace=True)
    input_df['Attribute12'].replace({'Real Estate': 'A121', 'savings agreement/ life insurance': 'A122', 'Car or other': 'A123', 'No Collateral': 'A124'}, inplace=True)
    input_df['Attribute14'].replace({'Bank': 'A141', 'Stores': 'A142', 'None': 'A143'}, inplace=True)
    input_df['Attribute15'].replace({'Rent': 'A151', 'Own': 'A152', 'other': 'A153'}, inplace=True)
    input_df['Attribute17'].replace({'unemployed/ unskilled  - non-resident': 'A171', 'unskilled - resident': 'A172', 'skilled employee / official': 'A173', 'management/ self-employed/highly qualified employee/ officer': 'A174'}, inplace=True)
    input_df['Attribute19'].replace({'Yes': 'A192', 'No': 'A191'}, inplace=True)
    input_df['Attribute20'].replace({'Yes': 'A201', 'No': 'A202'}, inplace=True)
    




if st.button('Train Model'): 
    prob = get_probability(input_df)
    if np.isnan(prob):
        st.warning('⚠️ Please fill out the form in its entirety')
    else:
        
        st.write(f'Your probability of default is {prob:.1f} %')
        gauge(
        gVal=prob/100,           # The value to display on the gauge
        gMode='gauge+number', # Display mode ('gauge+number', 'gauge', 'number')
        gSize="MED",         # Size of the gauge visualization
        gTheme="Black",       # Theme color for text and gauge labels
        grLow=0.3,            # Low threshold for range
        grMid=0.7,            # Mid threshold for range
        gcLow='#1B8720',      # Color for low range
        gcMid='#FF9400',      # Color for mid range
        gcHigh='#FF1708',     # Color for high range
        arBot=0,              # Minimum value on the gauge
        arTop=1,              # Maximum value on the gauge
        sFix="%",             # Suffix to append to the value (e.g., '%')
    )
    
        st.divider()
    
        if prob <= 10:
            st.write('The predicted credit default risk is very low, indicating a highly trustworthy financial profile. This borrower demonstrates consistent financial behavior and stability, making them an excellent candidate for credit approvals with favorable terms.')
        elif prob >10 and prob<=30:
            st.write('The predicted credit default risk is low, suggesting that the borrower has a strong ability to meet their financial obligations. While there may be a few minor concerns, they generally pose a low likelihood of default.')
        elif prob > 30 and prob <= 70:
            st.write('The predicted credit default risk is moderate, indicating a balanced credit profile. While there are some notable risk factors, the borrower has the potential to manage their financial obligations with appropriate measures.')
        elif prob > 70 and prob <= 90:
            st.write('The predicted credit default risk is high, highlighting notable financial challenges. There are several risk indicators that increase the probability of default, suggesting the need for caution and potential risk mitigation strategies.')
        elif prob>90:
            st.write('The predicted credit default risk is very high, signaling a highly vulnerable financial profile. The borrower demonstrates severe risk factors that make credit extension highly inadvisable without substantial safeguards.')
            
    
        
    
       
