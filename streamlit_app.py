import streamlit as st

st.title('💳 Credit risk checker')

st.info('This program uses machine learning to check for credit risk')

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


def get_probability(credentials_df):
    print('downloading test data')
 
# Fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    # Data (as pandas dataframes) 
    features = statlog_german_credit_data.data.features 
    targets = statlog_german_credit_data.data.targets -1
    print('test data downloaded, preprocessing...')


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
    print('data processed, training data...')
    
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
    print('training successful, loaded user credentials...')
    

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
    Attribute6 = st.selectbox('Balance in Savings account / Bonds', ('Less than £1000', 'Between £1000 and £5000', 'Between £5000 and £10000', 'Greater than £10000'),index=None, placeholder='Select one...')
    Attribute7 = st.selectbox('Present Employment Since', ('Unemployed', '< 1 year', '< 4 years and >= 1 year', '< 7 years and >= 4 years', 'more than 7 years'),index=None, placeholder='Select one...')
    Attribute8 = st.number_input('Installment rate in percentage to disposable income', value=0)
    Attribute9 = st.selectbox('Personal status and Sex', ('Male - divorced/separated', 'Female - divorced/separated/married', 'Male - single', 'Male - married/widowed', 'Female - single'),index=None, placeholder='Select one...')
    Attribute10 = st.selectbox('Other debtors/guarantors', ('None', 'Co-applicant', 'Guarantor'),index=None, placeholder='Select one...')
    Attribute11 = st.number_input('Present residence since', 0, 4)
    Attribute12 = st.selectbox('Collateral', ('Real Estate', 'savings agreement/ life insurance', 'Car or other', 'No Collateral'),index=None, placeholder='Select one...')
    Attribute13 = st.number_input("Age (in years)")
    Attribute14 = st.selectbox('Other Installment plans', ('Bank', 'Stores', 'None'),index=None, placeholder='Select one...')
    Attribute15 = st.selectbox('Current housing', ('Rent', 'Own', 'other'),index=None, placeholder='Select one...')
    Attribute16 = st.slider('Number of existing credits in this bank', 0, 4)
    Attribute17 = st.selectbox('Skill level', ('unemployed/ unskilled  - non-resident', 'unskilled - resident', 'skilled employee / official', 'management/ self-employed/highly qualified employee/ officer'),index=None, placeholder='Select one...')
    Attribute18 = st.slider('Dependents', 0, 2)
    Attribute19 = st.selectbox('Telephone', ('Yes', 'No'),index=None, placeholder='Select one...')
    Attribute20 = st.selectbox('Foreign Worker', ('Yes', 'No'),index=None, placeholder='Select one...')

input_df = pd.DataFrame({"Attribute1" : [Attribute1], "Attribute2" : [Attribute2], "Attribute3" : [Attribute3], "Attribute4" : [Attribute4], "Attribute5" : [Attribute5], "Attribute6" : [Attribute6], "Attribute7" : [Attribute7], "Attribute8" : [Attribute8], "Attribute9" : [Attribute9], "Attribute10" : [Attribute10], "Attribute11" : [Attribute11], "Attribute12" : [Attribute12], "Attribute13" : [Attribute13], "Attribute14" : [Attribute14], "Attribute15": [Attribute15], "Attribute16" : [Attribute16], "Attribute17" : [Attribute17], "Attribute18" : [Attribute18], "Attribute19" : [Attribute19], "Attribute20" : [Attribute20]})
input_df 
