import streamlit as st
import pandas as pd 
import numpy as np 
import sklearn
import joblib

# Reading Dataset
df_class = pd.read_csv("data/classData.csv")

# Representing faults in one Fault_Type Column
df_class['Fault_Type'] = df_class['G'].astype('str') + df_class['C'].astype('str') + df_class['B'].astype('str') + df_class['A'].astype('str')

# Replacing Values of Fault Type For Easy Visualization

df_class['Fault_Type'][df_class['Fault_Type'] == '0000' ] = 'NO Fault'
df_class['Fault_Type'][df_class['Fault_Type'] == '1001' ] = 'Line A to Ground Fault'
df_class['Fault_Type'][df_class['Fault_Type'] == '0110' ] = 'Line B to Line C Fault'
df_class['Fault_Type'][df_class['Fault_Type'] == '1011' ] = 'Line A Line B to Ground Fault'
df_class['Fault_Type'][df_class['Fault_Type'] == '0111' ] = 'Line A Line B Line C'
df_class['Fault_Type'][df_class['Fault_Type'] == '1111' ] = 'Line A Line B Line C to Ground Fault'

# Categorical to Numerical Conversion for Model Input
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_class['Fault_Type'] = encoder.fit_transform(df_class['Fault_Type'])

# Dependent and Independent Variable Sepration
X = df_class.drop(['Fault_Type','G','C','B','A'],axis=1)
y = df_class['Fault_Type']

# Train Test Split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=21)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
joblib.dump(random_forest, 'model/random_forest.pkl')

#Interface
st.set_page_config(layout="wide")
#st.write("# Electrical Line Fault Prediction")
st.title('⚡ Electrical Line Fault Prediction ⚡')
st.markdown("""
Normally, a power system operates under balanced conditions. When the system becomes unbalanced due to the failures of insulation at any point or due to the contact of live wires, a short–circuit or fault, is said to occur in the line. 
Faults may occur in the power system due to the number of reasons like natural disturbances (lightning, high-speed winds, earthquakes), insulation breakdown, falling of a tree, bird shorting, etc.         

This application enables field workers to inspect faults in electrical lines. They are required to input current and voltage values in order to obtain a projected fault type.            
""")

st.sidebar.title("Contact")

st.sidebar.info(
    """
    Sachin Shrivastav
    
    [Email: sachinxshrivastav@gmail.com]
    """
)

I_A = st.number_input('Value of I(A)')
I_B = st.number_input('Value of I(B)')
I_C = st.number_input('Value of I(C)')
V_A = st.number_input('Value of V(A)')
V_B = st.number_input('Value of V(B)')
V_C = st.number_input('Value of V(C)')


#Predict button
if st.button('Predict'):
    model = joblib.load('model/random_forest.pkl')
    X = np.array([I_A, I_B, I_C, V_A, V_B, V_C])
    if any(X == 0):
        st.markdown('### Please input values')
    else:
        st.markdown(f'### Predicted Fault : {encoder.inverse_transform(model.predict([[I_A, I_B, I_C, V_A, V_B, V_C]]))[0]}')

