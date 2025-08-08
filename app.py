import streamlit as st
import pandas as pd 
import random
import pickle
from sklearn.preprocessing import StandardScaler

# title 

col =  ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('Califonia Housing Price Prediction')

st.image('https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg')
st.header('Model of housing prices to predict median house values in California', divider = True)

# st.subheader(''' User Must Enter Given values to predict price :
 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.oIuTdWdvWKSfYhnRtMyq3QHaEf?cb=thfc1&rs=1&pid=ImgDetMain&o=7&rm=3')

temp_df = pd.read_csv('california.csv')

random.seed(12)

all_values = []
for i in temp_df[col]:
    min_value , max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
                           random.randint(int(min_value), int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])


with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]


import time 

st.write(pd.DataFrame(dict(zip(col,all_values)) , index = [1]))
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place = st.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677/11677497.gif',width = 80)
if price>0:
    
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
    
    body =f'Predicted Median House Price: ${round(price ,2)} Thousand Dollars'
    placeholder.empty()
    
    st.success(body)
else:
    body = 'Invalid House Features Values'
    st.warning(body)
