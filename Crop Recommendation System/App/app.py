import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.set_page_config(page_title="Efficient Crop Classification System", initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title

    html_temp = """
</style>
    <div>
    <h1 style="color:green;text-align:left;"> Efficient Crop Classification System </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col2,col1  = st.columns([2,2])
    
    
    with col2:
        st.subheader(" Find out  👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col2.write('''
		    ## Results 🔍 
		    ''')
            col2.success(f"{prediction.item().title()} are recommended your farm.")
      #code for html
    with col1: 
        '''
        '''



    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()