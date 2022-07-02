# import the streamlit library
import streamlit as st
 
# give a title to our app
st.title('Welcome to BMI Calculator')
 
# TAKE WEIGHTS INPUT
# radio button to choose weight format
status_weight = st.radio('Select your weight format: ',
                  ('lbs', 'kgs'))
 

# TAKE WEIGHT INPUT in kgs
weight = st.number_input("Enter your weight {status_weight}".format(status_weight= status_weight))
 
# TAKE HEIGHT INPUT
# radio button to choose height format
status = st.radio('Select your height format: ',
                  ('cms', 'meters', 'feet'))
 
# compare status value
if(status == 'cms'):
    try: 
    # take height input in centimeters
        height = st.number_input('Centimeters')
     
        if status_weight== 'lbs':
            bmi = 703*(weight/((height * .3937)**2))
        elif weight == 'kgs':
            bmi = weight / ((height/100)**2)

    except: 
            st.text("Enter some value of height")
         
elif(status == 'meters'):
    # take height input in meters
    height = st.number_input('Meters')
     
    try:
        if status_weight == 'lbs':
            bmi = 703 * (weight)/(((height*100)*.3937)**2)
        else:
            bmi = weight / ((height)**2)
    except:
            st.text("Enter some value of height")
         
else:
    # take height input in feet
    height = st.number_input('Feet')
     
    # 1 meter = 3.28
    try: #2 more undo's 
        if status_weight == 'lbs':
            bmi = 703  * (weight)/((height*12)**2)
        else:
            bmi = status
    except:
            st.text("Enter some value of height")
 
# check if the button is pressed or not
if(st.button('Calculate BMI')):
     
    # print the BMI INDEX
    st.text("Your BMI Index is {}.".format(bmi))
     
    # give the interpretation of BMI index
    if(bmi < 16):
        st.error("You are Extremely Underweight")
    elif(bmi >= 16 and bmi < 18.5):
        st.warning("You are Underweight")
    elif(bmi >= 18.5 and bmi < 25):
        st.success("Healthy")       
    elif(bmi >= 25 and bmi < 30):
        st.warning("Overweight")
    elif(bmi >= 30):
        st.error("Extremely Overweight")