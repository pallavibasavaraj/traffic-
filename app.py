from flask import *
from flask import send_from_directory
from werkzeug.utils import secure_filename
import mysql.connector
db = mysql.connector.connect(host='localhost',user='root',port=3306,database='parkingreservation')
cur = db.cursor()
import pandas as pd
from flask_mail import *
from detecto.core import Model
from shutil import copyfile
import show
import cv2
import torch
import tempfile
from ultralytics import YOLO
from norfair import Detection, Tracker
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key="fcb384r23823872380237r89irw78eduwsf78we4y"


# home page
@app.route("/")
def index():
    return render_template("index.html")

# admin login page
@app.route("/admin",methods=["POST","GET"])
def admin():
    if request.method=='POST':
        form = request.form
        adminname =  form['adminname']
        password = form['adminpassword']
        if adminname =='admin' and password == 'admin':
            return render_template('adminhome.html',admin=adminname)
        else:
            return render_template('admin.html',msg="invalid Credentials")
    return render_template("admin.html")


# adding parking Details
@app.route("/addparking",methods=["POST","GET"])
def addparking():
    if request.method=="POST":
        form = request.form
        form1= request.files
        parkingslot = form['parkingslot']
        Cost = form['Cost']
        address = request.form['address']
        nameofimage = form1['nameofimage']
        # imagename = nameofimage.filename
        nameofimage.save(f'static/projectimages/{secure_filename(nameofimage.filename)}')
        sql="insert into parkingslots(parkingslot,Cost,Address,Imagename)values('%s','%s','%s','%s')"%(parkingslot,Cost,address,nameofimage.filename)
        cur.execute(sql)
        db.commit()
    return render_template("addparking.html")

# customer login
@app.route("/customer",methods=["POST","GET"])
def customer():
    if request.method=="POST":
        form = request.form
        customeremail = form['customeremail']
        customerpassword = form['customerpassword']
        sql="select * from customerreg where customeremail='%s' and customerpassword='%s'"%(customeremail,customerpassword)
        cur.execute(sql)
        dc = cur.fetchall()
        if dc !=[]:
            session['useremail']=customeremail
            return redirect('customerhome')
        else:
            return render_template("customer.html",msg="invalid Credentials")

    return render_template("customer.html")

@app.route("/customerhome",methods=["POST","GET"])
def customerhome():
    return render_template("customerhome.html")



# customerreg
@app.route("/customerreg",methods=["POST","GET"])
def customerreg():
    if request.method=="POST":
        form = request.form
        customername = form['customername']
        customeremail = form['customeremail']
        customerpassword = form['customerpassword']
        confirmpassword = form['confirmpassword']
        customercontact = form['customercontact']
        customeraddress = form['customeraddress']
        if customerpassword == confirmpassword:
            sql="select * from customerreg where customeremail='%s' and customerpassword='%s'"%(customeremail,customerpassword)
            cur.execute(sql)
            d = cur.fetchall()
            if d ==[]:
                sql="insert into customerreg(customername,customeremail,customerpassword,customercontact,customeraddress)values(%s,%s,%s,%s,%s)"
                val=(customername,customeremail,customerpassword,customercontact,customeraddress)
                cur.execute(sql,val)
                db.commit()
                return render_template("customer.html")
            else:
                return render_template("customerreg.html",msg="Password not matched")
        else:
            return render_template("customerreg.html",msg="Password not matched")

    return render_template("customerreg.html")


# Parking details for user

@app.route('/view_parking')
def view_parking():
    sql="select * from parkingslots"
    data = pd.read_sql_query(sql,db)
    return render_template("viewparking.html",cols=data.columns.values,rows=data.values.tolist())


@app.route("/reserveslot/<c>")
def reserveslot(c=0):
    session['c'] = c
    sql = "select * from parkingslots where parkingslot='%s'"%(c)
    cur.execute(sql)
    dc = cur.fetchall()[0]
    return render_template("reserveslot.html",dc=dc)

@app.route("/bookslot",methods=["POST","GET"])
def bookslot():
    c =session['c']
    # print(c,'uvsfbsfinsadfinsafiuwbfiuasdfbwsdiufbsiufbiusdb')
    # print("123456789")
    sql = "select * from parkingslots where parkingslot='%s'"%(c)
    cur.execute(sql)
    dc = cur.fetchall()[0]
    if request.method=="POST":
        # print('0000000000000000000000000000000000000000000000')
        c = session['c']
        slotid = request.form['slotid']
        hourcost = int(request.form['hourcost'])
        nameoncard = request.form['nameoncard']
        cvv = request.form['cvv']
        expiredate = request.form['expiredate']
        totalhours = int(request.form['totalhours'])
        totalamount = request.form['totalamount']

        total_amount = int(hourcost)*int(totalhours)
        status = 'locked'
        sql="select * from bookslot where useremail='%s'"%(session['useremail'])
        print()
        cur.execute(sql)
        data = cur.fetchall()
        if data ==[]:
            # print('////////////////////////////////////////////////////')
            sql="insert into bookslot(slotid,hourcost,nameoncard,cvv,expiredate,totalhours,totalamount,status,useremail)values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (slotid,hourcost,nameoncard,cvv,expiredate,totalhours,totalamount,status,session['useremail'])
            cur.execute(sql,val)
            db.commit()
            sql="update parkingslots set status='locked' where parkingslot='%s'"%(c)
            cur.execute(sql)
            db.commit()
            return redirect('view_parking')
        else:
            return render_template("viewparking.html",dc=dc,msg="that slot already booked")
    return render_template("reserveslot.html",dc=dc)


@app.route("/userbookedslots")
def userbookedslots():
    sql="select id,slotid,hourcost,nameoncard,totalhours,totalamount from bookslot where status='locked'"
    data=pd.read_sql_query(sql,db)
    return render_template("userbookedslots.html",cols=data.columns.values,rows=data.values.tolist())


@app.route("/acceptrequest/<x>")
def acceptrequest(x=0):
    sender_address = 'sannidhinc.2003@gmail.com'
    sender_pass = 'ssyghhuvrmoplcer'
    content = "Your Request Is Accepted by the Admin, You Can Login Now"
    receiver_address = session['useremail']
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = "Online Parking Reservation System"
    message.attach(MIMEText(content, 'plain'))
    ss = smtplib.SMTP('smtp.gmail.com', 587)
    ss.starttls()
    ss.login(sender_address, sender_pass)
    text = message.as_string()
    ss.sendmail(sender_address, receiver_address, text)
    ss.quit()
    sql="update bookslot set status='accepted' where id='%s'"%(x)
    cur.execute(sql)
    db.commit()
    return redirect(url_for('userbookedslots'))

@app.route("/viewresponse")
def viewresponse():
    sql="select slotid,hourcost,totalhours,totalamount,status,useremail from bookslot where status='accepted' and useremail='%s'"%(session['useremail'])
    data = pd.read_sql_query(sql,db)
    return render_template("viewresponse.html",cols=data.columns.values,rows=data.values.tolist())


@app.route("/rejectrequest/<x>")
def rejectrequest(x=0):


    sender_address = 'sannidhinc.2003@gmail.com'
    sender_pass = 'ssyghhuvrmoplcer'
    content = "Your Request Is Rejected by the Admin because of no parking slots reservation"
    receiver_address = session['useremail']
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = "Online Parking Reservation System"
    message.attach(MIMEText(content, 'plain'))
    ss = smtplib.SMTP('smtp.gmail.com', 587)
    ss.starttls()
    ss.login(sender_address, sender_pass)
    text = message.as_string()
    ss.sendmail(sender_address, receiver_address, text)
    ss.quit()
    sql="update bookslot set status='rejected' where id='%s'"%(x)
    cur.execute(sql)
    db.commit()
    sql="update parkingslots set status='unlocked' where id='%s'"%(x)
    cur.execute(sql)
    db.commit()
    return redirect(url_for('userbookedslots'))

@app.route("/prediction",methods=["POST","GET"])
def prediction():
    if request.method == 'POST':
        video = request.files["upload"]
        file = open("video.mp4", 'wb')
        file.write(video.read())
        file.close()
        print("Working")
        model = Model.load('Objmodel1.h5', ['occupied', 'unoccupied'])
        show.detect(model, 'video.mp4', 'output.avi')
        copyfile('output.avi', 'static/video/output.avi')
        return redirect('/static/video/output.avi')
    return render_template("prediction.html")



# from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# import pandas as pd
# import joblib
# from datetime import datetime
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
# import random
# from flask import jsonify
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# sarima_fit = joblib.load('sarima_model_3h.pkl')
# # Define a mapping for days of the week
# day_of_week_mapping = {
#     'Monday': 0,
#     'Tuesday': 1,
#     'Wednesday': 2,
#     'Thursday': 3,
#     'Friday': 4,
#     'Saturday': 5,
#     'Sunday': 6
# }
# bcrypt = Bcrypt(app)



# # Load and preprocess data
# data = pd.read_csv('TrafficTwoMonth.csv')  # Replace with your actual data file path
# # Convert Date and Time to datetime
# data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.time
# # Encode categorical variables
# label_encoder = LabelEncoder()
# data['Day of the week'] = label_encoder.fit_transform(data['Day of the week'])
# data['Traffic Situation'] = label_encoder.fit_transform(data['Traffic Situation'])
# # Drop any remaining non-numeric columns if any
# data = data.select_dtypes(include=[np.number])
# data=data.drop(["Date","Total"],axis=1)
# print(data.columns)
# # Define features and target variable
# X = data.drop(['Traffic Situation'], axis=1)
# y = data['Traffic Situation']
# feature_names = X.columns.tolist()

# # Apply SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Normalize the resampled features
# scaler = StandardScaler()  # or MinMaxScaler()
# X_resampled_normalized = scaler.fit_transform(X_resampled)

# # Split the balanced and normalized dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_resampled_normalized, y_resampled, test_size=0.2, random_state=42)

# # Initialize and train the Decision Tree Classifier
# decision_tree = DecisionTreeClassifier()
# # Ensure that the model is trained with DataFrames having feature names
# X_train_df = pd.DataFrame(X_train, columns=feature_names)
# decision_tree.fit(X_train_df, y_train)




# @app.route('/predict1', methods=['GET', 'POST'])
# def predict1():
#     if request.method == 'POST':
#         forecast_hours = int(request.form.get('forecast_hours', 24))

#         print(f"Starting forecast for {forecast_hours} hours.")
#         # Load the data
#         file_path = 'traffic.csv'
#         data = pd.read_csv(file_path)

#         # Parse the DateTime column to datetime objects and set it as the index
#         data['DateTime'] = pd.to_datetime(data['DateTime'])
#         data.set_index('DateTime', inplace=True)

#         # Downsample the data to hourly frequency
#         data_3h = data.resample('H').sum()

#         # Select the 'Vehicles' column for forecasting
#         if 'Vehicles' not in data_3h.columns:
#             return "Error: 'Vehicles' column not found in the dataset."

#         series = data_3h['Vehicles']

#         # # Define and fit the SARIMA model on the selected column
#         # sarima_model = SARIMAX(series, 
#         #                        order=(1, 1, 1), 
#         #                        seasonal_order=(1, 1, 1, 24)) 
#         # sarima_fit = sarima_model.fit(disp=False)
#         sarima_model = SARIMAX(series, 
#                                order=(1, 1, 1), 
#                                seasonal_order=(1, 0, 0, 24))  # Simplified seasonal order
#         sarima_fit = sarima_model.fit(disp=False, maxiter=50) 

#         # Forecast
#         steps = forecast_hours
#         arima_forecast = sarima_fit.forecast(steps=steps)

#         decimal_places = 0
#         arima_forecast = [str(round(i, decimal_places)) for i in arima_forecast]

#         time_list = {
#             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
#             "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16,
#             "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24
#         }

#         time = time_list.get(str(forecast_hours), 0)
#         print(time)
#         print(f"SARIMA Forecast for next {forecast_hours} hours:\n{arima_forecast}\n")

#         return render_template('predict1.html', forecast=f'Forecasting Completed for {forecast_hours} hours', arima_forecast=arima_forecast, time=time)
#     return render_template('predict1.html')

# @app.route('/predict2', methods=['GET', 'POST'])
# def predict2():
#     if request.method == 'POST':
#         car_count = int(request.form.get('car_count', 0))
#         bike_count = int(request.form.get('bike_count', 0))
#         bus_count = int(request.form.get('bus_count', 0))
#         truck_count = int(request.form.get('truck_count', 0))
#         day_of_week = request.form.get('day_of_week', '')

#         day_of_week_encoded = day_of_week_mapping[day_of_week]

#         # Prepare the input data with the correct order of features
#         input_data = {
#             'Day of the week': day_of_week_encoded,  # Use mapped value
#             'CarCount': car_count,
#             'BikeCount': bike_count,
#             'BusCount': bus_count,
#             'TruckCount': truck_count
#         }

#         # Create DataFrame with feature names matching the training data
#         input_df = pd.DataFrame([input_data])  # Input data needs to be passed as a list of dictionaries
        
#         # Ensure the DataFrame columns match the training features
#         input_df = input_df[feature_names]  # Select only the columns in feature_names and in the correct order
        
#         # Apply the same scaling
#         input_df_scaled = scaler.transform(input_df)

#         # Convert back to DataFrame to keep feature names
#         input_df_scaled = pd.DataFrame(input_df_scaled, columns=feature_names)

#         # Predict using the Decision Tree model
#         prediction = decision_tree.predict(input_df_scaled)
#         print("prediction----", prediction)
#         class_name = label_encoder.inverse_transform(prediction)[0]  # Use inverse transform to get original class name
#         probabilities = decision_tree.predict_proba(input_df_scaled)[0]
#         print("probabilities: ", probabilities)
#         return render_template('predict2.html', class_name = class_name, probabilities = probabilities)
#     return render_template('predict2.html')




# @app.route('/get_traffic/<float:lat>/<float:lon>')
# def get_traffic(lat, lon):
#     # Randomly select a traffic condition
#     traffic_conditions = ['Heavy', 'Low', 'High', 'Normal']
#     traffic_condition = random.choice(traffic_conditions)
    
#     # Return the selected traffic condition as JSON
#     return jsonify({'traffic_condition': traffic_condition})


# @app.route('/map_view')
# def map_view():
#     return render_template('map.html')
# # YOLOv5 Model

# @app.route("/video", methods=["GET", "POST"])
# def video():
    
#     import os
#     os.system("streamlit run r.py")
#     return render_template("video.html")




# if __name__ =="__main__":
#     app.run(debug=True, port=5000)
