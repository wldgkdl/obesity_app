
from flask import Flask, render_template, request, flash, redirect, url_for

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import pandas as pd
import pickle
from statistics import mode

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import load_model
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# 1. Decision Tree 
saved_clf = load('trained_models/clf_after_grid.joblib')

# 2. Random Forest
saved_rf = load('trained_models/rf_after_grid.joblib')

# 3. Logistic Regression
saved_LR = pickle.load(open('trained_models/saved_LR.sav', 'rb'))

# 4. Support Vector Machine 
saved_svc = load('trained_models/svc.joblib')

# 5. Naive Bayes
saved_NB = load('trained_models/nb.joblib')

# 6. Neural Network
saved_NN = load_model('trained_models/trained_NN.h5')

# 7. Adaboost
# saved_AB = load('trained_models/AB.joblib')

# 8. XGBoost
saved_XGB = load('trained_models/XGB.joblib')

# 9. KNN
saved_knn = load('trained_models/trained_KNN.joblib')

app = Flask(__name__)
app.secret_key = "wjdghks3#"



@app.route("/", methods = ["POST", "GET"])
def spec():
    return render_template("question_form.html")


@app.route("/form", methods = ["POST"])
def form():

    # Bring personal spec from the form
    height = request.form.get('height input')
    weight = request.form.get('weight input')
    gender = request.form.get('gender')
    h_unit = request.form.get('Height unit')
    w_unit = request.form.get('Weight unit')
    original_h = height
    original_w = weight
    original_g = gender

    # Modify spec based on unit
    if w_unit == 'lb':
        weight = int(0.453592 * float(weight))

    if h_unit == 'inches':
        height = int(2.54 * float(height))
    elif h_unit == 'feet':
        height = int(30.48 * float(height))

    if gender == 'Female':
        gender = 1
    else:
        gender = 0

    w_unit = str("Unit: ") + w_unit
    h_unit = str("Unit: ") + h_unit

    # Start inference
    # height = min(200, int(height))
    # weight = min(160, int(weight))   
    data = [gender, min(200, int(height)), min(160, int(weight))]

    #DT = saved_clf.predict([data])
    DT_proba = saved_clf.predict_proba([data])
    DT = np.argmax(DT_proba[0])
    #print(DT_proba)
    #print("The results of Decision Tree is",DT[0])

    RF_proba = saved_rf.predict_proba([data])
    RF = np.argmax(RF_proba[0])
    #print(RF_proba)
    #print("The results of Random Forest is",RF[0])

    # AB_proba = saved_AB.predict_proba([data])
    # AB = np.argmax(AB_proba[0])
    
    SVC_proba = saved_svc.predict_proba([data])
    SVC = np.argmax(SVC_proba[0])
    #print(SVC_proba)
    #print("The results of SVC is",SVC[0])

    NB_proba = saved_NB.predict_proba([data])
    NB = np.argmax(NB_proba[0])
    #print(NB_proba)
    #print("The results of Naive Bayes is",NB[0])

    my_scaler = joblib.load('trained_models/scaler.gz')
    transformed_data = data[:]

    #LR = saved_LR.predict(my_scaler.transform([transformed_data]))
    LR_proba = saved_LR.predict_proba(my_scaler.transform([transformed_data]))
    LR = np.argmax(LR_proba[0])
    #print(LR_proba)
    #print("The results of Logistic Regression is",LR[0])

    XGB_proba = saved_XGB.predict_proba(my_scaler.transform([transformed_data]))
    XGB = np.argmax(XGB_proba[0])

    KNN_proba = saved_knn.predict_proba(my_scaler.transform([transformed_data]))
    KNN = np.argmax(KNN_proba[0])

    NN_proba = saved_NN.predict(my_scaler.transform([transformed_data]))
    NN = np.argmax(NN_proba)
    #print(NN_proba)

    #print("The results of Neural Network is",np.argmax(NN))

    #Ensemble_hard_voted = (DT + RF + SVC + NB + LR + NN)/6
    try:
        Ensemble_hard_voted = mode([DT ,RF ,SVC ,NB ,LR ,NN, KNN, XGB])   # AB
        # len_hard_voted = [DT ,RF ,SVC ,NB ,LR ,NN].count(Ensemble_hard_voted)
        # if len_hard_voted < 3:
        #     Ensemble_hard_voted = round((DT + RF + SVC + NB + LR + NN)/6)
    except:
        sum_list = [a + b + c + d + e + f + g + h for a, b, c, d, e, f, g, h in zip(DT_proba[0], 
                                                                      RF_proba[0], 
                                                                      SVC_proba[0], 
                                                                      NB_proba[0],
                                                                      LR_proba[0],
                                                                      KNN_proba[0],
                                                                      XGB_proba[0],
                                                                      # AB_proba[0],
                                                                      NN_proba)]
        # print(max(sum_list))
        # print(type(sum_list))
        # print(max(sum_list[0]))
        Ensemble_hard_voted = sum_list[0].tolist().index(max(sum_list[0]))
        # print(Ensemble_hard_voted)

    #print(len_hard_voted)
    #print("The results of Ensemble model is",int(round(Ensemble_hard_voted[0])))

    def index2class(result):
        index_info = ['Weak', 'Slim', 'Normal', 'Overweight','Obesity','Extreme obesity']
        final_class = index_info[result]
        return final_class

    def return_proba(result):
        int_proba = int(max(result[0])*100)
        return int_proba

    def return_color(result):
        color_info = ['Olive', 'DarkGreen', 'LimeGreen', 'Orange', 'OrangeRed', 'Red']
        final_color = color_info[result]
        # ans = '{ "fill": ["color", "#eeeeee"], "innerRadius": 70, "radius": 85 }'
        # final_color = ans.replace('color', final_color)
        return final_color

    DT_class = index2class(DT)
    DT_proba = return_proba(DT_proba)
    DT_color = return_color(DT)

    RF_class = index2class(RF)
    RF_proba = return_proba(RF_proba)
    RF_color = return_color(RF)

    # AB_class = index2class(AB)
    # AB_proba = return_proba(AB_proba)
    # AB_color = return_color(AB)

    SVC_class = index2class(SVC)
    SVC_proba = return_proba(SVC_proba)
    SVC_color = return_color(SVC)

    NB_class = index2class(NB)
    NB_proba = return_proba(NB_proba)
    NB_color = return_color(NB)

    LR_class = index2class(LR)
    LR_proba = return_proba(LR_proba)
    LR_color = return_color(LR)

    XGB_class = index2class(XGB)
    XGB_proba = return_proba(XGB_proba)
    XGB_color = return_color(XGB)

    KNN_class = index2class(KNN)
    KNN_proba = return_proba(KNN_proba)
    KNN_color = return_color(KNN)

    NN_class = index2class(NN)
    NN_proba = return_proba(NN_proba)
    NN_color = return_color(NN)

    Final_class = index2class(Ensemble_hard_voted)
    Final_color = return_color(Ensemble_hard_voted)

    return render_template("index2.html", original_h = original_h,
                                          original_w = original_w,
                                          original_g = original_g,
                                          h_unit = h_unit,
                                          w_unit = w_unit,
                                          DT_class = DT_class,
                                          DT_proba = DT_proba,
                                          DT_color = DT_color,
                                          RF_class = RF_class,
                                          RF_proba = RF_proba,
                                          RF_color = RF_color,
                                          SVC_class = SVC_class,
                                          SVC_proba = SVC_proba,
                                          SVC_color = SVC_color,
                                          NB_class = NB_class,
                                          NB_proba = NB_proba,
                                          NB_color = NB_color,
                                          LR_class = LR_class,
                                          LR_proba = LR_proba,
                                          LR_color = LR_color,
                                          NN_class = NN_class,
                                          NN_proba = NN_proba,
                                          NN_color = NN_color,
                                          # AB_class = AB_class,
                                          # AB_proba = AB_proba,
                                          # AB_color = AB_color,
                                          XGB_class = XGB_class,
                                          XGB_proba = XGB_proba,
                                          XGB_color = XGB_color,
                                          KNN_class = KNN_class,
                                          KNN_proba = KNN_proba,
                                          KNN_color = KNN_color,
                                          Final_class = Final_class,
                                          Final_color = Final_color
                                          )


                           

if __name__ == '__main__':
    #os.environ['FLASK_ENV'] = 'development'
    app.run(debug = True)





   

