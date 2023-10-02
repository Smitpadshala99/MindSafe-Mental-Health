from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, flash
import pandas as pd
import numpy as np
import joblib
import csv
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
authenticated=False
def hash_password(password):
    sha256 = hashlib.sha256()
    sha256.update(password.encode('utf-8'))
    return sha256.hexdigest()


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/importance')
def importance():
    return render_template('importance.html')

@app.route('/progress')
def progress():
    return render_template('progress.html')


@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/process_questions', methods=['POST'])
def process_questions():
    def reverse_mirror_number(num):
        if 1 <= num <= 7:
            return 8-num
        
    def reverse_mirror_number_cams(num):
        if 1 <= num <= 4:
            return 5-num
     # Load the saved model
    loaded_model = joblib.load('xgboost_model.joblib')
    print(request.form)
    # Retrieve values from the HTML form
    Age = int(request.form.get("Age"))
    Sex = int(request.form.get("Sex"))
    Disability = int(request.form.get("Disability"))
    hobbies_imp_1 = int(request.form.get('Hobbies_Imp_1'))
    hobbies_imp_2 = int(request.form.get('Hobbies_Imp_2'))
    hobbies_imp_3 = int(request.form.get('Hobbies_Imp_3'))
    hobbies_imp_4 = int(request.form.get('Hobbies_Imp_4'))
    hobbies_imp_6 = int(request.form.get('Hobbies_Imp_6'))
    cats_1 = int(request.form.get('CATS_1'))
    cats_4 = int(request.form.get('CATS_4'))
    sss_1 = int(request.form.get('SSS_1'))
    sss_12 = int(request.form.get('SSS_12'))
    cams_1 = int(request.form.get('CAMS_1'))
    cams_2 = int(request.form.get('CAMS_2'))
    cams_3 = int(request.form.get('CAMS_3'))
    cams_5 = int(request.form.get('CAMS_5'))
    cams_7 = int(request.form.get('CAMS_7'))
    ders_3 = int(request.form.get('DERS_3'))
    dass_4 = int(request.form.get('DASS_4'))
    dass_6 = int(request.form.get('DASS_6'))
    dass_15 = int(request.form.get('DASS_15'))
    dass_21 = int(request.form.get('DASS_21'))
    sss_6 = int(request.form.get('SSS_6'))


    
    # Repeat this for all your form inputs

    # Create an input array for prediction
    X_new = np.array([
        [hobbies_imp_1, hobbies_imp_2, hobbies_imp_3, hobbies_imp_4, hobbies_imp_6,
         cats_1, cats_4, sss_1, sss_12, cams_1, cams_2, cams_3, cams_5, cams_7, ders_3, Age, Sex, Disability, dass_4, dass_6, dass_15, dass_21, sss_6 ]
    ])
    avg_dass = (dass_4 + dass_6 + dass_15 + dass_21)*2
    avg_cats = cats_1+reverse_mirror_number(cats_4)/2
    avg_sss = (sss_1+sss_6+sss_12)/3
    avg_cams = (cams_1 + reverse_mirror_number_cams(cams_2) + reverse_mirror_number_cams(cams_7) + cams_3 + cams_5)/5
    your_score = avg_sss + avg_dass + avg_cams + avg_cats + ders_3
    # Use the loaded model for predictions
    predictions = loaded_model.predict(X_new)
    if(predictions[0]==1):
        answers="You are in good mental health"
    else:
        answers="Your mental health could be better"
    
    if((avg_cats) > 5):
        Cats = "Your sleep cycle is good"
    elif(avg_cats > 3):
        Cats = "Your sleep cycle could be improved"
    else:
        Cats = "You sleeping habits need immedieate attention"

    if(avg_sss >= 3):
        Sss = "Your are very socially active"
    else:
        Sss = "You need more social interactions"

    if(avg_cams > 2):
        Cams = "Your focus, attenstion, acceptance and awareness is good"
    else:
        Cams = "You can work on your awareness"

    if(ders_3 >=3 ):
        ders_3 = "You have difficulty in expressing or understanding your emotions"
    else:
        ders_3 = "You have good understanding of your emotions"
    
    if(avg_dass > 23 ):
        dass_2 = "You have symptoms of Depression Anxiety or Stress and should see a professional"
    elif(avg_dass > 11):
        dass_2 = "You have symptoms of Depression Anxiety or Stress"
    else:
        dass_2 = "You dont seem to have symptoms of Depression Anxiety or Stress"
    # if(cams_1+reverse_mirror_number_cams(cams_2))
    return render_template('output.html', answers=answers, CATS=Cats, SSS=Sss, CAMS=Cams, DERS=ders_3, DASS=dass_2, predictions=predictions, your_score=your_score)


@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/registration',methods=['POST'])
def register():
    firstname=request.form.get('firstName')
    lastname=request.form.get('lastName')
    dateofbirth=request.form.get('dob')
    gender=request.form.get('gender')
    email=request.form.get('email')
    password=request.form.get('password')
    conf_pass=request.form.get('confirmPassword')
    password=hash_password(password)
    if password==hash_password(conf_pass) :  
        file_exists = os.path.exists("regquiz.csv")
        if file_exists:
            with open("regquiz.csv",'r') as fr:
                reader=csv.DictReader(fr)
                for row in reader:
                    if(row['email']==email):
                        print("Email Id already taken")
                        return render_template('signup.html')
                    
        with open("regquiz.csv",'a') as fp:
            fieldnames=["firstname","lastname","dateofbirth","gender","email","password","Hobbies","CATS","Rested","SS","CAMS","Mindfullness_freq","DERS","DASS_2"]
            writer=csv.DictWriter(fp,fieldnames=fieldnames)
        
            if not file_exists:
                writer.writeheader()      
            writer.writerow({"firstname":firstname,"lastname":lastname,"dateofbirth":dateofbirth,"gender":gender,"email":email,"password":password})
            return render_template('login.html')
    else:
        flash('Passwords do not match. Please make sure your passwords match.')
        return render_template('signup.html')

@app.route('/authenticate',methods=['POST'])
def authenticate():
    global authenticated
    username=request.form.get('email')
    password=request.form.get('password')
    password=hash_password(password)
    checkbox_2=request.form.get('remember')
    print(checkbox_2)
    print("PASSWORD")
    with open(r'regquiz.csv','r') as fp:
        reader=csv.DictReader(fp)
        for row in reader:
           
            if(username==row['email'] ):
                if(password==row['password']):
                   print("authenticated")
                   authenticated = True
                   return redirect('/questions')
                else:
                    flash("wrong password")
                    return redirect('/login')
                
    return render_template('login.html')





if __name__ == '__main__':
    app.run(debug=True)
