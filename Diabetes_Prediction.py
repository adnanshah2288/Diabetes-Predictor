def Diabetes_Prediction():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import joblib

    # Collect input from user
    gender = float(input('Enter Your Gender (1 for Male, 0 for Female): '))
    age = float(input('Enter Your Age: '))
    hypertension = int(input('Do you have Hypertension? (1 for Yes, 0 for No): '))
    heart_disease = int(input('Do you have Heart Disease? (1 for Yes, 0 for No): '))
    smoking_history = int(input('Smoking History - (former-3, never-4, current-1, no info-0, not current-5, ever-2): '))
    bmi = float(input('Enter your BMI (e.g., 24.5): '))
    HbA1c_level = float(input('Enter your HbA1c Level (e.g., 5.6): '))
    blood_glucose_level = float(input('Enter your Blood Glucose Level (e.g., 120): '))

    # Create input DataFrame
    test_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }

    df = pd.DataFrame([test_data])

    # Load scaler and model
    scaler = joblib.load('scaler.pkl')
    model = tf.keras.models.load_model('Diabetes_Detection_Model.keras')

    # Apply scaling to numerical columns
    df.iloc[:, [1, 5, 6, 7]] = scaler.transform(df.iloc[:, [1, 5, 6, 7]])

    # Make prediction
    prediction_prob = model.predict(df)
    prediction = int(tf.round(prediction_prob).numpy()[0][0])

    if prediction == 1:
        print('✅ Yes, the patient **has** diabetes!')
    else:
        print('❎ No, the patient **does not** have diabetes.')

    return prediction

