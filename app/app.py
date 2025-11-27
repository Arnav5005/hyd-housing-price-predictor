from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from ydata_profiling import ProfileReport

app=Flask(__name__)

# load the trained model
model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'housing_model.joblib')
model=joblib.load(model_path)

# get all feature names from training data to ensure proper encoding
data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','Hyderabad.csv')
df=pd.read_csv(data_path)

upper=df["Price"].quantile(0.98)
df=df[df["Price"]<=upper]

# encoding
df_encoded=pd.get_dummies(df,columns=['Location'],drop_first=True)
feature_columns=[col for col in df_encoded.columns if col!='Price'] # all columns except price

@app.route('/')
def home():
    locations=sorted(df['Location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/profile')
def profile():
    # important columns
    cols_to_profile=["Price","Area","Location","No. of Bedrooms","Resale","MaintenanceStaff","Gymnasium"]

    # filter the columns
    df_small=df[cols_to_profile].copy()
    profile=ProfileReport(
        df_small, 
        title="Housing Data Profile",
        explorative=False, 
        minimal=True
    )
    return profile.to_html()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get form data
        area=float(request.form['Area'])
        location=request.form['Location']
        bedrooms=int(request.form['No. of Bedrooms'])
        resale=int(request.form['Resale'])
        maintenance_staff=int(request.form['MaintenanceStaff'])
        gymnasium=int(request.form['Gymnasium'])
        
        # Get all column names from original CSV (excluding Price)
        all_cols = [col for col in df.columns if col != 'Price']
        
        # Create input row with all columns, defaulting to 0
        input_dict = {}
        for col in all_cols:
            if col == 'Area':
                input_dict[col] = area
            elif col == 'Location':
                input_dict[col] = location
            elif col == 'No. of Bedrooms':
                input_dict[col] = bedrooms
            elif col == 'Resale':
                input_dict[col] = resale
            elif col == 'MaintenanceStaff':
                input_dict[col] = maintenance_staff
            elif col == 'Gymnasium':
                input_dict[col] = gymnasium
            else:
                input_dict[col] = 0
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Apply same encoding as training (get_dummies with drop_first=True)
        input_encoded = pd.get_dummies(input_df, columns=['Location'], drop_first=True)
        
        # Ensure all feature columns from training exist
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Select only the features that the model expects, in the exact order
        input_final = input_encoded[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_final)[0]
        
        # Return result as HTML
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Prediction Result</h1>
                </div>
                <h2>₹{prediction:,.0f}</h2>
                <div style="margin-top: 30px;">
                    <p><strong>Area:</strong> {area:,.0f} sqft</p>
                    <p><strong>Location:</strong> {location}</p>
                    <p><strong>Bedrooms:</strong> {bedrooms}</p>
                    <p><strong>Resale:</strong> {"Yes" if resale == 1 else "No"}</p>
                </div>
                <div style="margin-top: 30px; display: flex; gap: 15px; flex-direction: column;">
                    <a href="/" style="text-align: center;">← Back to form</a>
                    <a href="/profile" target="_blank" style="text-align: center; background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%); padding: 12px; border-radius: 8px; color: white;">📊 View Data Profile Report</a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>❌ Error</h1>
                </div>
                <p style="color: #ff6b6b;">An error occurred: {str(e)}</p>
                <pre style="color: #aaa; font-size: 12px; overflow: auto;">{error_detail}</pre>
                <a href="/">← Back to form</a>
            </div>
        </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(debug=True)

