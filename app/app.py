from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from ydata_profiling import ProfileReport

app = Flask(__name__)

# Load model and data
model = joblib.load('housing_model.joblib')
df = pd.read_csv('data/Hyderabad.csv')

# Clean data (remove outliers)
upper = df["Price"].quantile(0.98)
df = df[df["Price"] <= upper]

# Prepare feature columns
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)
feature_columns = [col for col in df_encoded.columns if col != 'Price']

@app.route('/')
def home():
    locations = sorted(df['Location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        location = request.form['location']
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = request.form['age']
        
        # Map age to numeric (simple encoding)
        age_map = {'0-5': 2.5, '5-10': 7.5, '10-20': 15, '20+': 25}
        age_years = age_map.get(age, 10)
        
        # Create input dictionary with all required columns
        input_dict = {}
        for col in df.columns:
            if col == 'Area':
                input_dict[col] = area
            elif col == 'Location':
                input_dict[col] = location
            elif col == 'No. of Bedrooms':
                input_dict[col] = bedrooms
            elif col == 'Age':
                input_dict[col] = age_years
            else:
                input_dict[col] = 0  # Default for other features
        
        # Convert to DataFrame and encode
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, columns=['Location'], drop_first=True)
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Select features and predict
        input_final = input_encoded[feature_columns]
        prediction = model.predict(input_final)[0]
        
        price_rupees = int(round(prediction, 0))
        return jsonify({'price': price_rupees})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/profile')
def profile():
    try:
        # Use only selected columns for the profile
        wanted_cols = ['Location', 'Area', 'No. of Bedrooms']
        cols = [c for c in wanted_cols if c in df.columns]
        if not cols:
            return "<pre>No requested columns found to profile.</pre>", 400
        df_subset = df[cols].copy()

        # Generate the profiling report on the subset
        report = ProfileReport(df_subset, title="Hyderabad Dataset Profile (Selected Columns)", explorative=True)
        return report.to_html()
    except Exception as e:
        return f"<pre>Failed to generate profile: {e}</pre>", 500

if __name__ == '__main__':
    app.run(debug=True)