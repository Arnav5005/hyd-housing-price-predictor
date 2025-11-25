# 🏠 Hyderabad House Price Predictor

A machine learning web application that predicts house prices in Hyderabad based on various property features.

## 📋 Features

- 🤖 **Random Forest ML Model** - Trained on 2500+ real estate listings
- 🎨 **Modern Dark UI** - Beautiful gradient interface with smooth animations
- 📊 **Data Profiling** - Interactive pandas profiling reports
- 🌍 **250+ Locations** - Covers all major areas in Hyderabad
- ⚡ **Real-time Predictions** - Instant price estimates

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, Random Forest Regressor
- **Data Processing**: pandas, joblib
- **Profiling**: ydata-profiling
- **Frontend**: HTML5, CSS3 (Dark theme with gradients)

## 📊 Model Performance

- **Algorithm**: Random Forest Regressor
- **Features**: 278 total features (including location encoding)
- **Data**: Hyderabad housing dataset with 40+ property attributes
- **Preprocessing**: Outlier removal (98th percentile), One-hot encoding

## 🚀 Installation

1. **Clone the repository**
   ```bash
   cd housing_model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (optional - model already included)
   ```bash
   python scripts/train.py
   ```

5. **Run the Flask app**
   ```bash
   python app/app.py
   ```

6. **Open browser**
   ```
   http://127.0.0.1:5000/
   ```

## 📁 Project Structure

```
housing_model/
│
├── app/
│   ├── app.py              # Flask application
│   ├── templates/
│   │   └── index.html      # Main form page
│   └── static/
│       └── style.css       # Dark theme styling
│
├── data/
│   └── Hyderabad.csv       # Training dataset
│
├── scripts/
│   └── train.py            # Model training script
│
├── housing_model.joblib    # Trained ML model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 How to Use

1. Enter property details:
   - **Area** (in square feet)
   - **Location** (select from dropdown)
   - **Number of Bedrooms**
   - **Resale** (Yes/No)
   - **Maintenance Staff** (Yes/No)
   - **Gymnasium** (Yes/No)

2. Click **"Predict Price"**

3. View the predicted price in rupees (₹)

4. Click **"View Data Profile Report"** for comprehensive dataset analysis

## 📊 Dataset Features

The model uses 40+ features including:
- Area, Location, Bedrooms
- Amenities: Gymnasium, Swimming Pool, Clubhouse, etc.
- Security: 24x7 Security, CCTV
- Utilities: Power Backup, Lift, Parking
- And many more...

## 🎨 UI Features

- **Dark Theme**: Professional navy blue background
- **Gradient Accents**: Cyan/blue gradients for visual appeal
- **Smooth Animations**: Hover effects and transitions
- **Responsive Design**: Works on desktop and mobile
- **Form Validation**: Required fields and input constraints

## 📈 Model Training

The model was trained using:
1. **Data Cleaning**: Removed outliers (98th percentile)
2. **Feature Engineering**: One-hot encoding for categorical variables
3. **Train-Test Split**: 80-20 split
4. **Algorithm**: Random Forest Regressor (default parameters)
5. **Evaluation**: RMSE and R² score metrics

## 🔮 Future Enhancements

- [ ] Add more property features (balcony, floor number, age)
- [ ] Deploy to cloud (AWS/Azure/Heroku)
- [ ] Add price trend charts
- [ ] Compare multiple properties
- [ ] User authentication and saved searches
- [ ] API endpoints for mobile apps

## 👨‍💻 Developer

Created with ❤️ for Hyderabad real estate analysis

## 📝 License

This project is open source and available for educational purposes.