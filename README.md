# Credit Card Fraud Detection

A machine learning web application for detecting fraudulent credit card transactions in real-time.

## Live Demo

http://fraud-detector-env.eba-wjpngskr.eu-west-2.elasticbeanstalk.com/

## Overview

This project implements a fraud detection system using a Random Forest classifier trained on credit transaction data. The web interface allows users to input transaction details and receive instant fraud predictions with probability scores.

## Features

- Real-time fraud prediction with confidence scores
- Transaction history tracking
- Analytics dashboard with fraud statistics
- PDF report generation
- Support for multiple ML models (Random Forest, Logistic Regression, XGBoost)

## Technical Stack

**Backend**
- Python 3.12
- Flask
- scikit-learn
- pandas, numpy
- SQLite

**Frontend**
- HTML/CSS/JavaScript
- Chart.js

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset and place in project root, then run:
```bash
python convert_dataset.py
python train_model.py
```

Start the server:
```bash
python app.py
```

Navigate to `http://localhost:5000`

## Project Structure
```
fraud-detection/
├── app.py                  # Flask application
├── train_model.py          # Model training
├── convert_dataset.py      # Data preprocessing
├── requirements.txt        # Dependencies
├── model.pkl              # Trained model
├── scaler.pkl             # Feature scaler
└── templates/
    └── index.html         # Frontend
```

## Model Performance

The Random Forest classifier achieves approximately 75% accuracy on the test set with the following metrics:

- Precision (Fraud): 0.60
- Recall (Fraud): 0.52
- ROC-AUC: 0.80

Training dataset contains 1,000 samples with 20 features and a 70/30 split between legitimate and fraudulent transactions.

## API Endpoints

### POST /predict_api
Returns fraud prediction for given transaction data.

Request body:
```json
{
  "Time": 0,
  "V1": 1.2,
  "V2": 0.5,
  "Amount": 150.0,
  "model_choice": "rf"
}
```

### GET /history
Returns list of recent predictions (requires API key).

### GET /admin_stats
Returns analytics data including fraud rates and transaction counts (requires API key).

## Usage

The dashboard provides three quick-fill options:
- Legitimate: Loads sample values for a normal transaction
- Fraudulent: Loads sample values for a fraudulent transaction
- Random: Generates random values for testing

After entering transaction details, click "Run Analysis" to get predictions. Results show the fraud probability and classification.

## Deployment

Configured for deployment on Render or similar platforms. Free tier deployments may experience cold starts on first load.

## Future Work

- Add SHAP values for model interpretability
- Implement user authentication
- Add email notifications for fraud alerts
- Export transaction history to CSV
- Train additional ensemble models

## License

MIT

## Author

Nawal
- GitHub: [@nawalali1](https://github.com/nawalali1)
- LinkedIn: www.linkedin.com/in/nawal-ali-871a09332
