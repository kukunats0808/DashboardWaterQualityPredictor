from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from main_torchHYBRID import prepare_data, HybridModel, predict, get_pollutant_level, calculate_wqi

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS for all routes

# Global variables for model and data
model = None
df = None
scaler = None
features = None
lookback = None  # Store the lookback period from the model


def load_model_and_data():
    """Load the trained model and preprocessed data"""
    global model, df, scaler, features, lookback

    print("⚙️ Loading data and model...")
    try:
        # Prepare data and load model
        X_train, y_train, X_test, y_test, scaler, features, df, test_dates = prepare_data('water_quality_data.csv')
        lookback = X_train.shape[1]  # Get the lookback period from training data

        model = HybridModel(input_size=len(features), seq_length=lookback)
        model.load_state_dict(torch.load('best_hybrid_model.pth', map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Model and data loaded successfully (Lookback: {lookback} months)")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


@app.route('/')
def serve_frontend():
    """Serve the main frontend interface"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files (CSS, JS, etc.)"""
    return send_from_directory('../frontend', path)


@app.route('/api/latest_params', methods=['GET'])
def get_latest_params():
    """API endpoint to get latest water quality parameters"""
    try:
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

        latest = df.iloc[-1]
        wqi = float(latest['WQI'])

        return jsonify({
            'status': 'success',
            'data': {
                'parameters': {
                    'ammonia': float(latest['Ammonia (mg/L)']),
                    'phosphate': float(latest['Phosphate (mg/L)']),
                    'nitrate': float(latest['Nitrate (mg/L)']),
                    'dissolvedOxygen': float(latest['Dissolved Oxygen (mg/L)']),
                    'pH': float(latest['pH Level']),
                    'temperature': float(latest['Surface Water Temp (°C)']),
                    'wqi': wqi,
                    'pollutantLevel': get_pollutant_level(wqi),
                },
                'date': latest.name.strftime('%Y-%m-%d'),
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/parameters', methods=['GET'])
def get_all_parameters():
    """API endpoint to get all parameters for dashboard"""
    try:
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

        # Get time range from query params
        time_range = request.args.get('range', '30days')

        if time_range == '30days':
            cutoff = df.index[-1] - pd.DateOffset(days=30)
            filtered_df = df[df.index >= cutoff].resample('M').mean()
        elif time_range == '6months':
            cutoff = df.index[-1] - pd.DateOffset(months=6)
            filtered_df = df[df.index >= cutoff]
        else:  # 1year
            cutoff = df.index[-1] - pd.DateOffset(years=1)
            filtered_df = df[df.index >= cutoff]



        # Prepare data for each parameter
        parameters = {
            'ammonia': {
                'values': filtered_df['Ammonia (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Ammonia (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Ammonia (mg/L)'].mean())
            },
            'phosphate': {
                'values': filtered_df['Phosphate (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Phosphate (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Phosphate (mg/L)'].mean())
            },
            'dissolvedoxygen': {
                'values': filtered_df['Dissolved Oxygen (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Dissolved Oxygen (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Dissolved Oxygen (mg/L)'].mean())
            },
            'nitrate': {
                'values': filtered_df['Nitrate (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Nitrate (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Nitrate (mg/L)'].mean())
            },
            'ph': {
                'values': filtered_df['pH Level'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'pH',
                'current': float(filtered_df['pH Level'].iloc[-1]),
                'avg': float(filtered_df['pH Level'].mean())
            },
            'temperature': {
                'values': filtered_df['Surface Water Temp (°C)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': '°C',
                'current': float(filtered_df['Surface Water Temp (°C)'].iloc[-1]),
                'avg': float(filtered_df['Surface Water Temp (°C)'].mean())
            }
        }

        return jsonify({
            'status': 'success',
            'data': parameters,
            'timeRange': time_range,
            'startDate': cutoff.strftime('%Y-%m-%d'),
            'endDate': df.index[-1].strftime('%Y-%m-%d'),
            'minDate': df.index[0].strftime('%Y-%m-%d'),
            'maxDate': (df.index[-1] + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    """API endpoint to make WQI predictions"""
    try:
        if model is None or df is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503

        data = request.json
        date_str = data.get('date')

        if not date_str:
            return jsonify({
                'status': 'error',
                'message': 'Date is required'
            }), 400

        # Validate and parse date
        try:
            date_obj = pd.to_datetime(date_str)
            min_date = pd.to_datetime(df.index[0])
            max_date = pd.to_datetime(df.index[-1] + pd.DateOffset(years=30))

            if date_obj < min_date:
                return jsonify({
                    'status': 'error',
                    'message': f'Date must be after {min_date.strftime("%Y-%m-%d")}'
                }), 400

            if date_obj > max_date:
                return jsonify({
                    'status': 'error',
                    'message': f'Date must be before {max_date.strftime("%Y-%m-%d")}'
                }), 400
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid date format: {str(e)}'
            }), 400

        # Make prediction
        prediction = predict(model, df, scaler, features, date_obj)

        if prediction is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed - insufficient historical data'
            }), 400

        wqi = float(prediction)
        return jsonify({
            'status': 'success',
            'data': {
                'wqi': wqi,
                'pollutantLevel': get_pollutant_level(wqi),
                'date': date_obj.strftime('%Y-%m-%d'),
                'confidence': 0.85,  # Placeholder for confidence score
                'modelInfo': {
                    'type': 'Hybrid CNN-LSTM',
                    'lookbackPeriod': f'{lookback} months',
                    'lastTrained': os.path.getmtime('best_hybrid_model.pth')
                }
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'lookback_period': lookback,
        'server_time': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Load model and data when starting the server
    load_model_and_data()

    # Run the Flask app
    print("\n🌍 Server running at http://localhost:5000")
    print("🔌 API endpoints:")
    print("   - GET  /api/latest_params")
    print("   - GET  /api/parameters?range=30days|6months|1year")
    print("   - POST /api/predict")
    print("   - GET  /api/health\n")

    app.run(host='0.0.0.0', port=5000, debug=True)