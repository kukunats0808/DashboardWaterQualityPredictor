import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import math

warnings.filterwarnings('ignore')


# 1. Simplified WQI Calculation (unchanged)
def calculate_wqi(row):
    """Simpler more robust WQI calculation"""
    params = {
        'pH': max(0, 1 - abs(row.get('pH Level', 7) - 7) / 3),
        'DO': min(row.get('Dissolved Oxygen (mg/L)', 8) / 15, 1),
        'Ammonia': max(0, 1 - row.get('Ammonia (mg/L)', 0.5) / 2),
        'Nitrate': max(0, 1 - row.get('Nitrate (mg/L)', 0.5) / 10),
        'Phosphate': max(0, 1 - row.get('Phosphate (mg/L)', 0.1) / 0.5),
        'Temp': max(0, 1 - abs(row.get('Surface Water Temp (°C)', 25) - 25) / 15)
    }
    return sum(params.values()) / len(params) * 100

def get_pollutant_level(wqi):
    if wqi >= 91:
        return "Excellent"
    elif wqi >= 71:
        return "Good"
    elif wqi >= 51:
        return "Average"
    elif wqi >= 26:
        return "Fair"
    else:
        return "Poor"


# 2. Hybrid CNN-LSTM Model
class HybridModel(nn.Module):
    def __init__(self, input_size, seq_length=12):
        super().__init__()
        self.seq_length = seq_length

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # LSTM Temporal Processor
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Final Prediction Layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Input shape: [batch, seq_len, features]
        batch_size = x.size(0)

        # CNN expects [batch, features, seq_len]
        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in)  # Output: [batch, 256, seq_len//2]

        # Prepare for LSTM: [batch, new_seq_len, features]
        lstm_in = cnn_out.permute(0, 2, 1)

        # Process temporal features
        lstm_out, _ = self.lstm(lstm_in)  # Output: [batch, new_seq_len, 256]

        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, 256]

        # Final prediction
        return self.fc(context)


# 3. Data Preparation (enhanced)
def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Create proper datetime index
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-15')
    df = df.sort_values('Date').set_index('Date')

    # Enhanced missing value handling
    df = df.replace(-999.0, np.nan)
    df = df.clip(lower=0)

    # Seasonal decomposition for better imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            seasonal = df[col].rolling(12, min_periods=1).mean()
            df[col] = df[col].fillna(seasonal)

    df = df.ffill().bfill()

    # Calculate WQI
    df['WQI'] = df.apply(calculate_wqi, axis=1)

    # Feature engineering
    df['Rainfall_log'] = np.log1p(df['Rainfall'])
    df['WindDir_sin'] = np.sin(np.radians(df['WindDirection']))
    df['WindDir_cos'] = np.cos(np.radians(df['WindDirection']))
    df['Temp_Δ'] = df['Surface Water Temp (°C)'].diff().fillna(0)
    df['WindSpeed_Δ'] = df['WindSpeed'].diff().fillna(0)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['Season'] = (df.index.month % 12 + 3) // 3  # 1=winter, 2=spring, etc.

    # Lag features
    for lag in [1, 3, 6, 12]:
        df[f'WQI_lag_{lag}'] = df['WQI'].shift(lag)
        df[f'Temp_lag_{lag}'] = df['Surface Water Temp (°C)'].shift(lag)

    # Rolling features
    df['WQI_rolling_3'] = df['WQI'].rolling(3).mean()
    df['WQI_rolling_6'] = df['WQI'].rolling(6).mean()
    df['Temp_rolling_3'] = df['Surface Water Temp (°C)'].rolling(3).mean()

    # Select features
    features = [
        'Surface Water Temp (°C)', 'pH Level',
        'Dissolved Oxygen (mg/L)', 'Ammonia (mg/L)',
        'Nitrate (mg/L)', 'Phosphate (mg/L)',
        'Rainfall_log', 'Tmax', 'Tmin', 'RH',
        'WindSpeed', 'WindDir_sin', 'WindDir_cos',
        'Temp_Δ', 'WindSpeed_Δ', 'Month_sin', 'Month_cos',
        'WQI_lag_1', 'WQI_lag_3', 'WQI_lag_12',
        'Temp_lag_1', 'Temp_lag_12',
        'WQI_rolling_3', 'WQI_rolling_6'
    ]

    # Drop rows with missing values from lags
    df = df.dropna(subset=features + ['WQI'])

    # Normalize data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    # Create sequences
    lookback = 18  # Increased lookback for better temporal context
    X, y, dates = [], [], []

    for i in range(lookback, len(df)):
        X.append(scaled[i - lookback:i])
        y.append(df['WQI'].iloc[i])
        dates.append(df.index[i])

    # Split chronologically
    split = int(0.8 * len(X))
    X_train, y_train = torch.FloatTensor(X[:split]), torch.FloatTensor(y[:split])
    X_test, y_test = torch.FloatTensor(X[split:]), torch.FloatTensor(y[split:])

    return X_train, y_train, X_test, y_test, scaler, features, df, dates[split:]


# 4. Enhanced Training Function
def train_model(model, X_train, y_train, X_test, y_test, epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Combined loss function
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

    # Corrected learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    best_loss = float('inf')
    best_r2 = -float('inf')
    train_losses, val_losses = [], []
    early_stop_counter = 0
    early_stop_patience = 200

    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        batch_loss = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss.append(loss.item())

        train_loss = np.mean(batch_loss)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test.to(device))
            val_loss = criterion(val_preds, y_test.to(device).unsqueeze(1))
            val_losses.append(val_loss.item())

            # Calculate validation R²
            val_r2 = r2_score(
                y_test.cpu().numpy(),
                val_preds.cpu().numpy().flatten()
            )

        scheduler.step(val_loss)  # Update based on validation loss

        # Manual LR reduction notification
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 20 == 0 or current_lr < 1e-6:
            print(
                f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val R²: {val_r2:.4f}, LR: {current_lr:.2e}')

        # Save best model based on validation loss and R²
        if val_loss < best_loss or val_r2 > best_r2:
            if val_loss < best_loss:
                best_loss = val_loss
            if val_r2 > best_r2:
                best_r2 = val_r2
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load best model
    model.load_state_dict(torch.load('best_hybrid_model.pth'))

    # Final validation metrics
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test.to(device))
        final_r2 = r2_score(
            y_test.cpu().numpy(),
            val_preds.cpu().numpy().flatten()
        )
    print(f'Final Validation R²: {final_r2:.4f}')

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, val_preds.cpu().numpy(), alpha=0.6)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.title(f'Actual vs Predicted (R²={final_r2:.4f})')
    plt.xlabel('Actual WQI')
    plt.ylabel('Predicted WQI')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('hybrid_model_performance.png')
    plt.show()

    return model.to('cpu'), train_losses, val_losses


# 5. Prediction Function (enhanced)
def predict(model, df, scaler, features, date):
    try:
        date = pd.to_datetime(date).replace(day=15)
        lookback = model.seq_length

        if date < df.index[0] + pd.DateOffset(months=lookback):
            raise ValueError(f"Need data from at least {lookback} months before {date.date()}")

        # Estimate monthly trend and seasonal components
        monthly_trend = df[features].diff(12).mean() / 12
        seasonal_means = df.groupby(df.index.month)[features].mean()
        global_mean = df[features].mean()
        seasonal_adjustment = seasonal_means - global_mean

        # Handle prediction into the future
        if date > df.index[-1]:
            print(f"Warning: Projecting into future beyond {df.index[-1].date()}")

            months_ahead = (date.year - df.index[-1].year) * 12 + (date.month - df.index[-1].month)
            base_point = df.iloc[-1][features].copy()

            # Generate synthetic future data
            seq = []
            for i in range(lookback):
                months_offset = months_ahead - (lookback - i)
                future_month = (df.index[-1] + pd.DateOffset(months=months_offset)).month

                point = base_point + monthly_trend * months_offset
                point += seasonal_adjustment.loc[future_month]

                # Add random noise
                noise = df[features].std() * 0.1 * np.random.randn(len(features))
                point += noise

                seq.append(point)

        else:
            # Normal prediction using existing historical data
            last_idx = df.index.get_loc(date)
            seq = [df.iloc[last_idx - lookback + i][features].copy() for i in range(lookback)]

        # Scale and predict
        seq_scaled = scaler.transform(np.array(seq))
        print(f"Predicting for: {date.strftime('%Y-%m')}, Seq mean: {np.mean(seq_scaled):.3f}, std: {np.std(seq_scaled):.3f}")
        with torch.no_grad():
            pred = model(torch.FloatTensor(seq_scaled).unsqueeze(0)).item()

        return max(0, min(100, pred))

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None



# 6. Main Execution
def main():
    print("Loading and preparing data...")
    X_train, y_train, X_test, y_test, scaler, features, df, test_dates = prepare_data('water_quality_data.csv')

    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature count: {len(features)}, Sequence length: {X_train.shape[1]}")

    print("\nInitializing Hybrid CNN-LSTM model...")
    model = HybridModel(input_size=len(features), seq_length=X_train.shape[1])
    print(model)

    print("\nTraining model...")
    model, train_losses, val_losses = train_model(model, X_train, y_train, X_test, y_test, epochs=500)

    # Final evaluation
    with torch.no_grad():
        test_preds = model(X_test).numpy().flatten()

    print("\nFinal Model Performance:")
    print(f"MAE: {mean_absolute_error(y_test, test_preds):.2f}")
    print(f"RMSE: {math.sqrt(mean_squared_error(y_test, test_preds)):.2f}")
    print(f"R²: {r2_score(y_test, test_preds):.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test.numpy(), 'b-', label='Actual WQI')
    plt.plot(test_dates, test_preds, 'r--', label='Predicted WQI')
    plt.title(f'Water Quality Index Prediction (R²={r2_score(y_test, test_preds):.4f})')
    plt.xlabel('Date')
    plt.ylabel('WQI')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('wqi_predictions.png')
    plt.show()

    # Interactive prediction
    while True:
        print("\nEnter date (YYYY-MM-DD) or 'q' to quit:")
        date = input("> ").strip()

        if date.lower() == 'q':
            break

        pred = predict(model, df, scaler, features, date)
        if pred is not None:
            level = get_pollutant_level(pred)
            print(f"\nPredicted WQI for {date}: {pred:.1f}")
            print(f"Pollutant Level: {level}")

            if pd.to_datetime(date) in df.index:
                actual = df.loc[date, 'WQI']
                error = abs(pred - actual)
                print(f"Actual WQI: {actual:.1f} (Error: {error:.1f})")
                print(f"Accuracy: {max(0, 100 - error):.1f}%")


if __name__ == "__main__":
    main()