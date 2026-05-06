import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    input_path: str
    outdir: str
    lookback: int = 30  # Use prvious 30 days to predict next day
    epochs: int = 30
    batch_size: int = 32
    lstm_units: int = 64
    dropout: float = 0.2
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"


def find_date_high_columns(df: pd.DataFrame):
    """
    Try to locate Date and High columns.
    """
    # lower case + remove space
    cols_lower = {}
    for c in df.columns:
        key = c.lower().strip()
        cols_lower[key] = c
    # cols_lower = {c.lower().strip(): c for c in df.columns}

    date_col = None
    high_col = None

    for key in cols_lower:
        if key == "date":
            date_col = cols_lower[key]
        if key == "high":
            high_col = cols_lower[key]

    if date_col is None:
        raise ValueError(f"Could not find a Date column. Available columns: {list(df.columns)}")
    if high_col is None:
        raise ValueError(f"Could not find a High column. Available columns: {list(df.columns)}")

    return date_col, high_col


def load_btc_data(path: str) -> pd.DataFrame:
    """
    Load BTC csv, keep only Date and High, parse/sort dates, and clean rows.
    """
    df = pd.read_csv(path)
    date_col, high_col = find_date_high_columns(df)

    df = df[[date_col, high_col]].copy()
    df.columns = ["Date", "High"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")

    df = df.dropna(subset=["Date", "High"]).sort_values("Date").reset_index(drop=True)  # reset_index 重置行号
    return df


def split_train_test(df: pd.DataFrame, train_end: str, test_start: str):
    """
    Split by date:
      Train: <= 2022-12-31
      Test : >= 2023-01-01
    """
    train_df = df[df["Date"] <= pd.Timestamp(train_end)].copy()
    test_df = df[df["Date"] >= pd.Timestamp(test_start)].copy()

    if train_df.empty:
        raise ValueError("Train split is empty. Check the date range in your data.")
    if test_df.empty:
        raise ValueError("Test split is empty. Check the date range in your data.")

    return train_df, test_df


def scale_series(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit MinMaxScaler only on train, then transform train and test.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))  # 缩放到0-1

                                # [[100],
    # [100, 120, 130]   --->    #  [120],
                                #  [130]]
    train_values = train_df["High"].values.reshape(-1, 1)
    test_values = test_df["High"].values.reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    return scaler, train_scaled, test_scaled


def create_sequences(values: np.ndarray, lookback: int):
    """
    Turn a 2D array of shape (n, 1) into:
      X: (n-lookback, lookback, 1)
      y: (n-lookback,)
    The target is the next day after each lookback window.
    """
    X, y = [], []
    for i in range(lookback, len(values)):  # 30 ~ n
        X.append(values[i - lookback:i])
        y.append(values[i, 0])
    # e.g.
    '''
        lookback = 3
        values = [
          [10],
          [11],
          [12],
          [13],
          [14]
        ]
        
        i = 3
        x:  values[i - lookback:i] = values[0:3] = [10, 11, 12]
        y:  values[i, 0] = values[3, 0] = 13
        
        i = 4
        x:  values[1:4] = [11, 12, 13]
        y:  values[i, 0] = values[4, 0] = 14
    '''

    return np.array(X), np.array(y)


def build_lstm_model(lookback: int, lstm_units: int = 64, dropout: float = 0.2):
    # lstm_units: lstm里神经元数量, 默认64
    # dropout比例: 默认0.2, 随机关掉20%, 防止过拟合overfitting
    # LSTM - 专门处理时间序列的神经网络
    """
    Build a simple LSTM regression model.
    """
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(lstm_units, return_sequences=False),  # False - 只输出最后一个时间步的总结果, True - 输出每个时间步的结果
        layers.Dropout(dropout),  # Avoid Overfitting
        layers.Dense(32, activation="relu"),
        layers.Dense(1)  # Final Price - only one node
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",  # MSE = Mean Square Error 均方误差(大误差会被惩罚的更重)
        metrics=["mae"]  # MAE = Mean Absolute Error 平均绝对误差(平均来说,预测值和真实值差多少)
    )
    return model


def make_test_windows(train_scaled: np.ndarray, test_scaled: np.ndarray, lookback: int):
    """
    Build test windows using the last `lookback` points from train + all test points.

    This lets the first test prediction use the final days of train history.
    """
    # To predict price for 2023-01-01
    # Need data from previous 30 days: 2022-12-02 → 2022-12-31
    combined = np.concatenate([train_scaled[-lookback:], test_scaled], axis=0)
    X_test, y_test = create_sequences(combined, lookback)
    return X_test, y_test


def inverse_transform_1d(scaler, arr_1d: np.ndarray):
    """
    Inverse-transform a 1D scaled vector back to original prices.
    """

    '''
    Inverse-transform 逆缩放:
    Min: 16000
    Max: 20000
    0.35 --> 16000 + 0.35 * (20000 - 16000) = 17400

    0.35 → 17400
    0.41 → 17640
    0.52 → 18080
    '''
    arr_2d = arr_1d.reshape(-1, 1)
    inv = scaler.inverse_transform(arr_2d).reshape(-1)

    # .reshape(-1):
    # [[17400],
    #  [17640],  --> [17400, 17640, 18080]
    #  [18080]]

    return inv


def plot_series(train_df, test_df, outpath):
    plt.figure(figsize=(12, 5))
    plt.plot(train_df["Date"], train_df["High"], label="Train High")
    plt.plot(test_df["Date"], test_df["High"], label="Test High")
    plt.xlabel("Date")
    plt.ylabel("BTC High Price")
    plt.title("Bitcoin High Price Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_history(history, outpath):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_predictions(test_dates, actual, predicted, outpath):
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, actual, label="Actual")
    plt.plot(test_dates, predicted, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("BTC High Price")
    plt.title("Actual vs Predicted BTC High Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()  # python btc_lstm.py --input "btc price.csv"
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "btc price.csv"),
        help="Path to BTC CSV file"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "btc"),
        help="Output folder"
    )
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    cfg = Config(
        input_path=args.input,
        outdir=args.outdir,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
    )

    os.makedirs(cfg.outdir, exist_ok=True)

    print("Loading BTC data...")
    df = load_btc_data(cfg.input_path)

    print("Splitting train/test by date...")
    train_df, test_df = split_train_test(df, cfg.train_end, cfg.test_start)

    print("Saving raw price plot...")
    plot_series(train_df, test_df, os.path.join(cfg.outdir, "btc_series.png"))

    print("Scaling data...")
    scaler, train_scaled, test_scaled = scale_series(train_df, test_df)

    print("Creating train sequences...")
    X_train, y_train = create_sequences(train_scaled, cfg.lookback)

    print("Creating test sequences...")
    X_test, y_test = make_test_windows(train_scaled, test_scaled, cfg.lookback)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Not enough data to create sequences. Try a smaller lookback.")

    # Time-series friendly validation split: use the last 10% of train sequences
    val_size = max(1, int(len(X_train) * 0.1))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    print(f"Train sequences: {X_tr.shape}, Val sequences: {X_val.shape}, Test sequences: {X_test.shape}")

    print("Building LSTM model...")
    model = build_lstm_model(
        lookback=cfg.lookback,
        lstm_units=cfg.lstm_units,
        dropout=cfg.dropout,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
    ]

    print("Training model...")
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )

    print("Saving loss plot...")
    plot_history(history, os.path.join(cfg.outdir, "training_loss.png"))

    print("Predicting on test set...")
    test_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)

    # Inverse transform back to actual price scale
    y_test_actual = inverse_transform_1d(scaler, y_test)
    y_test_pred = inverse_transform_1d(scaler, test_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    mae = mean_absolute_error(y_test_actual, y_test_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")

    # Dates aligned with test targets
    test_dates = test_df["Date"].iloc[:len(y_test)].reset_index(drop=True)

    print("Saving prediction plot...")
    plot_predictions(
        test_dates,
        y_test_actual,
        y_test_pred,
        os.path.join(cfg.outdir, "actual_vs_predicted.png")
    )

    # Save comparison table
    results_df = pd.DataFrame({
        "Date": test_dates,
        "Actual_High": y_test_actual,
        "Predicted_High": y_test_pred,
        "Abs_Error": np.abs(y_test_actual - y_test_pred)
    })
    results_path = os.path.join(cfg.outdir, "btc_predictions.csv")
    results_df.to_csv(results_path, index=False)

    print(f"Saved predictions to: {results_path}")
    print("Saved plots:")
    print(f"  - {os.path.join(cfg.outdir, 'btc_series.png')}")
    print(f"  - {os.path.join(cfg.outdir, 'training_loss.png')}")
    print(f"  - {os.path.join(cfg.outdir, 'actual_vs_predicted.png')}")

    # Print a few rows
    print("\nSample predictions:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()