# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data, save_path, target_column="Price", drop_columns=None, test_size=0.2, random_state=42):
    """
    Fungsi preprocessing dataset House Price Makassar
    Tahapan:
    1. Hapus duplikasi
    2. Tangani outlier dengan IQR
    3. Imputasi nilai Price minimum dengan rata-rata
    4. Split train-test
    5. Standarisasi fitur numerik

    Parameters
    ----------
    data : pd.DataFrame
        Dataset input
    target_column : str
        Kolom target (default: 'Price')
    drop_columns : list
        Kolom yang ingin dihapus selain target
    test_size : float
        Proporsi data test
    random_state : int
        Seed random

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame
    """

    df = data.copy()

    # --- 1. Hapus duplikasi ---
    df = df.drop_duplicates()

    # --- 2. Tangani outlier dengan metode IQR ---
    numeric_cols = df.select_dtypes(include="number").columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    filter_outliers = ~(
        (df[numeric_cols] < (Q1 - 1.5 * IQR)) |
        (df[numeric_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    df = df[filter_outliers]

    # --- 3. Imputasi nilai target minimum dengan rata-rata ---
    if target_column in df.columns:
        mean_price = df[target_column].mean()
        min_price_index = df[df[target_column] == df[target_column].min()].index
        if len(min_price_index) > 0:
            df.loc[min_price_index[0], target_column] = mean_price

    # --- 4. Drop kolom yang tidak dipakai ---
    if drop_columns is None:
        drop_columns = []
    X = df.drop(columns=[target_column] + drop_columns, errors="ignore")
    y = df[target_column]

    # --- 5. Split train-test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- 6. Standarisasi fitur numerik ---
    numerical_features = ["Jumlah Kamar Tidur", "Jumlah Kamar Mandi", "Luas Bangunan", "Luas Tanah", "Car Port"]
    scaler = StandardScaler()

    # Fit di train, transform di train & test
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    print("✅ Preprocessing selesai.")
    print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

    return X_train, X_test, y_train, y_test
