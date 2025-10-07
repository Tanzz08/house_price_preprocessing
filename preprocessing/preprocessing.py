import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump


def preprocess_data(
    data,
    save_path,
    target_column="Price",
    drop_columns=None,
    test_size=0.2,
    random_state=42
):
    """
    Fungsi preprocessing dataset House Price Makassar
    Tahapan:
    1. Hapus duplikasi
    2. Tangani outlier dengan IQR
    3. Imputasi nilai target minimum dengan rata-rata
    4. Split train-test
    5. Buat pipeline dengan ColumnTransformer
    6. Simpan preprocessor (joblib)
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

    # --- 4. Pisahkan fitur dan target ---
    if drop_columns is None:
        drop_columns = []
    X = df.drop(columns=[target_column] + drop_columns, errors="ignore")
    y = df[target_column]

    # --- 5. Identifikasi tipe kolom ---
    numeric_features = ["Jumlah Kamar Tidur", "Jumlah Kamar Mandi", "Luas Bangunan", "Luas Tanah", "Car Port"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # --- 6. Definisikan transformer ---
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # --- 7. Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --- 8. Fit preprocessor di train dan transform ---
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --- 9. Simpan pipeline ---
    dump(preprocessor, save_path)
    print(f"✅ Pipeline preprocessing disimpan di {save_path}")

    # --- 10. Konversi hasil ke DataFrame ---
    # Catatan: hasil OneHotEncoder bisa menghasilkan banyak kolom baru
    X_train_processed = pd.DataFrame(X_train_transformed.toarray() if hasattr(X_train_transformed, "toarray") else X_train_transformed)
    X_test_processed = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed)

    print("✅ Preprocessing selesai.")
    print(f"Data latih: {X_train_processed.shape}, Data uji: {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train.reset_index(drop=True), y_test.reset_index(drop=True)
