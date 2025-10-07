# automate.py

import pandas as pd
import os
from preprocessing import preprocess_data


def automate_preprocessing(file_path, target_column="Price"):
    """
    Otomatisasi preprocessing dataset House Price Makassar
    """
    print(f"📂 Loading dataset dari {file_path} ...")
    df = pd.read_csv(file_path)

    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        target_column=target_column,
        drop_columns=["Timestamp", "Title", "Lokasi"],
        save_path="preprocessor_pipeline.joblib"
    )

    print("✅ Preprocessing selesai. Dataset siap dilatih 🚀")

    # Pastikan folder preprocessing ada
    os.makedirs("preprocessing", exist_ok=True)

    # Simpan hasil
    X_train.to_csv("preprocessing/X_train.csv", index=False)
    X_test.to_csv("preprocessing/X_test.csv", index=False)
    y_train.to_csv("preprocessing/y_train.csv", index=False)
    y_test.to_csv("preprocessing/y_test.csv", index=False)

    print("💾 Dataset berhasil disimpan (X_train.csv, X_test.csv, y_train.csv, y_test.csv)")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "housePrice_makassar_from_rumah123.csv")

    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = automate_preprocessing(file_path)
