import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""
Persiapan Data
"""
# muat dataset diabetes
diabetes = load_diabetes()

# konversi dataset menjadi dataframe
df = pd.DataFrame(data = np.c_[diabetes['data'], diabetes['target']], columns = diabetes['feature_names'] + ['target'])

# pisahkan fitur (X) dan target (y)
X = df.drop('target', axis=1)
y = df['target']

# bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Pembuatan Model dan Pelatihan
"""
# buat tiga model dengan n_estimators yang berbeda
model_50 = RandomForestRegressor(n_estimators=50, random_state=42)
model_100 = RandomForestRegressor(n_estimators=100, random_state=42)
model_200 = RandomForestRegressor(n_estimators=200, random_state=42)

# latih setiap model
model_50.fit(X_train, y_train)
model_100.fit(X_train, y_train)
model_200.fit(X_train, y_train)


"""
Melakukan prediksi dan Evaluasi Model
"""
# lakukan prediksi
pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)
pred_200 = model_200.predict(X_test)

# hitung RMSE dan MAE untuk setiap model
rmse_50 = np.sqrt(mean_squared_error(y_test, pred_50))
mae_50 = mean_absolute_error(y_test, pred_50)

rmse_100 = np.sqrt(mean_squared_error(y_test, pred_100))
mae_100 = mean_absolute_error(y_test, pred_100)

rmse_200 = np.sqrt(mean_squared_error(y_test, pred_200))
mae_200 = mean_absolute_error(y_test, pred_200)

# Tampilkan hasil evaluasi
print("Model dengan n_estimators=50:")
print("RMSE:", rmse_50)
print("MAE:", mae_50)
print("\nModel dengan n_estimators=100:")
print("RMSE:", rmse_100)
print("MAE:", mae_100)
print("\nModel dengan n_estimators=200:")
print("RMSE:", rmse_200)
print("MAE:", mae_200)