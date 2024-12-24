import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

url = 'population_data.csv'
data = pd.read_csv(url, index_col='Year', parse_dates=True)
ts = data['Population size']

seasonal_periods = 10

def holt_winters(ts, alpha, beta, gamma, seasonal_periods):
    n = len(ts)
    a = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)
    y_hat = np.zeros(n)

    a[0] = ts[:seasonal_periods].mean()
    b[0] = (ts[seasonal_periods:2*seasonal_periods].mean() - ts[:seasonal_periods].mean()) / seasonal_periods
    for i in range(seasonal_periods):
        s[i] = ts[i] - a[0]

    for t in range(1, n):
        a[t] = alpha * (ts[t] - s[t % seasonal_periods]) + (1 - alpha) * (a[t-1] + b[t-1])
        b[t] = beta * (a[t] - a[t-1]) + (1 - beta) * b[t-1]
        s[t] = gamma * (ts[t] - a[t]) + (1 - gamma) * s[t % seasonal_periods]
        y_hat[t] = a[t] + b[t] + s[t % seasonal_periods]

    return a, b, s, y_hat

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

alpha_values = np.linspace(0.1, 0.9, 9)
beta_values = np.linspace(0.01, 0.3, 6)
gamma_values = np.linspace(0.01, 0.3, 6)

best_mse = float('inf')
best_params = {}

for alpha, beta, gamma in product(alpha_values, beta_values, gamma_values):
    forecast = holt_winters(ts.values, alpha, beta, gamma, seasonal_periods)
    current_mse = mse(ts.values, forecast)
    if current_mse < best_mse:
        best_mse = current_mse
        best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}

print(f"Оптимальные параметры: {best_params}")

n_preds = 10
a, b, s, forecast = holt_winters(ts.values, best_params['alpha'], best_params['beta'], best_params['gamma'], seasonal_periods)

forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1), periods=n_preds, freq='YS')

plt.figure(figsize=(10, 6))
plt.plot(ts.index, ts, label='Исходные данные')
plt.plot(forecast_dates, forecast[-n_preds:], label='Прогноз', marker='.')
plt.legend(loc='best')
plt.title('Модель Хольта-Уинтерса')
plt.xlabel('Год')
plt.ylabel('Численность населения')
plt.show()

forecast_series = pd.Series(forecast[-n_preds:], index=forecast_dates)
print("Прогноз на следующие 10 лет:")
print(forecast_series)

print("\nКоэффициенты парной регрессии:")
print(f"Коэффициент наклона (a): {a[-1]:.3f}")
print(f"Свободный член (b): {b[-1]:.3f}")
print(f"Сезонность (s):")
for S in s[-seasonal_periods:]:
    print(F"{S:.3f}")