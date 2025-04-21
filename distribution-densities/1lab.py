import numpy as np
import matplotlib.pyplot as plt

# Коэффициент k
k = 0.3747

# Функция обратного преобразования
def inverse_transform(u):
    pi_6 = np.pi / 6
    threshold = (np.pi ** 4) / 324  # 0.3006

    if u <= threshold:
        return (4 * u) ** (1 / 4) - pi_6
    else:
        return np.arccos((np.sqrt(3) / 2) - (u - threshold) / k)

# Генерация случайных чисел
N = 10000  # Количество выборок
random_numbers = np.random.uniform(0, 1, N)
samples = np.array([inverse_transform(u) for u in random_numbers])

# Функция плотности f(x)
def f_x(x):
    if -np.pi/6 <= x < np.pi/6:
        return (x + np.pi/6) ** 3
    elif np.pi/6 <= x <= np.pi:
        return k * np.sin(x)
    return 0

# Создаем точки для f(x)
x_vals = np.linspace(-np.pi/6 - 0.2, np.pi + 0.2, 500)
y_vals = np.array([f_x(x) for x in x_vals])

# Построение гистограммы и графика плотности
plt.figure(figsize=(10, 5))
plt.hist(samples, bins=50, density=True, alpha=0.6, label="Гистограмма случайных чисел")
plt.plot(x_vals, y_vals, 'r', label="Плотность f(x)", linewidth=2)
plt.xlabel("x")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.title("Гистограмма случайных чисел и плотность распределения f(x)")
plt.grid()
plt.show()
