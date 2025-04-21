import numpy as np
import matplotlib.pyplot as plt

# Параметры
n_states = 10  # Количество состояний
n_steps = 100  # Длина цепи Маркова

# Функция проверки двустохастичности
def is_doubly_stochastic(matrix, tolerance=1e-3):
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    return (np.allclose(row_sums, 1.0, atol=tolerance) and 
           (np.allclose(col_sums, 1.0, atol=tolerance)))

# Функция для создания двухстохастической матрицы
def create_doubly_stochastic_matrix(n):
    matrix = np.random.rand(n, n)
    for _ in range(10):  # Увеличили число итераций для лучшей сходимости
        matrix /= matrix.sum(axis=1, keepdims=True)
        matrix /= matrix.sum(axis=0, keepdims=True)
    return matrix

def print_matrix(matrix, name, max_size=10):
    print(f"\n{name} ({matrix.shape[0]}x{matrix.shape[1]}):")
    n = min(max_size, matrix.shape[0])
    for row in matrix[:n]:
        print(' '.join([f"{x:.3f}" for x in row[:n]]))
        if n < matrix.shape[0]: print("...")
    if n < matrix.shape[0]: print("...")

# Создаем и проверяем матрицы
matrix1 = create_doubly_stochastic_matrix(n_states)
matrix2 = create_doubly_stochastic_matrix(n_states)

# Функция для генерации цепи Маркова
def generate_markov_chain(matrix, n_steps):
    states = [np.random.randint(0, matrix.shape[0])]
    for _ in range(n_steps - 1):
        current_state = states[-1]
        next_state = np.random.choice(matrix.shape[0], p=matrix[current_state])
        states.append(next_state)
    return np.array(states)

def generate_exponential_values(states, scale=1.0):
    return np.random.exponential(scale, size=len(states))

def normalize_values(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)

def compute_frequencies(states, n_states):
    return np.bincount(states, minlength=n_states) / len(states)

try:
    states1 = generate_markov_chain(matrix1, n_steps)
    states2 = generate_markov_chain(matrix2, n_steps)
except ValueError as e:
    print(f"Ошибка: {e}")
    exit()

values1 = generate_exponential_values(states1)
normalized_values1 = normalize_values(values1)
frequencies1 = compute_frequencies(states1, n_states)

values2 = generate_exponential_values(states2)
normalized_values2 = normalize_values(values2)
frequencies2 = compute_frequencies(states2, n_states)


print("\nПроверка матриц:")
print_matrix(matrix1, "Матрица 1")
print(f"• Двустохастическая: {is_doubly_stochastic(matrix1)}")
print("Сгенерированные значения:", values1)
print("Переходы матрицы:", states1)
print("Частоты  матрицы: ",frequencies1)

print_matrix(matrix2, "Матрица 2")
print(f"• Двустохастическая: {is_doubly_stochastic(matrix2)}")
print("Сгенерированные значения:", values2)
print("Переходы матрицы:", states2)
print("Частоты матрицы: ",frequencies2)

# Построение графиков
plt.figure(figsize=(14, 12))

plt.subplot(2, 2, 1)
plt.plot(states1[:100], label='Матрица 1', color='blue')
plt.plot(states2[:100], label='Матрица 2', color='red', linestyle='--')
plt.title("Сравнение траекторий состояний")
plt.xlabel("Шаг")
plt.ylabel("Состояние")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(normalized_values1[:100], label='Матрица 1', color='blue')
plt.plot(normalized_values2[:100], label='Матрица 2', color='red', linestyle='--')
plt.title("Сравнение нормированных значений")
plt.xlabel("Шаг")
plt.ylabel("Значение")
plt.legend()

plt.subplot(2, 2, 3)
plt.acorr(normalized_values1, maxlags=20, color='blue', label='Матрица 1')
plt.acorr(normalized_values2, maxlags=20, color='red', linestyle='--', label='Матрица 2')
plt.title("Сравнение автокорреляции")
plt.xlabel("Лаг")
plt.ylabel("Автокорреляция")
plt.legend()

plt.subplot(2, 2, 4)
bar_width = 0.4
x = np.arange(n_states)
plt.bar(x - bar_width/2, frequencies1, width=bar_width, label='Матрица 1', color='blue')
plt.bar(x + bar_width/2, frequencies2, width=bar_width, label='Матрица 2', color='red')
plt.title("Сравнение частот посещений состояний")
plt.xlabel("Состояние")
plt.ylabel("Частота")
plt.legend()

plt.tight_layout()
plt.show()