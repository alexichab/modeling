import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_points(num_points, size=10, seed=None):
    """Генерация случайного набора точек. Если seed не указан, используется случайное значение."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.uniform(0, size, (num_points, 2))
    return points

def prob_exp(d, a, b):
    """Экспоненциальная вероятность: P(d) = e^(-a * d^b)"""
    return np.exp(-a * d**b)

def prob_inverse(d, b):
    """Степенная вероятность: P(d) = min(1, 10 / d^b), если d > 1, иначе 1"""
    return min(1, 10 / d**b) if d > 1 else 1

def build_random_tree(points, prob_func, params, method, max_degree=None, min_dist=2, max_dist=40):
    """Построение случайного дерева на основе вероятностных функций с ограничениями"""
    n = len(points)
    G = nx.Graph()
    G.add_node(0)  # Начинаем с корневой вершины
    connected_nodes = {0}

    for j in range(1, n):
        best_i = None
        max_p = 0

        # Пробуем соединить с существующей вершиной на основе вероятности
        for i in connected_nodes:
            if max_degree and G.degree(i) >= max_degree:
                continue
            d = np.linalg.norm(points[i] - points[j])
            if min_dist <= d <= max_dist:
                p = prob_func(d, *params)
                if p > max_p and np.random.rand() < p:
                    max_p = p
                    best_i = i

        if best_i is not None:
            G.add_edge(best_i, j)
        else:
            # Резервный вариант: соединяем с ближайшей доступной вершиной
            available_nodes = [i for i in connected_nodes if not max_degree or G.degree(i) < max_degree]
            if available_nodes:
                closest = min(available_nodes, key=lambda i: np.linalg.norm(points[i] - points[j]))
                G.add_edge(closest, j)
        connected_nodes.add(j)

    return G

def plot_tree(G, points, title, filename, root=0):
    """Отрисовка дерева с выделением корневой вершины"""
    pos = {i: points[i] for i in range(len(points))}
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=50, node_color='blue', with_labels=False)
    # Выделяем корневую вершину
    plt.scatter(points[root, 0], points[root, 1], color='red', s=100, label='Root')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Параметры
num_points = 300
size = 30

# Эксперименты
a_values = [0.1, 0.5, 1.0]
b_values = [0.2, 0.5, 1.0]
max_degrees = [3, 5, None]  # None означает отсутствие ограничения

for a in a_values:
    for b in b_values:
        for max_deg in max_degrees:
            # Генерация нового набора точек для каждого эксперимента
            points = generate_points(num_points, size)

            # Модель экспоненциального затухания
            G_exp = build_random_tree(points, prob_exp, (a, b), method="exp", max_degree=max_deg)
            title_exp = f'Экспоненциальное затухание: a={a}, b={b}, max_deg={max_deg}'
            filename_exp = f'exp_a{a}_b{b}_deg{max_deg}.png'
            plot_tree(G_exp, points, title_exp, filename_exp)

            # Модель степенного закона
            G_power = build_random_tree(points, prob_inverse, (b,), method="inverse", max_degree=max_deg)
            title_power = f'Степенной закон: b={b}, max_deg={max_deg}'
            filename_power = f'power_b{b}_deg{max_deg}.png'
            plot_tree(G_power, points, title_power, filename_power)
