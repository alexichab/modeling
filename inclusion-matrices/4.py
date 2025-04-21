import numpy as np

def generate_sequence(n):
    return np.arange(1, n + 1)

def draw_sample_replacement(data, count, replace = True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [int(x) for x in np.random.choice(data, size=count, replace=replace)]

def build_freq_matrix(base_seq, sample, iterations=3, use_seed=True):
    n = len(sample)
    m = len(base_seq)
    freq_mat = np.zeros((n, m), dtype=int)
    
    base_int = [int(x) for x in base_seq]
    index_map = {val: idx for idx, val in enumerate(base_int)}
    
    for _ in range(iterations):
        seed_val = np.random.randint(0, 10000) if use_seed else None
        drawn = draw_sample_replacement(sample, count=n, seed=seed_val)
        for pos, val in enumerate(drawn):
            freq_mat[pos, index_map[val]] += 1
    return freq_mat

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{int(val):2d}" for val in row))

# Определяем критерии фильтрации
def is_even(x):
    return x % 2 == 0

def divisible_by_three(x):
    return x % 3 == 0

def is_perfect_square(x):
    return int(np.sqrt(x)) ** 2 == x

import numpy as np

def generate_sequence(n):
    return np.arange(1, n + 1)

def draw_sample_replacement(data, count, replace = True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [int(x) for x in np.random.choice(data, size=count, replace=replace)]

def build_freq_matrix(base_seq, filtered, demo_count, iterations=10, use_seed=True, replace=True):
    n = demo_count  # Размер выборки в каждой итерации
    m = len(base_seq)
    freq_mat = np.zeros((n, m), dtype=int)
    
    base_int = [int(x) for x in base_seq]
    index_map = {val: idx for idx, val in enumerate(base_int)}
    
    for _ in range(iterations):
        seed_val = np.random.randint(0, 10000) if use_seed else None
        # Делаем новую выборку из filtered в каждой итерации
        drawn = draw_sample_replacement(filtered, count=n, replace=replace, seed=seed_val)
        for pos, val in enumerate(drawn):
            freq_mat[pos, index_map[val]] += 1
    return freq_mat



def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{int(val):2d}" for val in row))

# Определяем критерии фильтрации
def is_even(x):
    return x % 2 == 0

def divisible_by_three(x):
    return x % 3 == 0

def is_perfect_square(x):
    return int(np.sqrt(x)) ** 2 == x

def main():
    sizes = [10, 30]
    criteria = [
        ("Четные числа", is_even),
        ("Делятся на три", divisible_by_three),
        ("Квадратные числа", is_perfect_square)
    ]
    
    for size in sizes:
        base = generate_sequence(size)
        base_list = [int(x) for x in base.tolist()]
        print(f"\nБазовый массив (size={size}): {base_list}")
        
        for crit_name, crit_func in criteria:
            filtered = [int(x) for x in base if crit_func(x)]
            print(f"\nВыборка ({len(filtered)} элементов) по критерию: {crit_name}")
            print(filtered)
            
            if not filtered:
                print(f"Нет элементов, удовлетворяющих критерию '{crit_name}'.")
                continue
            
            demo_count = min(len(filtered), 3)
            
            sample_rep = draw_sample_replacement(filtered, count=demo_count, replace=True)
            print("\nПример выборки с возвращением:", sample_rep)
            
            # Выборка с возвращением
            freq_matrix_single = build_freq_matrix(base, filtered, demo_count, iterations=10, use_seed=True, replace=True)
            print("Частотная матрица выборки с возвращением c 10 запусками:")
            print_matrix(freq_matrix_single)
            
            sample_no_rep = draw_sample_replacement(filtered, count=demo_count, replace=False)
            print("\nПример выборки без возвращения:", sample_no_rep)
            
            # Выборка без возвращения
            freq_matrix_multi = build_freq_matrix(base, filtered, demo_count, iterations=10, use_seed=True, replace=False)
            print("Частотная матрица выборки без возвращения c 10 запусками:")
            print_matrix(freq_matrix_multi)

if __name__ == "__main__":
    main()
