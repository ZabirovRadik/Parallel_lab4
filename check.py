import os
import numpy as np

base_dir = "data"

def read_matrix_from_file(filename):
    matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = [int(x) for x in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)


def check_multiplication(matrix_a, matrix_b, matrix_c, num_threads):
    expected_result = matrix_a @ matrix_b
    if np.array_equal(expected_result, matrix_c):
        print(f"Multiplication with {num_threads} threads is correct.")
    else:
        print(f"Multiplication with {num_threads} threads is incorrect.")

with open('data/trust_me.txt', 'w') as f:
    for matrix_dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, matrix_dir)):
            matrix_a = read_matrix_from_file(os.path.join(base_dir, matrix_dir, "A.txt"))
            matrix_b = read_matrix_from_file(os.path.join(base_dir, matrix_dir, "B.txt"))
            mul_matrices = matrix_a @ matrix_b

            error_with = list()
            for thread_dir in os.listdir(os.path.join(base_dir, matrix_dir)):
                if os.path.isdir(os.path.join(base_dir, matrix_dir, thread_dir)) and thread_dir.startswith("threads_"):
                    num_threads = int(thread_dir.split("_")[1])
                    matrix_c = read_matrix_from_file(os.path.join(base_dir, matrix_dir, thread_dir, "multiplyed.txt"))
                    if not np.array_equal(mul_matrices, matrix_c):
                        error_with.append(num_threads)
            
            if not error_with:
                f.write(f"Для {matrix_a.shape[1]} посчитано правильно с разным количеством потоков")
            else:
                f.write(f"Для {matrix_a.shape[1]} неправильно посчитано для количества потоков: ")
                for thread in error_with:
                    f.write(f"{thread}, ")
            f.write("\n")