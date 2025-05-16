import re
import matplotlib.pyplot as plt

# Функция для парсинга строки
def parse_line(line):
    pattern = r'Size: (\d+), Threads: (\d+), Time: (\d+) ms'
    match = re.match(pattern, line)
    if match:
        size = int(match.group(1))
        threads = int(match.group(2))
        time = int(match.group(3))
        return size, threads, time
    return None, None, None

# Чтение данных из файла
data = []
with open('data/results.txt', 'r') as file:
    for line in file:
        size, threads, time = parse_line(line)
        if size is not None:
            data.append((size, threads, time))

# Группировка данных по количеству потоков
grouped_data = {}
for size, threads, time in data:
    if threads not in grouped_data:
        grouped_data[threads] = []
    grouped_data[threads].append((size, time))

# Построение графика
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Зависимость времени выполнения от размера матрицы')
ax.set_xlabel('Размер матрицы')
ax.set_ylabel('Время выполнения, мс')

for threads, data_points in grouped_data.items():
    sizes, times = zip(*data_points)
    ax.plot(sizes, times, label=f'Количество потоков: {threads}')

ax.legend()
plt.grid()
plt.savefig("data/graph.png")
plt.show()

