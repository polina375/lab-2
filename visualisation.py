import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загружаем CSV (поменяй путь, если нужно)
data = pd.read_csv("points.csv") 

# Разделяем по классам
class0 = data[data['label'] == 0]
class1 = data[data['label'] == 1]

# Параметры линии (используй те же, что в main.cpp)
k = 0.75
b = -0.25
x_line = np.linspace(data['x'].min()-0.5, data['x'].max()+0.5, 200)
y_line = k * x_line + b

# Рисуем график
plt.figure(figsize=(8,6))
plt.scatter(class0['x'], class0['y'], color='blue', label='Class 0', alpha=0.7)
plt.scatter(class1['x'], class1['y'], color='red', label='Class 1', alpha=0.7)
plt.plot(x_line, y_line, color='green', linewidth=2, label='y = kx + b')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Binary Classification Visualization')
plt.legend()
plt.grid(True)
plt.show()
