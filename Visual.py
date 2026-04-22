import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.interpolate import griddata

# Загрузка данных
points = pd.read_csv("points.csv")          # исходные точки (x, y, true_label)
pred = pd.read_csv("predictions.csv")       # предсказания (x, y, true_label, pred_label, probability)

# Параметры линейной границы (из main.cpp)
k, b = 0.75, -0.25

# ---- 1. Сравнение истинных и предсказанных классов ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Левый график: истинные метки
ax = axes[0]
ax.scatter(points[points['label']==0]['x'], points[points['label']==0]['y'],
           c='blue', label='Class 0 (true)', alpha=0.6)
ax.scatter(points[points['label']==1]['x'], points[points['label']==1]['y'],
           c='red', label='Class 1 (true)', alpha=0.6)
x_line = np.linspace(points['x'].min()-0.5, points['x'].max()+0.5, 200)
ax.plot(x_line, k*x_line + b, 'g-', lw=2, label='y = kx + b')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('True labels')
ax.legend(); ax.grid(True)

# Правый график: предсказания нейросети
ax = axes[1]
pred0 = pred[pred['pred_label'] == 0]
pred1 = pred[pred['pred_label'] == 1]
wrong = pred[pred['true_label'] != pred['pred_label']]
ax.scatter(pred0['x'], pred0['y'], c='cyan', label='Predicted 0', alpha=0.6)
ax.scatter(pred1['x'], pred1['y'], c='orange', label='Predicted 1', alpha=0.6)
if not wrong.empty:
    ax.scatter(wrong['x'], wrong['y'], marker='x', c='black', s=80, label='Misclassified')
ax.plot(x_line, k*x_line + b, 'g--', lw=2, label='Linear boundary')
ax.set_xlabel('X'); ax.set_ylabel('Y')
ax.set_title('Neural network predictions')
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('nn_comparison.png', dpi=150)
plt.show()

# ---- 2. Матрица ошибок ----
cm = confusion_matrix(pred['true_label'], pred['pred_label'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Neural Network)')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ---- 3. Тепловая карта вероятностей (интерполяция) ----
# Создаём сетку
x_grid = np.linspace(points['x'].min(), points['x'].max(), 100)
y_grid = np.linspace(points['y'].min(), points['y'].max(), 100)
X, Y = np.meshgrid(x_grid, y_grid)
# Интерполируем вероятности
Z = griddata(pred[['x','y']].values, pred['probability'].values, (X, Y), method='cubic')

plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, levels=50, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Probability of class 1')
plt.scatter(points[points['label']==0]['x'], points[points['label']==0]['y'],
            c='blue', edgecolors='k', s=20, label='True class 0')
plt.scatter(points[points['label']==1]['x'], points[points['label']==1]['y'],
            c='red', edgecolors='k', s=20, label='True class 1')
plt.plot(x_line, k*x_line + b, 'g-', lw=2, label='Linear boundary')
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Neural network decision probability')
plt.legend()
plt.grid(True)
plt.savefig('probability_heatmap.png', dpi=150)
plt.show()

# ---- 4. Сравнение точности линейной модели и нейросети ----
linear_correct = (points['label'] == (points['y'] > k*points['x'] + b).astype(int)).mean()
nn_correct = (pred['true_label'] == pred['pred_label']).mean()
print(f"\nLinear model accuracy: {linear_correct:.4f}")
print(f"Neural network accuracy: {nn_correct:.4f}")
print(f"Improvement: {(nn_correct - linear_correct)*100:.2f}%")