#pragma once
#include <vector>
#include "types.h"
#include "linear.h"

namespace Evaluation {
      // Вычисляет среднюю абсолютную ошибку(Mean Absolute Error) для датасета
// Аргументы:
//  - data: вектор точек Point2D, каждая точка содержит координаты x и y и метку
//  - k, b: коэффициенты прямой y = k*x + b, относительно которой считаем ошибку
// Возвращает:
//  - среднее значение абсолютной разницы между реальными y и предсказанными по прямой y = k*x + b
    static float meanAbsError(const std::vector<Point2D>& data, float k, float b) {
        float total = 0;

        // Проходим по всем точкам в датасете
        for (auto& p : data) {

            // Вычисляем предсказанное значение y по прямой
            float pred = k * p.x + b;

           // Суммируем абсолютную разницу между реальным y и предсказанным
            total += Linear::absDiff(p.y, pred);
        }

        // Возвращаем среднюю абсолютную ошибку
        return total / data.size();
    }// 3 человек 
    static float accuracyLinear(const std::vector<Point2D>& data, float k, float b) {
        int correct = 0;

        for (const auto& p : data) {
            int predicted = (p.y > k * p.x + b) ? 1 : 0;
            if (predicted == static_cast<int>(p.label)) {
                correct++;
            }
        }

        return static_cast<float>(correct) / data.size();
    }
}
