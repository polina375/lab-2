#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "types.h"
#include "neuralnet.h"
#include "console.h"

namespace Metrics {
    // Структура для хранения основных метрик качества бинарной классификации
    struct ClassificationMetrics {
        float accuracy = 0.0f; // Доля правильно классифицированных точек
        float precision = 0.0f; // Точность предсказания положительного класса
        float recall = 0.0f; // Полнота (сколько реальных положительных точек найдено)
        float f1 = 0.0f;    // F1-мера — гармоническое среднее precision и recall
    }; 

    // ================================================
    // 1. Вычисление всех метрик
    // ================================================

    // Подсчитывает TP, TN, FP, FN и вычисляет Accuracy, Precision, Recall и F1-score
    ClassificationMetrics evaluate(Neural::NeuralNetwork<float>& net,
        const std::vector<Point2D>& data)
    {
        int TP = 0, TN = 0, FP = 0, FN = 0;

        for (const auto& p : data) {
            int true_label = static_cast<int>(p.label);
            int pred = net.predictClass(p);

            if (true_label == 1 && pred == 1) TP++;
            else if (true_label == 0 && pred == 0) TN++;
            else if (true_label == 0 && pred == 1) FP++;
            else if (true_label == 1 && pred == 0) FN++;
        }

        float total = static_cast<float>(TP + TN + FP + FN);
        float accuracy = (total > 0) ? static_cast<float>(TP + TN) / total : 0.0f;
        float precision = (TP + FP > 0) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
        float recall = (TP + FN > 0) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
        float f1 = (precision + recall > 0.0f)
            ? 2.0f * precision * recall / (precision + recall)
            : 0.0f;

        return { accuracy, precision, recall, f1 };
    }

    // ================================================
    // 2. Сохранение предсказаний в CSV
    // ================================================
    // Создаёт файл predictions.csv с колонками : x, y, true_label, pred_label, probability
    bool savePredictions(Neural::NeuralNetwork<float>& net,
        const std::vector<Point2D>& data,
        const char* filename)
    {
        std::ofstream file(filename);
        if (!file) {
            Console::info("Error: cannot open predictions file");
            return false;
        }

        file << "x,y,true_label,pred_label,probability\n";

        for (const auto& p : data) {
            float prob = net.forward(p);
            int pred = net.predictClass(p);

            file << p.x << "," << p.y << ","
                << static_cast<int>(p.label) << ","
                << pred << "," << prob << "\n";
        }

        Console::info("Predictions saved to");
        Console::info(filename);
        return true;
    }

    // ================================================
    // 3. Вывод матрицы ошибок (Confusion Matrix)
    // ================================================
    // Показывает, сколько точек правильно и неправильно классифицировано
    void printConfusionMatrix(Neural::NeuralNetwork<float>& net,
        const std::vector<Point2D>& data)
    {
        int TP = 0, TN = 0, FP = 0, FN = 0;

        for (const auto& p : data) {
            int true_label = static_cast<int>(p.label);
            int pred = net.predictClass(p);

            if (true_label == 1 && pred == 1) TP++;
            else if (true_label == 0 && pred == 0) TN++;
            else if (true_label == 0 && pred == 1) FP++;
            else if (true_label == 1 && pred == 0) FN++;
        }

        std::cout << "\n Confusion Matrix \n";
        std::cout << "          Predicted\n";
        std::cout << "          0      1\n";
        std::cout << "True 0   " << TN << "      " << FP << "\n";
        std::cout << "     1   " << FN << "      " << TP << "\n\n";
    }

    // ================================================
    // 4. Демонстрация предсказаний на сетке точек
    // ================================================

    // Показывает, как сеть предсказывает класс для точек на равномерной сетке
    // Используется для визуальной проверки поведения модели
    void predictOnGrid(Neural::NeuralNetwork<float>& net,
        float k, float b, int steps = 8)
    {
        Console::info("Demo: Predictions on 8x8 grid");

        float step = 4.0f / (steps - 1.0f);
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "   x      y      prob    pred   true\n";
        std::cout << "------------------------------------\n";

        for (int i = 0; i < steps; ++i) {
            float x = -2.0f + i * step;
            for (int j = 0; j < steps; ++j) {
                float y = -2.0f + j * step;
                Point2D p{ x, y, 0.0f };

                float prob = net.forward(p);
                int pred = net.predictClass(p);
                int true_label = (y > k * x + b) ? 1 : 0;

                std::cout << std::setw(7) << x
                    << std::setw(7) << y
                    << std::setw(8) << prob
                    << std::setw(6) << pred
                    << std::setw(6) << true_label << "\n";
            }
        }
        std::cout << "\n";
    }
} // namespace Metrics
