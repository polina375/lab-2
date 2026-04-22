#include "head/types.h"
#include "head/dataset.h"
#include "head/linear.h"
#include "head/evaluation.h"
#include "head/console.h"
#include "head/neuralnet.h"
#include "head/trainer.h"
#include "head/metrics.h"    
#include <ctime>
#include <cstdlib>

int main() {
    // Фиксируем seed для повторяемости результатов
    srand(42);

    Console::info("Laboratory Work 2: Neural Network Binary Classification");

    float k = 0.75f;
    float b = -0.25f;

    // 1. Генерация датасета
    Console::info("Generating dataset...");
    auto allData = Dataset::generate(500, k, b);

    // 2. Разделение на обучающую и тестовую выборки (80/20)
    auto [trainData, testData] = Dataset::split(allData, 0.8f);

    Console::value("Total samples", static_cast<float>(allData.size()));
    Console::value("Train samples", static_cast<float>(trainData.size()));
    Console::value("Test samples", static_cast<float>(testData.size()));

    // 3. Создание и обучение нейронной сети
    Console::info("Creating Neural Network (hidden size = 8)");
    Neural::NeuralNetwork<float> net(8);

    Console::info("Training network with backpropagation...");
    Neural::Trainer::train(net, trainData, 250, 0.12f);

    // 4. Оценка качества на тестовой выборке
    Console::info("=== Evaluation on Test Set ===");
    Metrics::ClassificationMetrics metrics = Metrics::evaluate(net, testData);

    Console::value("Accuracy ", metrics.accuracy);
    Console::value("Precision", metrics.precision);
    Console::value("Recall   ", metrics.recall);
    Console::value("F1-score ", metrics.f1);

     // 5. Сравнение с линейной моделью из первой лабораторной
    Console::info("Comparison with Linear Classifier (y = k*x + b)");
    float linearAcc = Evaluation::accuracyLinear(testData, k, b);
    Console::value("Linear model accuracy", linearAcc);
    Console::value("Neural Network accuracy", metrics.accuracy);

    // 6. Сохранение предсказаний в CSV
    Metrics::savePredictions(net, allData, "../predictions.csv");

     // 7. Вывод матрицы ошибок
    Metrics::printConfusionMatrix(net, testData);

     // 8. Демонстрация работы сети на сетке точек
    Metrics::predictOnGrid(net, k, b, 8);

    Console::info("Laboratory work 2 completed successfully!");
    return 0;
}
