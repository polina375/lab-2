#pragma once
#include <vector>
#include <cmath>
#include "types.h"      // Здесь находятся Arithmetic и Point2D
#include "console.h"

namespace Neural {

    // Шаблонный класс простой нейронной сети для бинарной классификации
    // Архитектура: 2 входа → скрытый слой (hiddenSize нейронов) → 1 выход (сигмоида)
    template<Arithmetic T>
    class NeuralNetwork {
    public:
        // Конструктор
        // hiddenSize - количество нейронов в скрытом слое (по умолчанию 4)
        explicit NeuralNetwork(int hiddenSize = 4);

        // Прямой проход (forward pass)
        // Принимает точку и возвращает вероятность класса 1 (значение от 0 до 1)
        T forward(const Point2D& p);

        // Предсказание класса
        // Возвращает 0 или 1 в зависимости от того, превышает ли вероятность порог
        int predictClass(const Point2D& p, T threshold = T(0.5)) const;

        // ===================== Геттеры и сеттеры =====================
        // Эти методы нужны Trainer'у для обучения (backpropagation)

        int getHiddenSize() const { return hiddenSize; }

        // Возвращает активации скрытого слоя после последнего forward
        const std::vector<T>& getHiddenA() const { return a_h; }

        // Геттеры весов и смещений (используются при обновлении весов)
        const std::vector<T>& getWeightsHO() const { return w_ho; }
        const std::vector<T>& getWeightsIH() const { return w_ih; }
        const std::vector<T>& getBiasesH() const { return b_h; }
        T getBiasO() const { return b_o; }

        // Сеттеры — позволяют Trainer обновлять веса после вычисления градиентов
        void setWeightsHO(const std::vector<T>& w) { w_ho = w; }
        void setWeightsIH(const std::vector<T>& w) { w_ih = w; }
        void setBiasesH(const std::vector<T>& b) { b_h = b; }
        void setBiasO(T b) { b_o = b; }

    private:
        // Инициализация всех весов и смещений случайными значениями
        void initWeights();

        // ===================== Основные параметры сети =====================
        int hiddenSize;                 // количество нейронов в скрытом слое

        std::vector<T> w_ih;            // веса от входа к скрытому слою (2 * hiddenSize)
        std::vector<T> b_h;             // смещения скрытого слоя
        std::vector<T> w_ho;            // веса от скрытого слоя к выходу
        T b_o;                          // смещение выходного нейрона

        // ===================== Кэш для forward pass =====================
        // Эти поля заполняются при вызове forward() и используются в обучении
        std::vector<T> z_h;             // взвешенная сумма (до активации) на скрытом слое
        std::vector<T> a_h;             // активации (после сигмоиды) скрытого слоя
        T z_o = 0;                      // взвешенная сумма на выходном слое
        T a_o = 0;                      // финальная активация (вероятность)

        // Функция активации — сигмоида
        T sigmoid(T x) const {
            return T(1) / (T(1) + std::exp(-x));
        }
    };

    // ===================== Реализация методов =====================

    // Конструктор
    template<Arithmetic T>
    NeuralNetwork<T>::NeuralNetwork(int hs) : hiddenSize(hs) {
        w_ih.resize(2 * hs);
        b_h.resize(hs);
        w_ho.resize(hs);
        initWeights();
    }

    // Инициализация весов случайными значениями в диапазоне [-0.5, 0.5]
    template<Arithmetic T>
    void NeuralNetwork<T>::initWeights() {
        for (auto& w : w_ih) w = T(rand()) / RAND_MAX - T(0.5);
        for (auto& b : b_h)  b = T(rand()) / RAND_MAX - T(0.5);
        for (auto& w : w_ho) w = T(rand()) / RAND_MAX - T(0.5);
        b_o = T(rand()) / RAND_MAX - T(0.5);
    }

    // Прямой проход через сеть
    template<Arithmetic T>
    T NeuralNetwork<T>::forward(const Point2D& p) {
        z_h.resize(hiddenSize);
        a_h.resize(hiddenSize);

        // === Скрытый слой ===
        for (int i = 0; i < hiddenSize; ++i) {
            // Вычисляем взвешенную сумму + смещение
            z_h[i] = w_ih[2 * i] * T(p.x) + w_ih[2 * i + 1] * T(p.y) + b_h[i];
            // Применяем сигмоиду
            a_h[i] = sigmoid(z_h[i]);
        }

        // === Выходной слой ===
        z_o = b_o;                                   // начинаем со смещения
        for (int i = 0; i < hiddenSize; ++i) {
            z_o += w_ho[i] * a_h[i];                 // суммируем взвешенные активации
        }

        a_o = sigmoid(z_o);                          // финальная сигмоида
        return a_o;
    }

    // Предсказание класса (0 или 1)
    template<Arithmetic T>
    int NeuralNetwork<T>::predictClass(const Point2D& p, T threshold) const {
        // Временный вектор для активаций скрытого слоя (не сохраняем в кэш)
        std::vector<T> a(hiddenSize);

        // Считаем скрытый слой
        for (int i = 0; i < hiddenSize; ++i) {
            T z = w_ih[2 * i] * T(p.x) + w_ih[2 * i + 1] * T(p.y) + b_h[i];
            a[i] = sigmoid(z);
        }

        // Считаем выход
        T z = b_o;
        for (int i = 0; i < hiddenSize; ++i) {
            z += w_ho[i] * a[i];
        }

        // Применяем порог
        return sigmoid(z) >= threshold ? 1 : 0;
    }

} // namespace Neural