#pragma once
#include <vector>
#include "neuralnet.h"
#include "types.h"
#include "console.h"
#include <algorithm>

namespace Neural {

    namespace Trainer {

        /**
         * ‘ункци€ потерь: бинарна€ кросс-энтропи€ (Binary Cross-Entropy).
         * »спользуетс€ дл€ задач бинарной классификации.
         *
         * y_true Ч истинна€ метка (0 или 1)
         * y_pred Ч предсказанна€ веро€тность (от 0 до 1)
         */
        template<Arithmetic T>
        T binaryCrossEntropy(T y_true, T y_pred) {
            const T eps = T(1e-7); // защита от log(0)
            y_pred = std::max(eps, std::min(T(1) - eps, y_pred));
            return -(y_true * std::log(y_pred) + (T(1) - y_true) * std::log(T(1) - y_pred));
        }

        /**
         * ќсновной метод обучени€ нейронной сети.
         * –еализует алгоритм обратного распространени€ ошибки (backpropagation).
         *
         * net Ч нейронна€ сеть
         * data Ч обучающий датасет
         * epochs Ч количество эпох обучени€
         * learningRate Ч скорость обучени€
         */
        template<Arithmetic T>
        void train(NeuralNetwork<T>& net,
            const std::vector<Point2D>& data,
            int epochs,
            T learningRate)
        {
            int n = (int)data.size();

            // цикл по эпохам
            for (int epoch = 0; epoch < epochs; ++epoch) {
                T totalLoss = 0;

                // проход по всем объектам датасета
                for (const auto& p : data) {

                    // === FORWARD PASS ===
                    T y_pred = net.forward(p);   // предсказание сети
                    T y_true = T(p.label);       // истинна€ метка
                    totalLoss += binaryCrossEntropy(y_true, y_pred);

                    int H = net.getHiddenSize();
                    const auto& a_h = net.getHiddenA(); // активации скрытого сло€

                    // === BACKPROPAGATION ===
                    // градиент по выходному слою
                    T dz_o = y_pred - y_true;

                    // градиенты скрытого сло€
                    std::vector<T> dz_h(H);
                    for (int i = 0; i < H; ++i) {
                        T da = a_h[i] * (T(1) - a_h[i]); // производна€ сигмоиды
                        dz_h[i] = dz_o * net.getWeightsHO()[i] * da;
                    }

                    // === ќЅЌќ¬Ћ≈Ќ»≈ ¬≈—ќ¬ ===

                    // выходной слой
                    auto w_ho = net.getWeightsHO();
                    T b_o = net.getBiasO();

                    for (int i = 0; i < H; ++i)
                        w_ho[i] -= learningRate * dz_o * a_h[i];

                    b_o -= learningRate * dz_o;

                    // скрытый слой
                    auto w_ih = net.getWeightsIH();
                    auto b_h = net.getBiasesH();

                    for (int i = 0; i < H; ++i) {
                        w_ih[2 * i] -= learningRate * dz_h[i] * T(p.x);
                        w_ih[2 * i + 1] -= learningRate * dz_h[i] * T(p.y);
                        b_h[i] -= learningRate * dz_h[i];
                    }

                    // записываем обновлЄнные параметры обратно в сеть
                    net.setWeightsHO(w_ho);
                    net.setBiasO(b_o);
                    net.setWeightsIH(w_ih);
                    net.setBiasesH(b_h);
                }

                // средн€€ ошибка по эпохе
                totalLoss /= n;

                // вывод каждые 20 эпох
                if (epoch % 20 == 0 || epoch == epochs - 1) {
                    Console::value("Epoch", epoch);
                    Console::value("Loss", totalLoss);
                }
            }
        }

        /**
         * ѕодсчЄт точности модели (accuracy).
         *
         * ¬озвращает долю правильно классифицированных объектов.
         */
        template<Arithmetic T>
        T accuracy(NeuralNetwork<T>& net, const std::vector<Point2D>& data) {
            int correct = 0;

            for (const auto& p : data) {
                if (net.predictClass(p) == (int)p.label)
                    ++correct;
            }

            return T(correct) / data.size();
        }

    } // namespace Trainer
} // namespace Neural// namespace Neural