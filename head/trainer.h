#pragma once
#include <vector>
#include "neuralnet.h"
#include "types.h"
#include "console.h"

namespace Neural {

    namespace Trainer {

        template<Arithmetic T>
        T binaryCrossEntropy(T y_true, T y_pred) {
            const T eps = T(1e-7);
            y_pred = std::max(eps, std::min(T(1) - eps, y_pred));
            return -(y_true * std::log(y_pred) + (T(1) - y_true) * std::log(T(1) - y_pred));
        }

        template<Arithmetic T>
        void train(NeuralNetwork<T>& net,
            const std::vector<Point2D>& data,
            int epochs,
            T learningRate)
        {
            int n = (int)data.size();

            for (int epoch = 0; epoch < epochs; ++epoch) {
                T totalLoss = 0;

                for (const auto& p : data) {
                    T y_pred = net.forward(p);
                    T y_true = T(p.label);
                    totalLoss += binaryCrossEntropy(y_true, y_pred);

                    int H = net.getHiddenSize();
                    const auto& a_h = net.getHiddenA();

                    T dz_o = y_pred - y_true;

                    // градиенты скрытого слоя
                    std::vector<T> dz_h(H);
                    for (int i = 0; i < H; ++i) {
                        T da = a_h[i] * (T(1) - a_h[i]);
                        dz_h[i] = dz_o * net.getWeightsHO()[i] * da;
                    }

                    // обновляем выходной слой
                    auto w_ho = net.getWeightsHO();
                    T b_o = net.getBiasO();
                    for (int i = 0; i < H; ++i)
                        w_ho[i] -= learningRate * dz_o * a_h[i];
                    b_o -= learningRate * dz_o;

                    // обновляем скрытый слой
                    auto w_ih = net.getWeightsIH();
                    auto b_h = net.getBiasesH();
                    for (int i = 0; i < H; ++i) {
                        w_ih[2 * i] -= learningRate * dz_h[i] * T(p.x);
                        w_ih[2 * i + 1] -= learningRate * dz_h[i] * T(p.y);
                        b_h[i] -= learningRate * dz_h[i];
                    }

                    net.setWeightsHO(w_ho);
                    net.setBiasO(b_o);
                    net.setWeightsIH(w_ih);
                    net.setBiasesH(b_h);
                }

                totalLoss /= n;
                if (epoch % 20 == 0 || epoch == epochs - 1) {
                    Console::value("Epoch", (float)epoch);
                    Console::value("Loss", (float)totalLoss);
                }
            }
        }

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
} // namespace N
