#pragma once
#include "types.h"

/**
 * Шаблонная структура Vec2 представляет двумерный вектор (точку) с координатами x и y.
 * Шаблонный параметр T позволяет хранить координаты любого арифметического типа,
 * например float, double, int.
 */
template<typename T>
struct Vec2 {
    T x, y;
};

/**
 * Linear содержит набор статических методов для выполнения линейных
 * вычислений, таких как преобразование вектора матрицей, вычисление значения
 * прямой и модуля разности. Все методы шаблонизированы для поддержки различных
 * типов данных (float, double и т.д.).
 */
namespace Linear {

    // Выполняет линейное преобразование двумерного вектора с помощью матрицы 2x2.
    template<Arithmetic T>
    static Vec2<T> transform(T matrix[2][2], T x, T y) {
        return {
            matrix[0][0] * x + matrix[0][1] * y,
            matrix[1][0] * x + matrix[1][1] * y
        };
    }

    // Вычисляет значение прямой y = k * x + b и возвращает |y|
    template<Arithmetic T>
    static T line(T x, T k, T b) {
        T y = k * x + b;
        return y < T(0) ? -y : y;
    }

    // Вычисляет абсолютную разницу между двумя числами: |a - b|
    template<Arithmetic T>
    static T absDiff(T a, T b) {
        T d = a - b;
        return d < T(0) ? -d : d;
    }
};