#pragma once
#include <type_traits>

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

struct Point2D {
    float x;
    float y;
    float label;
};