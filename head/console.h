#pragma once
#include <iostream>

namespace Console {

    // Вывод информационного сообщения
    static void info(const char* msg) {
        std::cout << "[INFO] " << msg << "\n";
    }

    // Шаблонная версия — работает с любым типом
    template<typename T>
    static void value(const char* name, T val) {
        std::cout << name << ": " << val << "\n";
    }
};
