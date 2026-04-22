#pragma once
#include <vector>
#include <fstream>
#include <cstdlib>
#include "types.h"
#include <algorithm>      
#include <random>
#include "linear.h"

namespace Dataset {

    //шаблонный метод генерации набора случайных точек.
    
       // T - тип чисел для коэффициентов прямой(float, double и т.д.)
       //n - количество генерируемых точек.
       // k - угловой коэффициент разделяющей прямой(тип T).
        //b - свободный член разделяющей прямой(тип T).
        //return вектор структур Point2D, содержащий сгенерированные точки.

    template<Arithmetic T>
    static std::vector<Point2D> generate(int n, T k, T b) {
        std::vector<Point2D> data;
        data.reserve(n); // резервируем память для эффективности

        for (int i = 0; i < n; i++) {

            // генерация случайных координат в диапазоне [-2, 2]
            float x = -2 + (float)rand() / RAND_MAX * 4;
            float y = -2 + (float)rand() / RAND_MAX * 4;
            // определение метки: приводим k и b к float для сравнения
            float label = (y > static_cast<float>(k) * x + static_cast<float>(b)) ? 1.0f : 0.0f;
            data.push_back({ x, y, label });
        }

        return data;
    }

    static bool saveCSV(const std::vector<Point2D>& data, const char* filename) {

        //охраняет набор точек в CSV-файл с заголовком "x,y,label".
       
         // data - вектор точек, которые нужно сохранить.
         // filename - имя файла(путь) для сохранения.
         ///return true, если файл успешно открыт и запись выполнена, иначе false.
        
        std::ofstream file(filename);
        if (!file) return false;

        file << "x,y,label\n";
        // Проходим по всем точкам в контейнере data
        for (auto& p : data)
            // Для каждой точки записываем строку с координатами и меткой через запятую
            file << p.x << "," << p.y << "," << p.label << "\n";
        return true;
    }
    // Разделяет датасет на обучающую и тестовую выборки
    // trainRatio = 0.8  →  80% train, 20% test
    static std::pair<std::vector<Point2D>, std::vector<Point2D>>
        split(const std::vector<Point2D>& data, float trainRatio = 0.8f) {

        std::vector<Point2D> train, test;
        train.reserve(static_cast<size_t>(data.size() * trainRatio));
        test.reserve(data.size() - train.capacity());

        // Создаём индексы и перемешиваем их (чтобы данные были в случайном порядке)
        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < data.size(); ++i) indices[i] = i;

        // Фиксированный seed для воспроизводимости (42)
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t train_size = static_cast<size_t>(data.size() * trainRatio);

        for (size_t i = 0; i < data.size(); ++i) {
            if (i < train_size) {
                train.push_back(data[indices[i]]);
            }
            else {
                test.push_back(data[indices[i]]);
            }
        }

        return { train, test };
    }

} // namespace Dataset
