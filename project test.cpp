#include "types.h"
#include "linear.h"
#include "dataset.h"
#include "evaluation.h"
#include "console.h"


#include <vector>

void test_dataset() {
    auto data = Dataset::generate(10, 0.5f, 1.0f);

    Console::info("Test: Dataset size");
    Console::value("Size", data.size());

    if (data.size() != 10)
        throw "Dataset test failed";
}

void test_linear() {
    float matrix[2][2] = {
        {1.0f, 0.1f},
        {0.0f, 1.0f}
    };

    auto data = Dataset::generate(1, 0.0f, 0.0f);

    float x = data[0].x;
    float y = data[0].y;

    auto result = Linear::transform(matrix, x, y);

    Console::info("Test: Linear transform");
    Console::value("Old X", x);
    Console::value("New X", result.x);

    if (result.x == x && result.y == y)
        throw "Linear test failed";
}

void test_evaluation() {
    auto data = Dataset::generate(5, 1.0f, 0.0f);

    float err = Evaluation::meanAbsError(data, 1.0f, 0.0f);

    Console::info("Test: Evaluation");
    Console::value("Error", err);

    if (err < 0)
        throw "Evaluation test failed";
}

void test_csv() {
    auto data = Dataset::generate(5, 0.5f, 1.0f);

    bool ok = Dataset::saveCSV(data, "test_points.csv");

    Console::info("Test: CSV");
    Console::value("Saved", ok);

    if (!ok)
        throw "CSV test failed";
}

int main() {
    Console::info("Running tests...");

    bool all_ok = true;

    try { test_dataset(); }
    catch (...) { Console::info("Dataset failed"); all_ok = false; }

    try { test_linear(); }
    catch (...) { Console::info("Linear failed"); all_ok = false; }

    try { test_evaluation(); }
    catch (...) { Console::info("Evaluation failed"); all_ok = false; }

    try { test_csv(); }
    catch (...) { Console::info("CSV failed"); all_ok = false; }

    if (all_ok)
        Console::info("ALL TESTS PASSED");
    else
        Console::info("SOME TESTS FAILED");

    return all_ok ? 0 : 1;
}
