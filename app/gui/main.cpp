#include "raylib.h"
#include "csv_loader.hpp"
#include <iostream>

int main() {
    auto nodes = load_nodes_csv("gui_artifacts/nodes.csv");
    std::cout << "Loaded " << nodes.size() << " nodes\n";

    InitWindow(1280, 720, "MicroTorch GNN Dashboard");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // temporary: draw each node as a circle in a simple grid, just to see the data
        int x = 100;
        int y = 100;
        for (size_t i = 0; i < nodes.size(); ++i) {
            Color c = (nodes[i].pred_label == 1) ? RED : GREEN;
            DrawCircle(x, y, 20, c);
            DrawText(TextFormat("%lld", nodes[i].account_id), x - 30, y + 25, 10, BLACK);

            x += 150;
            if (x > 1100) {
                x = 100;
                y += 150;
            }
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}