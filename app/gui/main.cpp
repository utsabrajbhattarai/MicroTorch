#include "raylib.h"
#include "csv_loader.hpp"
#include <iostream>

int main() {
    auto nodes = load_nodes_csv("gui_artifacts/nodes.csv");
    std::cout << "Loaded " << nodes.size() << " nodes\n";
    for (const auto& n : nodes) {
        std::cout << n.node_id << " " << n.account_id << " " << n.pred_prob << "\n";
    }

    InitWindow(1280, 720, "MicroTorch GNN Dashboard");
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText("Hello, raylib!", 20, 20, 20, BLACK);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}