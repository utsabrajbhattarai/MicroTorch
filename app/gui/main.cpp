#include "raylib.h"
#include "csv_loader.hpp"
#include <iostream>

int main() {
    auto nodes = load_nodes_csv("gui_artifacts/nodes.csv"); //Load nodes from CSV file
    auto edges = load_edges_csv("gui_artifacts/edges.csv"); //Load edges from CSV file
    auto accounts = load_accounts_csv("gui_artifacts/accounts.csv"); //Load accounts from CSV file
    auto metrics = load_metrics_csv("gui_artifacts/metrics.csv"); //Load metrics from CSV file

    std::cout << "Loaded " << nodes.size() << " nodes, " //check if data is loaded correctly
          << edges.size() << " edges, "
          << accounts.size() << " accounts\n";
    std::cout << "Metrics: precision=" << metrics.precision
          << " recall=" << metrics.recall
          << " auroc=" << metrics.auroc << "\n";
    std::cout << "Loaded " << nodes.size() << " nodes\n"; //Print the number of nodes loaded

    InitWindow(1280, 720, "MicroTorch GNN Dashboard"); //Initialize window and OpenGL context
    SetTargetFPS(60); //Set target FPS

    while (!WindowShouldClose()) { //Main game loop
        BeginDrawing(); //Setup canvas to start drawing
        ClearBackground(RAYWHITE); //background color

        //temporary: draw each node as a circle in a simple grid, just to see the data
        int x = 100; 
        int y = 100;
        for (size_t i = 0; i < nodes.size(); ++i) { 
            Color c = (nodes[i].pred_label == 1) ? RED : GREEN; 
            DrawCircle(x, y, 20, c); //Draw a color-filled circle representing the node
            DrawText(TextFormat("%lld", nodes[i].account_id), x - 30, y + 25, 10, BLACK);

            x += 150;
            if (x > 1100) { //If we've reached the end of the row
                x = 100; //Reset x to the start of the row
                y += 150; //Move down to the next row
            }
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}