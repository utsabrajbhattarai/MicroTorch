#include "raylib.h"
#include "csv_loader.hpp"
#include <iostream>
#include <cmath>

std::vector<int> get_neighbors(int node_id, const std::vector<Edge>& edges) { //Find all nodes connected to node_id
    std::vector<int> neighbors;
    for (const auto& e : edges) {
        if (e.src_node_id == node_id) neighbors.push_back(e.dst_node_id);
        else if (e.dst_node_id == node_id) neighbors.push_back(e.src_node_id);
    }
    return neighbors;
}

void draw_metrics(const Metrics& m) { //Draw the metrics on the left side of the window
    int x = 20, y = 550;
    DrawText("METRICS", x, y, 16, BLACK);
    y += 25;
    DrawText(TextFormat("Precision: %.3f", m.precision), x, y, 12, BLACK); y += 20;
    DrawText(TextFormat("Recall:    %.3f", m.recall), x, y, 12, BLACK); y += 20;
    DrawText(TextFormat("F1:        %.3f", m.f1), x, y, 12, BLACK); y += 20;
    DrawText(TextFormat("Accuracy:  %.3f", m.accuracy), x, y, 12, BLACK); y += 20;
    DrawText(TextFormat("AUROC:     %.3f", m.auroc), x, y, 12, BLACK); y += 20;
    DrawText(TextFormat("TP:%d FP:%d TN:%d FN:%d", m.TP, m.FP, m.TN, m.FN), x, y, 12, BLACK);
}

int draw_ranked_accounts(const std::vector<Account>& accounts, 
                          const std::vector<Node>& nodes,
                          int selected_account_id) { //Draw the ranked accounts on the left side of the window, return the selected account_id
    int x = 20, y = 150;
    int row_height = 30;
    int row_width = 400;

    for (const auto& acc : accounts) { //Draw each account as a row in the table
        Rectangle row = { (float)x, (float)y, (float)row_width, (float)row_height };

        bool hovered = CheckCollisionPointRec(GetMousePosition(), row);
        bool clicked = hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

        Color bg = (acc.account_id == selected_account_id) ? SKYBLUE
                   : hovered ? LIGHTGRAY : RAYWHITE;
        DrawRectangleRec(row, bg);
        DrawRectangleLinesEx(row, 1, GRAY);
        DrawText(TextFormat("#%d  acct:%lld  risk:%.2f", acc.rank, acc.account_id, acc.risk_score),
                 x + 5, y + 7, 12, BLACK);

        if (clicked) { //If the row is clicked, update the selected account_id
            selected_account_id = acc.account_id;
        }

        y += row_height;
    }

    return selected_account_id;
}

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

    long long selected_account_id = accounts[0].account_id; //default: top-ranked account

    while (!WindowShouldClose()) { //Main game loop
        BeginDrawing(); //Setup canvas to start drawing
        ClearBackground(RAYWHITE); //background color

        selected_account_id = draw_ranked_accounts(accounts, nodes, selected_account_id); //draw table, get updated selection
        draw_metrics(metrics); //draw metrics on the left side of the window
        //find which node_id corresponds to the selected account
        int focal_node_id = 0;
        for (const auto& n : nodes) {
            if (n.account_id == selected_account_id) {
                focal_node_id = n.node_id;
                break;
            }
        }

        //circular subgraph layout: focal node at center, neighbors arranged around it
        float cx = 800, cy = 360; //center of subgraph area (right side of window)
        float radius = 200;

        auto neighbors = get_neighbors(focal_node_id, edges); //get all neighbors of the focal node
        int k = neighbors.size();

        //compute each neighbor's position first, so we can draw edges before nodes
        std::vector<Vector2> neighbor_pos;
        for (int i = 0; i < k; ++i) {
            float angle = 2 * PI * i / k;
            float x = cx + radius * std::cos(angle);
            float y = cy + radius * std::sin(angle);
            neighbor_pos.push_back({x, y});
        }

        //draw edges first
        for (int i = 0; i < k; ++i) {
            DrawLine(cx, cy, neighbor_pos[i].x, neighbor_pos[i].y, GRAY);
        }

        //draw focal node
        DrawCircle(cx, cy, 25, BLUE);
        DrawText(TextFormat("%lld", nodes[focal_node_id].account_id), cx - 30, cy + 30, 10, BLACK);

        //draw neighbor nodes, colored by prediction
        for (int i = 0; i < k; ++i) {
            int nid = neighbors[i];
            Color c = (nodes[nid].pred_label == 1) ? RED : GREEN; //red = predicted illicit, green = predicted licit
            DrawCircle(neighbor_pos[i].x, neighbor_pos[i].y, 20, c);
            DrawText(TextFormat("%lld", nodes[nid].account_id),
                     neighbor_pos[i].x - 30, neighbor_pos[i].y + 25, 10, BLACK);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}