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

void draw_metrics(const Metrics& m, Font font) { //Draw the metrics on the left side of the window
    int x = 20, y = 550;
    DrawTextEx(font, "METRICS", {(float)x, (float)y}, 24, 1, DARKBLUE);
    y += 25;
    //added font parameter to DrawTextEx for better text rendering
    DrawTextEx(font, TextFormat("Precision: %.3f", m.precision), {(float)x, (float)y}, 18, 1, BLACK); y += 20;
    DrawTextEx(font, TextFormat("Recall:    %.3f", m.recall), {(float)x, (float)y}, 18, 1, BLACK); y += 20;
    DrawTextEx(font, TextFormat("F1:        %.3f", m.f1), {(float)x, (float)y}, 18, 1, BLACK); y += 20;
    DrawTextEx(font, TextFormat("Accuracy:  %.3f", m.accuracy), {(float)x, (float)y}, 18, 1, BLACK); y += 20;
    DrawTextEx(font, TextFormat("AUROC:     %.3f", m.auroc), {(float)x, (float)y}, 18, 1, BLACK); y += 20;
    DrawTextEx(font, TextFormat("TP:%d FP:%d TN:%d FN:%d", m.TP, m.FP, m.TN, m.FN), {(float)x, (float)y}, 18, 1, BLACK);
}

int draw_ranked_accounts(const std::vector<Account>& accounts, 
                          const std::vector<Node>& nodes,
                          int selected_account_id,
                          Font font,
                          int& accounts_page,      //now takes the page by reference so it can change it
                          int max_rows = 8) { //Draw the ranked accounts table on the left side of the window, return the selected account_id
    int x = 20, y = 220;
    int row_height = 35;
    int row_width = 400;

    int col1 = x + 5, col2 = x + 60, col3 = x + 220; //fixed pixel positions per column

    int total = (int)accounts.size();
    int total_pages = (total + max_rows - 1) / max_rows; //ceiling division
    if (accounts_page >= total_pages) accounts_page = total_pages - 1; //clamp if data shrank
    if (accounts_page < 0) accounts_page = 0; //clamp if negative

    int start = accounts_page * max_rows; //starting index for the current page
    int end = std::min(start + max_rows, total); //ending index for the current page

    for (int i = start; i < end; ++i) { //loop through the accounts for the current page
        const auto& acc = accounts[i];
        Rectangle row = { (float)x, (float)y, (float)row_width, (float)row_height }; //define the rectangle for the current row

        bool hovered = CheckCollisionPointRec(GetMousePosition(), row); //check if the mouse is hovering over the row
        bool clicked = hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON); //check if the row is clicked

        Color bg = (acc.account_id == selected_account_id) ? SKYBLUE //highlight the selected account
                   : hovered ? LIGHTGRAY : RAYWHITE;
        DrawRectangleRec(row, bg); //draw the background of the row
        DrawRectangleLinesEx(row, 1, GRAY); //draw the border of the row

        DrawTextEx(font, TextFormat("#%d", acc.rank), {(float)col1, (float)(y + 9)}, 16, 1, BLACK); //draw the rank of the account
        DrawTextEx(font, TextFormat("%lld", acc.account_id), {(float)col2, (float)(y + 9)}, 16, 1, BLACK); //draw the account_id of the account
        DrawTextEx(font, TextFormat("risk:%.2f", acc.risk_score), {(float)col3, (float)(y + 9)}, 16, 1, BLACK); //draw the risk_score of the account

        if (clicked) { //update the selected account_id if the row is clicked
            selected_account_id = acc.account_id;
        }

        y += row_height;
    }

    //pagination controls: Prev / Next buttons + page indicator
    Rectangle prev_btn = {(float)x, (float)y + 10, 60, 25}; //offset y by 10 pixels for spacing
    Rectangle next_btn = {(float)(x + 70), (float)y + 10, 60, 25}; //offset y by 10 pixels for spacing

    bool prev_hovered = CheckCollisionPointRec(GetMousePosition(), prev_btn); //check if the previous button is hovered
    bool next_hovered = CheckCollisionPointRec(GetMousePosition(), next_btn); //check if the next button is hovered

    DrawRectangleRec(prev_btn, prev_hovered ? LIGHTGRAY : RAYWHITE); //draw the previous button with hover effect
    DrawRectangleLinesEx(prev_btn, 1, GRAY); //draw the border of the previous button
    DrawTextEx(font, "< Prev", {prev_btn.x + 5, prev_btn.y + 4}, 14, 1, BLACK); //draw the text for the previous button

    DrawRectangleRec(next_btn, next_hovered ? LIGHTGRAY : RAYWHITE); //draw the next button with hover effect
    DrawRectangleLinesEx(next_btn, 1, GRAY); //draw the border of the next button
    DrawTextEx(font, "Next >", {next_btn.x + 5, next_btn.y + 4}, 14, 1, BLACK); //draw the text for the next button

    if (prev_hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && accounts_page > 0) //check if the previous button is hovered and clicked, and if the current page is greater than 0
        accounts_page--;
    if (next_hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && accounts_page < total_pages - 1) //check if the next button is hovered and clicked, and if the current page is less than the total pages - 1
        accounts_page++;

    DrawTextEx(font, TextFormat("Page %d / %d", accounts_page + 1, total_pages), //draw the page indicator text
               {(float)(x + 150), (float)y + 15}, 14, 1, DARKGRAY);

    return selected_account_id;
}

void draw_preview_table(const std::vector<Node>& nodes, Font font, int max_rows = 5) {
    int x = 20, y = 20;
    DrawTextEx(font, "DATA PREVIEW", {(float)x, (float)y}, 24, 1, DARKBLUE);
    y += 30;

    int col1 = x, col2 = x + 80, col3 = x + 220, col4 = x + 340;  // fixed pixel positions per column
    DrawTextEx(font, "node_id", {(float)col1, (float)y}, 16, 1, DARKGRAY);
    DrawTextEx(font, "account_id", {(float)col2, (float)y}, 16, 1, DARKGRAY);
    DrawTextEx(font, "pred_prob", {(float)col3, (float)y}, 16, 1, DARKGRAY);
    DrawTextEx(font, "pred_label", {(float)col4, (float)y}, 16, 1, DARKGRAY);
    y += 20;

    int rows_to_show = std::min((int)nodes.size(), max_rows);
    for (int i = 0; i < rows_to_show; ++i) {
        const auto& n = nodes[i];
        DrawTextEx(font, TextFormat("%d", n.node_id), {(float)col1, (float)y}, 16, 1, BLACK);
        DrawTextEx(font, TextFormat("%lld", n.account_id), {(float)col2, (float)y}, 16, 1, BLACK);
        DrawTextEx(font, TextFormat("%.2f", n.pred_prob), {(float)col3, (float)y}, 16, 1, BLACK);
        DrawTextEx(font, TextFormat("%d", n.pred_label), {(float)col4, (float)y}, 16, 1, BLACK);
        y += 20;
    }
}

    Color cream = {252, 249, 240, 255}; //defined a color for background. 
    //You may change the background color by changing the RGBA values.

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

    Font uiFont = LoadFontEx("app/gui/assets/Inter-VariableFont_opsz,wght.ttf", 32, nullptr, 0); //Load font
    SetTextureFilter(uiFont.texture, TEXTURE_FILTER_BILINEAR); //Set texture scaling filter mode

    long long selected_account_id = accounts[0].account_id; //default: top-ranked account
    int accounts_page = 0; //current page of the ranked accounts table, starting at 0

    while (!WindowShouldClose()) { //Main game loop
        BeginDrawing(); //Setup canvas to start drawing
        ClearBackground(cream); //background color

        //panel backgrounds for visual separation
        DrawRectangle(10, 10, 430, 190, LIGHTGRAY);      //preview table backing
        DrawRectangle(10, 210, 430, 8*35+10, Fade(SKYBLUE, 0.08f)); //accounts table backing, matches max_rows=8
        DrawRectangle(10, 540, 430, 170, LIGHTGRAY); //metrics backing

        //vertical divider between left panels and subgraph area
        DrawLine(460, 0, 460, 720, GRAY);

        draw_preview_table(nodes, uiFont); //draw preview table on the left side of the window
        selected_account_id = draw_ranked_accounts(accounts, nodes, selected_account_id, uiFont, accounts_page); //draw table, get updated selection
        draw_metrics(metrics, uiFont); //draw metrics on the left side of the window
    
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
            float x = cx + radius * std::cos(angle); //compute x position of neighbor node
            float y = cy + radius * std::sin(angle); //compute y position of neighbor node
            neighbor_pos.push_back({x, y});
        }

        //draw edges first
        for (int i = 0; i < k; ++i) {
            DrawLine(cx, cy, neighbor_pos[i].x, neighbor_pos[i].y, GRAY);
        }

        //draw focal node
        DrawCircle(cx, cy, 25, BLUE);
        DrawTextEx(uiFont, TextFormat("%lld", nodes[focal_node_id].account_id),
                   {cx - 30, cy + 30}, 16, 1, BLACK);

        //draw neighbor nodes, colored by prediction
        for (int i = 0; i < k; ++i) {
            int nid = neighbors[i];
            Color c = (nodes[nid].pred_label == 1) ? RED : GREEN; //red = predicted illicit, green = predicted licit
            DrawCircle(neighbor_pos[i].x, neighbor_pos[i].y, 20, c);
            DrawTextEx(uiFont, TextFormat("%lld", nodes[nid].account_id),
                       {neighbor_pos[i].x - 30, neighbor_pos[i].y + 25}, 16, 1, BLACK);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}