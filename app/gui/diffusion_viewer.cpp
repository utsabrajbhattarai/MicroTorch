#include "raylib.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

//basically a function to load frames.csv into a vector of vectors of Vector2 frames[n] mean's nth list of points and frames[n][m] means mth point of that frame  
std::vector<std::vector<Vector2>> load_frames(const std::string& path) {
    std::vector<std::vector<Vector2>> frames;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string a, b, c;
        std::getline(ss, a, ',');
        std::getline(ss, b, ',');
        std::getline(ss, c, ',');
        int f = std::stoi(a);
        float x = std::stof(b);
        float y = std::stof(c);
        if (f >= (int)frames.size()) frames.resize(f + 1);   //grow as new frames appear
        frames[f].push_back({ x, y });
    }
    return frames;
}

// math-space point -> screen pixels. NOTE the minus on y (screen y grows downward)
Vector2 to_screen(float x, float y, float cx, float cy, float scale) {
    return { cx + x * scale, cy - y * scale };
}

int main() {
    auto frames = load_frames("frames.csv");
    std::cout << "loaded " << frames.size() << " frames, "
              << (frames.empty() ? 0 : frames[0].size()) << " points each\n";

    InitWindow(900, 900, "MicroTorch Diffusion Viewer");
    SetTargetFPS(60);

    float cx = 450, cy = 450;   //center of window 

    // after loading frames, finding the max coordinate to find proper scaling
    float max_range = 1.0f;
    for (const auto& frame : frames)
        for (const Vector2& p : frame) {
            max_range = std::max(max_range, std::abs(p.x));
            max_range = std::max(max_range, std::abs(p.y));
        }
    float scale = (450.0f * 0.9f) / max_range;   //0.9 for leaving a  margin

    int show = static_cast<int>(frames.size() - 1);        //STEP 1: just show the LAST frame (the shape)

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (const Vector2& p : frames[show]) {
            Vector2 s = to_screen(p.x, p.y, cx, cy, scale);
            DrawCircle(s.x, s.y, 2.5f, BLUE);
        }

        DrawText(TextFormat("frame %d / %d", show, (int)frames.size() - 1), 20, 20, 20, DARKGRAY);

        EndDrawing();
    }
    CloseWindow();
    return 0;
}