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

    //slider position and width:
    float slider_x = 100;     //left most part 
    float slider_y = 830;   //vertical position (near the bottom of 900px window)
    float slider_w = 700;   //track width in pixels
    float slider_value = 1.0f;        //0.0 = first frame (noise), 1.0 = last frame (shape)

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


    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        //slider input if mouse is dragging within the track's x-range, update value
        Rectangle track = { slider_x, slider_y, slider_w, 8 };   //the bar: x, y, width, height
        Vector2 mouse = GetMousePosition();

        //are we holding the mouse button down anywhere near the track?
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON) &&
            mouse.x >= slider_x - 10 && mouse.x <= slider_x + slider_w + 10 &&
            mouse.y >= slider_y - 20 && mouse.y <= slider_y + 20)//note that also conditioning for nearer so thats why range of +-10/20
            {
            //map the mouse's x position to a 0..1 value along the track
            slider_value = (mouse.x - slider_x) / slider_w;
            if (slider_value < 0.0f) slider_value = 0.0f;   //clamping  so it can't go past the ends
            if (slider_value > 1.0f) slider_value = 1.0f;
        }

        //based on slider_value 0-1 show the frame
        int show = static_cast<int>(slider_value * (frames.size() - 1));

        for (const Vector2& p : frames[show]) {
            Vector2 s = to_screen(p.x, p.y, cx, cy, scale);
            DrawCircle(s.x, s.y, 2.5f, BLUE);
        }

        DrawText(TextFormat("frame %d / %d", show, (int)frames.size() - 1), 20, 20, 20, DARKGRAY);

        DrawRectangleRec(track, LIGHTGRAY);                                  //the track bar
        float handle_x = slider_x + slider_value * slider_w;                 //handle position from value
        DrawCircle(static_cast<int>(handle_x), static_cast<int>(slider_y + 4), 10, DARKGREEN);        //the draggable knob


        EndDrawing();
    }
    CloseWindow();
    return 0;
}