#include "raylib.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>

//enumerator for dashboard options: MENU or VIEWER:
enum Screen { MENU, VIEWER };
Screen screen = MENU;    //start on the menu by default

struct Shape {
    std::string name;      //display label for eg. "Spiral"
    std::string file;      //basically the path "frames/frames_simple_spiral.csv"
};
//prenaming the shapes structure
std::vector<Shape> shapes = {
    { "Spiral",    "frames/frames_simple_spiral.csv" },
    { "Two Moons", "frames/frames_two_moons.csv" },
    { "Heart",     "frames/frames_heart.csv" },
    { "Infinity",  "frames/frames_infinity.csv" },
};

//basically a function to load frames.csv into a vector of vectors of Vector2 frames[n] mean's nth list of points and frames[n][m] means mth point of that frame  
std::vector<std::vector<Vector2>> load_frames(const std::string& path);

// math-space point -> screen pixels. NOTE the minus on y (screen y grows downward)
inline Vector2 to_screen(float x, float y, float cx, float cy, float scale) {
    return { cx + x * scale, cy - y * scale };
}

//compute the auto-scale for a given set of frames (each shape has a different range)
float compute_scale(const std::vector<std::vector<Vector2>>& frames) {
    float max_range = 1.0f;
    for (const auto& frame : frames)
        for (const Vector2& p : frame) {
            max_range = std::max(max_range, std::abs(p.x));
            max_range = std::max(max_range, std::abs(p.y));
        }
    return (450.0f * 0.85f) / max_range;   //not completely 1 to leave some margin
}

int main() {
    //slider position and width:
    float slider_x = 100;     //left most part 
    float slider_y = 840;   //vertical position (near the bottom of 900px window)
    float slider_w = 700;   //track width in pixels
    float slider_value = 0.0f;        //0.0 = first frame (noise), 1.0 = last frame (shape)

    //auto-play constraints
    bool playing = false;      //stores true/false based on playing/not playing atm
    int tick = 0;              //counts screen-frames to control play speed
    const int TICKS_PER_STEP = 2;   //how many animation frame per screen frame?

    InitWindow(900, 900, "MicroTorch Diffusion Viewer");
    SetTargetFPS(60);

    //load each shape's FINAL frame once, just for the menu previews
    //just keeping final shape and discarding all
    std::vector<std::vector<Vector2>> previews;   //previews[i] = final frame of shapes[i]
    for (const Shape& s : shapes) {
        auto f = load_frames(s.file);
        previews.push_back(f.empty() ? std::vector<Vector2>{} : f.back());   //keep only the last frame
    }

    float cx = 450, cy = 450;   //center of window 

    //the currently-loaded shape's frames + its scale (empty until a card is clicked)
    std::vector<std::vector<Vector2>> frames;
    float scale = 1.0f;

    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();   //declared once so BOTH screens can use it

        BeginDrawing();
        ClearBackground(RAYWHITE);

        if (screen == MENU) {
            DrawText("MicroTorch Diffusion — pick a shape", 250, 60, 28, DARKGRAY);

            for (int i = 0; i < (int)shapes.size(); i++) {
                //card in a 2x2 grid (just like react)
                int col = i % 2, row = i / 2;
                Rectangle card = { 200.0f + col * 300, 200.0f + row * 200, 250, 150 };

                bool hover = CheckCollisionPointRec(mouse, card);
                DrawRectangleRec(card, hover ? LIGHTGRAY : RAYWHITE);   //highlight it when hovered
                DrawRectangleLinesEx(card, 2, DARKGRAY);    //boarder 
                DrawText(shapes[i].name.c_str(), card.x + 20, card.y + 15, 24, DARKBLUE);   //label near the top


                //draw a mini preview of the shape's final frame inside the card
                if (!previews[i].empty()) {
                    //finding this preview's range so we can fit it in the card
                    float pmax = 1.0f;
                    for (const Vector2& p : previews[i]) {
                        pmax = std::max(pmax, std::abs(p.x));
                        pmax = std::max(pmax, std::abs(p.y));
                    }
                    float pcx = card.x + card.width / 2;    //card center x
                    float pcy = card.y + card.height / 2;   //card center y
                    float pscale = (card.height * 0.4f) / pmax;   //fit within 40% of card height
                    for (const Vector2& p : previews[i]) {
                        Vector2 s = to_screen(p.x, p.y, pcx, pcy, pscale);
                        DrawCircle(s.x, s.y, 1.5f, Fade(DARKBLUE, 0.5f));
                    }
                }

                if (hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                    frames = load_frames(shapes[i].file);   //load that shape's frames
                    scale = compute_scale(frames);          //re-auto-scale for this shape
                    slider_value = 0.0f;                    //start at noise
                    playing = false;
                    screen = VIEWER;                        //"navigate" to the viewer
                }
            }
        }

        else {
            //in the viewing branch

            //slider input if mouse is dragging within the track's x-range, update value
            Rectangle track = { slider_x, slider_y, slider_w, 8 };   //the bar: x, y, width, height

            //play-pause input
            Rectangle play_btn = { slider_x, slider_y - 50, 80, 30 };   //keeping this just above the slider
            bool play_hovered = CheckCollisionPointRec(mouse, play_btn);    //is mouse over the button?
            if (play_hovered && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) { //if pressed toggle it 
                playing = !playing;   // toggle
            }

            // in the VIEWER branch, somewhere near the top
            Rectangle back_btn = { 20, 20, 80, 30 };
            if (CheckCollisionPointRec(mouse, back_btn) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
                screen = MENU;   // "navigate" back
            DrawRectangleRec(back_btn, LIGHTGRAY);
            DrawText("< Back", back_btn.x + 10, back_btn.y + 7, 18, BLACK);

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

            //auto-advance when playing
            if (playing) {
                tick++;
                if (tick >= TICKS_PER_STEP) { //just a fps controlling mechanism 
                    tick = 0;
                    slider_value += 1.0f / (frames.size() - 1);   //step forward by one frame's worth
                    if (slider_value >= 1.0f) { //if final frame reached
                        slider_value = 1.0f;
                        playing = false;   //stopping at the end 
                    }
                }
            }

            //based on slider_value 0-1 show the frame
            int show = static_cast<int>(slider_value * (frames.size() - 1));

            //faintly drawing the final shape outline/overlay
            for (const Vector2& p : frames.back()) {
                Vector2 s = to_screen(p.x, p.y, cx, cy, scale);
                DrawCircle(s.x, s.y, 2.0f, Fade(GRAY, 0.15f));   //faint, slightly smaller
            }

            for (const Vector2& p : frames[show]) {
                Vector2 s = to_screen(p.x, p.y, cx, cy, scale);
                DrawCircle(s.x, s.y, 2.5f, BLUE); //drawing each point as circles
            }

            const char* label = TextFormat("frame %d / %d", show, (int)frames.size() - 1);
            int tw = MeasureText(label, 20);    //pixel width of the text at size 20
            DrawText(label, 900 - tw - 20, 20, 20, DARKGRAY);   //20px margin from the right edge

            DrawRectangleRec(play_btn, LIGHTGRAY);  //the play/pause button
            DrawText(playing ? "Pause" : "Play", play_btn.x + 15, play_btn.y + 7, 18, BLACK);   //to display the state/toggle option

            DrawRectangleRec(track, LIGHTGRAY);     //the track bar
            float handle_x = slider_x + slider_value * slider_w;        //handle position from value
            DrawCircle(static_cast<int>(handle_x), static_cast<int>(slider_y + 4), 10, DARKGREEN);        //the draggable knob
        }

        EndDrawing();
    }
    CloseWindow();
    return 0;
}

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