#include "raylib.h"

int main() {
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