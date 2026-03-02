#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定義要載入的影像數量
#define NUM_RAW_IMAGES 3
#define NUM_BMP_IMAGES 3
#define TOTAL_IMAGES (NUM_RAW_IMAGES + NUM_BMP_IMAGES)

// 遊戲畫面狀態
typedef enum GameScreen
{
    MAIN_MENU,
    ASSIGNMENT_1
} GameScreen;

typedef struct
{
    int width;
    int height;
    int bpp; // Bytes Per Pixel
    unsigned char *originalData;
    unsigned char *processedData;
    Texture2D texture;
} ImageAsset;

unsigned char *LoadRawImageData(const char *fileName, int width, int height)
{
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (data == NULL)
    {
        perror("Failed to allocate memory for RAW image data");
        return NULL;
    }
    FILE *file = fopen(fileName, "rb");
    if (file == NULL)
    {
        printf("ERROR: Failed to open RAW file: %s\n", fileName);
        free(data);
        return NULL;
    }
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = (y * width) + x;
            size_t read_count = fread(&data[index], sizeof(unsigned char), 1, file);
            if (read_count < 1)
            {
                printf("Warning: Unexpected end of file at pixel (%d, %d)\n", x, y);
                goto cleanup;
            }
        }
    }
cleanup:
    fclose(file);
    return data;
}

unsigned char *LoadBmpImageData(const char *fileName, int *width, int *height)
{
    FILE *file = fopen(fileName, "rb");
    if (file == NULL)
    {
        printf("ERROR: Failed to open BMP file: %s\n", fileName);
        return NULL;
    }

    // --- 1. 讀取並驗證檔案標頭 ---
    unsigned char fileHeader[14];
    if (fread(fileHeader, 1, 14, file) != 14)
    {
        printf("ERROR: Invalid BMP file header for %s\n", fileName);
        fclose(file);
        return NULL;
    }
    if (fileHeader[0] != 'B' || fileHeader[1] != 'M')
    {
        printf("ERROR: Not a valid BMP file: %s\n", fileName);
        fclose(file);
        return NULL;
    }

    // --- 2. 讀取 DIB 標頭 (資訊標頭) ---
    int dataOffset = *(int *)&fileHeader[10];
    unsigned char infoHeader[40];
    if (fread(infoHeader, 1, 40, file) != 40)
    {
        printf("ERROR: Invalid BMP info header for %s\n", fileName);
        fclose(file);
        return NULL;
    }

    *width = *(int *)&infoHeader[4];
    *height = *(int *)&infoHeader[8];
    short bpp = *(short *)&infoHeader[14]; // 每個像素的位元數 (Bits Per Pixel)

    printf("--- Loading '%s': Width=%d, Height=%d, BPP=%d ---\n", fileName, *width, *height);

    int finalImageSize = (*width) * (*height) * 3;
    unsigned char *imageData = (unsigned char *)malloc(finalImageSize);
    if (imageData == NULL)
    {
        perror("Failed to allocate memory for final image data");
        fclose(file);
        return NULL;
    }

    fseek(file, dataOffset, SEEK_SET);

    // ==========================================================
    //  路徑 A：處理 8-bit 索引色 BMP
    // ==========================================================
    if (bpp == 8)
    {
        // 8-bit BMP 需要先讀取顏色表 (Color Palette)
        // 顏色表緊接在 infoHeader 後面，通常有 256 種顏色，每種顏色 4 bytes (B,G,R,A/Reserved)
        Color palette[256] = {0};
        fseek(file, 14 + 40, SEEK_SET); // 跳到 DIB 標頭之後
        fread(palette, sizeof(Color), 256, file);

        // 重新將指標移回像素數據起始點
        fseek(file, dataOffset, SEEK_SET);

        // 8-bit 檔案的 Padding 計算 (每行像素數需為 4 的倍數)
        int rowPadded = (*width + 3) & (~3);
        int padding = rowPadded - *width;

        unsigned char *rowBuffer = (unsigned char *)malloc(rowPadded);
        if (rowBuffer == NULL)
        {
            perror("Failed to allocate memory for row buffer");
            free(imageData);
            fclose(file);
            return NULL;
        }

        for (int y = (*height) - 1; y >= 0; y--)
        {
            // 從檔案讀取一整行包含 padding 的索引數據
            fread(rowBuffer, 1, rowPadded, file);

            for (int x = 0; x < *width; x++)
            {
                // 取得當前像素的顏色表索引
                unsigned char paletteIndex = rowBuffer[x];
                // 從顏色表中查詢對應的 RGB 顏色
                Color c = palette[paletteIndex];
                // 將查到的 RGB 顏色寫入我們最終的 imageData 緩衝區
                int destIndex = (y * (*width) + x) * 3;
                imageData[destIndex + 0] = c.r; // Red
                imageData[destIndex + 1] = c.g; // Green
                imageData[destIndex + 2] = c.b; // Blue
            }
        }
        free(rowBuffer);
    }
    fclose(file);
    return imageData;
}
/**
 * @brief Prints a report of the centered 10x10 pixel values to the console.
 */
void ReportCenteredPixels(Image image, const char *imageName)
{
    printf("---- Centered 10x10 Pixel Values for: %s ----\n", imageName);
    if (image.data == NULL)
    {
        printf("Image data is not available.\n\n");
        return;
    }
    int startX = (image.width - 10) / 2;
    int startY = (image.height - 10) / 2;
    for (int y = 0; y < 10; y++)
    {
        for (int x = 0; x < 10; x++)
        {
            int currentX = startX + x;
            int currentY = startY + y;
            if (image.format == PIXELFORMAT_UNCOMPRESSED_GRAYSCALE)
            {
                unsigned char *data = (unsigned char *)image.data;
                unsigned char grayValue = data[currentY * image.width + currentX];
                printf("%3d ", grayValue);
            }
            else
            {
                Color pixel = GetImageColor(image, currentX, currentY);
                printf("%3d ", pixel.r);
            }
        }
        printf("\n");
    }
    printf("----------------------------------------------------------\n\n");
}

void apply_image_negative(const unsigned char *input, unsigned char *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = 255 - input[i];
    }
}

void apply_log_transform(const unsigned char *input, unsigned char *output, int size)
{
    // c = 255 / log(1 + 255)
    float c = 255.0f / logf(1.0f + 255.0f);
    for (int i = 0; i < size; i++)
    {
        output[i] = (unsigned char)(c * logf(1.0f + input[i]));
    }
}

void apply_gamma_transform(const unsigned char *input, unsigned char *output, int size, float gamma)
{
    for (int i = 0; i < size; i++)
    {
        // 1. 將像素值正規化到 [0, 1]
        float normalized_input = input[i] / 255.0f;
        // 2. 套用 gamma
        float corrected = powf(normalized_input, gamma);
        // 3. 轉換回 [0, 255]
        output[i] = (unsigned char)(corrected * 255.0f);
    }
}

/**
 * @brief 使用最近鄰插值演算法進行影像縮放。
 * @return 回傳一個包含新尺寸像素數據的記憶體區塊，需要手動 free。
 */
unsigned char *resample_nearest_neighbor(const unsigned char *srcData, int srcW, int srcH, int destW, int destH, int bpp)
{
    unsigned char *destData = (unsigned char *)malloc(destW * destH * bpp);
    if (!destData)
        return NULL;

    float scaleX = (float)srcW / destW;
    float scaleY = (float)srcH / destH;

    for (int dy = 0; dy < destH; dy++)
    {
        for (int dx = 0; dx < destW; dx++)
        {
            int sx = (int)(dx * scaleX);
            int sy = (int)(dy * scaleY);

            // 邊界檢查
            if (sx >= srcW)
                sx = srcW - 1;
            if (sy >= srcH)
                sy = srcH - 1;

            int srcIndex = (sy * srcW + sx) * bpp;
            int destIndex = (dy * destW + dx) * bpp;

            memcpy(&destData[destIndex], &srcData[srcIndex], bpp);
        }
    }
    return destData;
}

/**
 * @brief 使用雙線性插值演算法進行影像縮放。
 * @return 回傳一個包含新尺寸像素數據的記憶體區塊，需要手動 free。
 */
unsigned char *resample_bilinear(const unsigned char *srcData, int srcW, int srcH, int destW, int destH, int bpp)
{
    unsigned char *destData = (unsigned char *)malloc(destW * destH * bpp);
    if (!destData)
        return NULL;

    float scaleX = (float)srcW / destW;
    float scaleY = (float)srcH / destH;

    for (int dy = 0; dy < destH; dy++)
    {
        for (int dx = 0; dx < destW; dx++)
        {
            float sx_float = dx * scaleX;
            float sy_float = dy * scaleY;

            int x1 = (int)sx_float;
            int y1 = (int)sy_float;
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // 邊界檢查
            if (x2 >= srcW)
                x2 = srcW - 1;
            if (y2 >= srcH)
                y2 = srcH - 1;

            float x_frac = sx_float - x1;
            float y_frac = sy_float - y1;

            int destIndex = (dy * destW + dx) * bpp;

            for (int c = 0; c < bpp; c++)
            {
                float q11 = srcData[(y1 * srcW + x1) * bpp + c];
                float q12 = srcData[(y2 * srcW + x1) * bpp + c]; // bottom-left
                float q21 = srcData[(y1 * srcW + x2) * bpp + c]; // top-right
                float q22 = srcData[(y2 * srcW + x2) * bpp + c];

                float top_interp = q11 * (1 - x_frac) + q21 * x_frac;
                float bottom_interp = q12 * (1 - x_frac) + q22 * x_frac;
                float final_val = top_interp * (1 - y_frac) + bottom_interp * y_frac;

                destData[destIndex + c] = (unsigned char)(final_val + 0.5f); // +0.5 for better rounding
            }
        }
    }
    return destData;
}

void PerformResample(ImageAsset *img, int targetW, int targetH, int method)
{
    if (!img || !img->originalData)
        return;

    printf("Resampling from %dx%d to %dx%d using %s\n", img->width, img->height, targetW, targetH, method == 0 ? "Nearest-Neighbor" : "Bilinear");

    unsigned char *resampledData = NULL;
    if (method == 0)
    { // Nearest-Neighbor
        resampledData = resample_nearest_neighbor(img->originalData, img->width, img->height, targetW, targetH, img->bpp);
    }
    else
    { // Bilinear
        resampledData = resample_bilinear(img->originalData, img->width, img->height, targetW, targetH, img->bpp);
    }

    if (resampledData)
    {
        // 釋放舊的處理後數據和紋理
        free(img->processedData);
        UnloadTexture(img->texture);

        // 更新 ImageAsset
        img->processedData = resampledData;

        // 建立新的 raylib Image 和 Texture
        Image r_img = {
            .data = img->processedData,
            .width = targetW,
            .height = targetH,
            .mipmaps = 1,
            .format = (img->bpp == 1) ? PIXELFORMAT_UNCOMPRESSED_GRAYSCALE : PIXELFORMAT_UNCOMPRESSED_R8G8B8};
        img->texture = LoadTextureFromImage(r_img);
    }
}

void PerformChainedResample(ImageAsset* img, int targetW, int targetH, int method)
{
    if (!img || !img->processedData) return;

    // 獲取當前 processedData 的尺寸來建立一個暫時的 Image
    Image currentImage = {
        .data = img->processedData,
        .width = img->texture.width, // 從 texture 獲取當前尺寸
        .height = img->texture.height,
        .mipmaps = 1,
        .format = (img->bpp == 1) ? PIXELFORMAT_UNCOMPRESSED_GRAYSCALE : PIXELFORMAT_UNCOMPRESSED_R8G8B8
    };

    printf("Chained Resampling from %dx%d to %dx%d using %s\n", currentImage.width, currentImage.height, targetW, targetH, method == 0 ? "Nearest-Neighbor" : "Bilinear");
    
    unsigned char* resampledData = NULL;
    if (method == 0) {
        resampledData = resample_nearest_neighbor(currentImage.data, currentImage.width, currentImage.height, targetW, targetH, img->bpp);
    } else {
        resampledData = resample_bilinear(currentImage.data, currentImage.width, currentImage.height, targetW, targetH, img->bpp);
    }

    if (resampledData) {
        free(img->processedData);
        UnloadTexture(img->texture);
        img->processedData = resampledData;
        
        Image r_img = {
            .data = img->processedData, .width = targetW, .height = targetH, .mipmaps = 1, .format = currentImage.format
        };
        img->texture = LoadTextureFromImage(r_img);
    }
}

int main(void)
{
    const int screenWidth = 1920;
    const int screenHeight = 1000;
    InitWindow(screenWidth, screenHeight, "Image Enhancement Toolkit");

    GameScreen currentScreen = MAIN_MENU;
    Rectangle assignment1Button = {screenWidth / 2.0f - 150, screenHeight / 2.0f - 50, 300, 100};

    ImageAsset images[TOTAL_IMAGES] = {0};
    bool resourcesLoaded = false;
    int selectedImage = -1;      // -1 表示沒有選中任何圖片
    int interpolationMethod = 0; // 0: Nearest, 1: Bilinear

    const char *rawImageFiles[NUM_RAW_IMAGES] = {"data/peppers.raw", "data/lena.raw", "data/goldhill.raw"};
    const int rawImgWidth = 512, rawImgHeight = 512;
    const char *bmpImageFiles[NUM_BMP_IMAGES] = {"data/baboon.bmp", "data/boat.bmp", "data/F16.bmp"};

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        // ==========================================================
        //  邏輯更新 (Update)
        // ==========================================================
        switch (currentScreen)
        {
        case MAIN_MENU:
        {
            if (CheckCollisionPointRec(GetMousePosition(), assignment1Button) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
            {
                currentScreen = ASSIGNMENT_1;
                printf("Loading images and generating reports...\n\n");

                // 1. 載入 RAW 影像
                for (int i = 0; i < NUM_RAW_IMAGES; i++)
                {
                    images[i].width = rawImgWidth;
                    images[i].height = rawImgHeight;
                    images[i].bpp = 1; // Grayscale
                    images[i].originalData = LoadRawImageData(rawImageFiles[i], rawImgWidth, rawImgHeight);
                    if (images[i].originalData)
                    {
                        int size = images[i].width * images[i].height * images[i].bpp;
                        images[i].processedData = (unsigned char *)malloc(size);
                        memcpy(images[i].processedData, images[i].originalData, size);

                        Image r_img = {.data = images[i].processedData, .width = images[i].width, .height = images[i].height, .mipmaps = 1, .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE};
                        images[i].texture = LoadTextureFromImage(r_img);
                        ReportCenteredPixels(r_img, rawImageFiles[i]);
                    }
                }

                // 2. 載入 BMP 影像
                for (int i = 0; i < NUM_BMP_IMAGES; i++)
                {
                    int index = NUM_RAW_IMAGES + i;
                    images[index].bpp = 3; // RGB
                    images[index].originalData = LoadBmpImageData(bmpImageFiles[i], &images[index].width, &images[index].height);
                    if (images[index].originalData)
                    {
                        int size = images[index].width * images[index].height * images[index].bpp;
                        images[index].processedData = (unsigned char *)malloc(size);
                        memcpy(images[index].processedData, images[index].originalData, size);

                        Image r_img = {.data = images[index].processedData, .width = images[index].width, .height = images[index].height, .mipmaps = 1, .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8};
                        images[index].texture = LoadTextureFromImage(r_img);
                        ReportCenteredPixels(r_img, bmpImageFiles[i]);
                    }
                }
                resourcesLoaded = true;
            }
        }
        break;

        case ASSIGNMENT_1:
        {
            // --- 處理返回 ---
            if (IsKeyPressed(KEY_B))
            { /* ... 資源釋放邏輯 ... */
            }

            // --- 處理圖片選擇 ---
            for (int i = 0; i < TOTAL_IMAGES; i++)
            {
                if (IsKeyPressed(KEY_ONE + i))
                {
                    selectedImage = i;
                    printf("Selected image %d\n", i + 1);
                }
            }

            // --- 處理影像效果套用 ---
            if (selectedImage != -1)
            {
                ImageAsset *img = &images[selectedImage];
                int size = img->width * img->height * img->bpp;

                if (IsKeyPressed(KEY_N))
                { // 負片
                    printf("Applying Negative effect to image %d\n", selectedImage + 1);
                    apply_image_negative(img->originalData, img->processedData, size);
                    UpdateTexture(img->texture, img->processedData);
                }
                if (IsKeyPressed(KEY_L))
                { // 對數
                    printf("Applying Log Transform to image %d\n", selectedImage + 1);
                    apply_log_transform(img->originalData, img->processedData, size);
                    UpdateTexture(img->texture, img->processedData);
                }
                if (IsKeyPressed(KEY_G))
                { // Gamma
                    // Gamma < 1 變亮, Gamma > 1 變暗
                    float gamma = 0.5f;
                    printf("Applying Gamma Correction (gamma=%.2f) to image %d\n", gamma, selectedImage + 1);
                    apply_gamma_transform(img->originalData, img->processedData, size, gamma);
                    UpdateTexture(img->texture, img->processedData);
                }
                if (IsKeyPressed(KEY_R))
                { // 還原
                    printf("Restoring image %d to original\n", selectedImage + 1);
                    PerformResample(img, 512, 512, interpolationMethod);
                }
                if (IsKeyPressed(KEY_M))
                {
                    interpolationMethod = 1 - interpolationMethod; // 0和1之間切換
                }
                if (IsKeyPressed(KEY_F1))
                    PerformResample(&images[selectedImage], 128, 128, interpolationMethod); // i
                if (IsKeyPressed(KEY_F2))
                    PerformResample(&images[selectedImage], 32, 32, interpolationMethod); // ii
                if (IsKeyPressed(KEY_F3))
                {
                    // 情境iii是從32x32放大，我們先縮小再放大來模擬
                    PerformResample(&images[selectedImage], 32, 32, interpolationMethod);
                    PerformChainedResample(img, 512, 512, interpolationMethod);
                }
                if (IsKeyPressed(KEY_F4))
                    PerformResample(&images[selectedImage], 1024, 512, interpolationMethod); // iv
                if (IsKeyPressed(KEY_F5))
                {
                    PerformResample(&images[selectedImage], 128, 128, interpolationMethod);
                    PerformChainedResample(img, 256, 512, interpolationMethod);
                }
            }
        }
        break;
        }

        // ==========================================================
        //  繪圖 (Draw)
        // ==========================================================
        BeginDrawing();
        ClearBackground(RAYWHITE);

        switch (currentScreen)
        {
        case MAIN_MENU:
        {
            DrawRectangleRec(assignment1Button, LIGHTGRAY);
            if (CheckCollisionPointRec(GetMousePosition(), assignment1Button))
            {
                DrawRectangleLinesEx(assignment1Button, 4, DARKGRAY);
            }
            DrawText("Assignment 1: Toolkit", assignment1Button.x + 20, assignment1Button.y + 35, 30, DARKGRAY);
        }
        break;
        case ASSIGNMENT_1:
        {
            if (resourcesLoaded)
            {
                ClearBackground(DARKGRAY);
                int padding = 10, startX = 5, startY = 50;
                int thumbWidth = 512, thumbHeight = 512;

                // 繪製所有圖片
                for (int i = 0; i < TOTAL_IMAGES; i++)
                {
                    int row = i / NUM_RAW_IMAGES;
                    int col = i % NUM_RAW_IMAGES;
                    int posX = startX + col * (thumbWidth + padding);
                    int posY = startY + row * (thumbHeight + padding + 25);

                    if (images[i].texture.id > 0)
                    {
                        DrawTextureEx(images[i].texture, (Vector2){posX, posY}, 0.0f, (float)thumbWidth / images[i].width, WHITE);
                        DrawText(TextFormat("%d", i + 1), posX + 5, posY + 5, 20, YELLOW);
                    }

                    // 繪製選中框
                    if (i == selectedImage)
                    {
                        DrawRectangleLines(posX, posY, thumbWidth, thumbHeight, YELLOW);
                    }
                }

                // 繪製操作說明
                const char* methodName = (interpolationMethod == 0) ? "Nearest-Neighbor" : "Bilinear";
                DrawText(TextFormat("METHOD: %s (Press M to switch)", methodName), startX, 15, 20, RAYWHITE);
                DrawText("Select: 1-6 | Effects: [N]egative, [L]og, [G]amma | [R]estore | [B]ack to Menu", startX, screenHeight - 30, 20, RAYWHITE);
            }
            else
            {
                DrawText("Some images failed to load...", 20, 20, 20, RED);
            }
        }
        break;
        }
        EndDrawing();
    }

    // --- 釋放所有資源 ---
    if (resourcesLoaded)
    {
        for (int i = 0; i < TOTAL_IMAGES; i++)
        {
            UnloadTexture(images[i].texture);
            free(images[i].originalData);
            free(images[i].processedData);
        }
    }
    CloseWindow();
    return 0;
}