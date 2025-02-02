#pragma once

#include <glm/glm.hpp>
#include "Quad.hpp"
#include "Program.hpp"

class Scene;
Program* LoadShaders(const ShaderSource& vertShaderObj, const ShaderSource& fragShaderObj);

struct RenderOptions {
    glm::ivec2 render_resolution=glm::ivec2(1280, 720);
    glm::ivec2 window_resolution=glm::ivec2(1280, 720);
    glm::vec3 uniform_light_col = glm::vec3(0.3f, 0.3f, 0.3f);
    glm::vec3 background_col = glm::vec3(1.0f, 1.0f, 1.0f);
    int tile_width = 100;
    int tile_height = 100;
    int max_depth = 2;
    int max_spp = -1;
    int RR_depth = 2;
    int tex_array_width = 2048;
    int tex_array_height = 2048;
    int denoiser_frame_cnt = 20;
    bool enable_RR = true;
    bool enable_denoiser = false;
    bool enable_tonemap = true;
    bool enable_aces = false;
    bool simple_aces_fit = false;
    bool opengl_normal_map = true;
    bool enable_env_map = false;
    bool enable_uniform_light = false;
    bool hide_emitters = false;
    bool enable_background = false;
    bool transparent_background = false;
    bool independent_render_size = false;
    bool enable_roughness_mollification = false;
    bool enable_volume_MIS = false;
    float env_map_intensity = 1.0f;
    float env_map_rot = 0.0f;
    float roughness_mollification_amt = 0.0f;
};

class Renderer {
public:
    Renderer(Scene* scene, const std::string& shadersDirectory);
    ~Renderer();

    void resizeRenderer();
    void reloadShaders();
    void render();
    void present();
    void update(float secondsElapsed);
    float getProgress();
    int getSampleCount();
    void getOutputBuffer(unsigned char**, int& w, int& h);

    void initGPUDataBuffers();
    void initFBOs();
    void initShaders();

protected:
    Scene* scene;
    Quad* quad;

    // Opengl buffer objects and textures for storing scene data on the GPU
    unsigned int BVHBuffer;
    unsigned int BVHTex;
    unsigned int vertexIndicesBuffer;
    unsigned int vertexIndicesTex;
    unsigned int verticesBuffer;
    unsigned int verticesTex;
    unsigned int normalsBuffer;
    unsigned int normalsTex;
    unsigned int materialsTex;
    unsigned int transformsTex;
    unsigned int lightsTex;
    unsigned int textureMapsArrayTex;
    unsigned int envMapTex;
    unsigned int envMapCDFTex;

    // FBOs
    unsigned int pathTraceFBO;
    unsigned int pathTraceFBOLowRes;
    unsigned int accumFBO;
    unsigned int outputFBO;

    // Shaders
    std::string shadersDirectory;
    Program* pathTraceShader;
    Program* pathTraceShaderLowRes;
    Program* outputShader;
    Program* tonemapShader;

    // Render textures
    unsigned int pathTraceTextureLowRes;
    unsigned int pathTraceTexture;
    unsigned int accumTexture;
    unsigned int tileOutputTexture[2];
    unsigned int denoisedTexture;

    // Render resolution and window resolution
    glm::ivec2 renderSize;
    glm::ivec2 windowSize;

    // Variables to track rendering status
    glm::ivec2 tile;
    glm::ivec2 numTiles;
    glm::vec2 invNumTiles;
    int tileWidth;
    int tileHeight;
    int currentBuffer;
    int frameCounter;
    int sampleCounter;
    float pixelRatio;

    // Denoiser output
    glm::vec3* denoiserInputFramePtr;
    glm::vec3* frameOutputPtr;
    bool denoised;

    bool initialized;
};
