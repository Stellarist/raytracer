#pragma once

// #include <glm/glm.hpp>

struct ivec2{
    int x{0};
    int y{0};
};

struct LaunchParams{
    int frame_id{0};
    uint32_t* color_buffer{nullptr};
    // glm::ivec2 fb_size{};
    ivec2 fb_size{};
};

