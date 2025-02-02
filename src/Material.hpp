#pragma once

#include <glm/glm.hpp>

enum AlphaMode
{
    Opaque,
    Blend,
    Mask
};

enum MediumType
{
    None,
    Absorb,
    Scatter,
    Emissive
};

class Material {
public:
    Material();

    glm::vec3 base_color;
    float anisotropic;

    glm::vec3 emission;
    float padding1;

    float metallic;
    float roughness;
    float subsurface;
    float specular_tint;

    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;

    float spec_trans;
    float ior;
    float medium_type;
    float medium_density;
    
    glm::vec3 medium_color;
    float medium_anisotropy;

    float base_color_texid;
    float metallic_roughness_texid;
    float normalmap_texid;
    float emissionmap_texid;

    float opacity;
    float alpha_mode;
    float alpha_cutoff;
    float padding2;
};

inline Material::Material()
{
    base_color = glm::vec3(1.0f, 1.0f, 1.0f);
    anisotropic = 0.0f;
    emission = glm::vec3(0.0f, 0.0f, 0.0f);

    metallic     = 0.0f;
    roughness    = 0.5f;
    subsurface   = 0.0f;
    specular_tint = 0.0f;

    sheen          = 0.0f;
    sheen_tint      = 0.0f;
    clearcoat      = 0.0f;
    clearcoat_gloss = 0.0f;

    spec_trans        = 0.0f;
    ior              = 1.5f;
    medium_type       = 0.0f;
    medium_density    = 0.0f;

    medium_color      = glm::vec3(1.0f, 1.0f, 1.0f);
    medium_anisotropy = 0.0f;

    base_color_texid         = -1.0f;
    metallic_roughness_texid = -1.0f;
    normalmap_texid         = -1.0f;
    emissionmap_texid       = -1.0f;

    opacity     = 1.0f;
    alpha_mode   = 0.0f;
    alpha_cutoff = 0.0f;
}
