#include "Model.hpp"

#include <print>
#include <stb_image.h>

#include "Mesh.hpp"

Model::Model(const std::string& path)
{
    loadModel(path);
}

void Model::loadModel(const std::string& path)
{
    Assimp::Importer importer;
    const aiScene* scene(importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace));
    if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
        std::println("ERROR::ASSIMP::{}", importer.GetErrorString());
        return;
    }

    directory=path.substr(0, path.find_last_of('/'));
    processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode* node, const aiScene* scene)
{
    for(size_t i=0; i<node->mNumMeshes; i++){
        aiMesh* mesh=scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene));

    }

    for(size_t i=0; i<node->mNumChildren; i++){
        processNode(node->mChildren[i], scene);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture> textures;

    for(size_t i=0; i<mesh->mNumVertices; i++){
        Vertex vertex;
        auto vert=mesh->mVertices[i];
        auto norm=mesh->mNormals[i];

        vertex.position={vert.x, vert.y, vert.z};
        vertex.normal={norm.x, norm.y, norm.z};

        if(mesh->mTextureCoords[0]){
            auto tex_coord=mesh->mTextureCoords[0][i];
            auto tagent=mesh->mTangents[i];
            auto bitangent=mesh->mBitangents[i];

            vertex.tex_coords={tex_coord.x, tex_coord.y};
            vertex.tangent={tagent.x, tagent.y, tagent.z};
            vertex.bitangent={bitangent.x, bitangent.y, bitangent.z};
        }else
            vertex.tex_coords={0.0f, 0.0f};

        vertices.push_back(vertex);
    }

    for(size_t i=0; i<mesh->mNumFaces; i++){
        aiFace face=mesh->mFaces[i];
        for(size_t j=0; j<face.mNumIndices; j++){
            indices.push_back(face.mIndices[j]);
        }
    }

    aiMaterial* material=scene->mMaterials[mesh->mMaterialIndex];

    std::vector<Texture> diffuse_maps=loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse");
    textures.insert(textures.end(), diffuse_maps.begin(), diffuse_maps.end());

    std::vector<Texture> specular_maps=loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
    textures.insert(textures.end(), specular_maps.begin(), specular_maps.end());

    std::vector<Texture> normal_maps=loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal");
    textures.insert(textures.end(), normal_maps.begin(), normal_maps.end());

    std::vector<Texture> height_maps=loadMaterialTextures(material, aiTextureType_AMBIENT, "texture_height");
    textures.insert(textures.end(), height_maps.begin(), height_maps.end());

    return Mesh(vertices, indices, textures);
}

void Model::draw(Shader& shader)
{
    for(auto& mesh: meshes)
        mesh.draw(shader);
}

std::vector<Texture> Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string type_name)
{
    std::vector<Texture> textures;
    for(size_t i=0; i<mat->GetTextureCount(type); i++){
        aiString str;
        mat->GetTexture(type, i, &str);

        bool skip=false;
        for(size_t j=0; j<textures_loaded.size(); j++){
            if(std::strcmp(textures_loaded[j].getName().data(), str.C_Str())==0){
                textures.push_back(textures_loaded[j]);
                skip=true;
                break;
            }
        }

        if(!skip){
            Texture texture(directory+'/'+str.C_Str(), type_name);
            textures.push_back(texture);
            textures_loaded.push_back(texture);
        }
    }

    return textures;
}
