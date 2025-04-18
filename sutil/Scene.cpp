/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda/whitted.h>
#include <sutil/CuBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/Record.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>

#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION        // Implementation in sutil.cpp
//#define STB_IMAGE_WRITE_IMPLEMENTATION  //
#if defined( WIN32 )
#    pragma warning( push )
#    pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#    pragma warning( pop )
#endif

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace sutil
{

namespace
{

float3 make_float3_from_double( double x, double y, double z )
{
    return make_float3( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ) );
}

float4 make_float4_from_double( double x, double y, double z, double w )
{
    return make_float4( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ), static_cast<float>( w ) );
}

typedef Record<whitted::HitGroupData> HitGroupRecord;

void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

template<typename T>
BufferView<T> bufferViewFromGLTF( const tinygltf::Model& model, Scene& scene, const int32_t accessor_idx )
{
    if( accessor_idx == -1 )
        return BufferView<T>();

    const auto& gltf_accessor    = model.accessors[ accessor_idx ];
    const auto& gltf_buffer_view = model.bufferViews[ gltf_accessor.bufferView ];

    const int32_t elmt_byte_size =
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE  ? 1 :
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2 :
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT   ? 4 :
            gltf_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT          ? 4 :
            0;
    if( !elmt_byte_size )
        throw Exception( "gltf accessor component type not supported" );

    const CUdeviceptr buffer_base = scene.getBuffer( gltf_buffer_view.buffer );
    BufferView<T> buffer_view;
    buffer_view.data           = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;
    buffer_view.byte_stride    = static_cast<uint16_t>( gltf_buffer_view.byteStride );
    buffer_view.count          = static_cast<uint32_t>( gltf_accessor.count );
    buffer_view.elmt_byte_size = static_cast<uint16_t>( elmt_byte_size );

    if( buffer_view.byte_stride == 0 )
    {
        buffer_view.byte_stride = static_cast< uint16_t >( elmt_byte_size );
        switch( gltf_accessor.type )
        {
        case TINYGLTF_TYPE_VEC2: buffer_view.byte_stride *= 2;  break;
        case TINYGLTF_TYPE_VEC3: buffer_view.byte_stride *= 3;  break;
        case TINYGLTF_TYPE_VEC4: buffer_view.byte_stride *= 4;  break;
        case TINYGLTF_TYPE_MAT2: buffer_view.byte_stride *= 4;  break;
        case TINYGLTF_TYPE_MAT3: buffer_view.byte_stride *= 9;  break;
        case TINYGLTF_TYPE_MAT4: buffer_view.byte_stride *= 16; break;
        default: break;
        }
    }

    return buffer_view;
}

void processGLTFNode(
        Scene& scene,
        const tinygltf::Model& model,
        const tinygltf::Node& gltf_node,
        const Matrix4x4& parent_matrix
        )
{
    const Matrix4x4 translation = gltf_node.translation.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::translate( make_float3_from_double(
                    gltf_node.translation[0],
                    gltf_node.translation[1],
                    gltf_node.translation[2]
                    ) );

    const Matrix4x4 rotation = gltf_node.rotation.empty() ?
        Matrix4x4::identity() :
        Quaternion(
                static_cast<float>( gltf_node.rotation[3] ),
                static_cast<float>( gltf_node.rotation[0] ),
                static_cast<float>( gltf_node.rotation[1] ),
                static_cast<float>( gltf_node.rotation[2] )
                ).rotationMatrix();

    const Matrix4x4 scale = gltf_node.scale.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::scale( make_float3_from_double(
                    gltf_node.scale[0],
                    gltf_node.scale[1],
                    gltf_node.scale[2]
                    ) );

    std::vector<float> gltf_matrix;
    for( double x : gltf_node.matrix )
        gltf_matrix.push_back( static_cast<float>( x ) );
    const Matrix4x4 matrix = gltf_node.matrix.empty() ?
        Matrix4x4::identity() :
        Matrix4x4( reinterpret_cast<float*>( gltf_matrix.data() ) ).transpose();

    const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale ;

    if( gltf_node.camera != -1 )
    {
        const auto& gltf_camera = model.cameras[ gltf_node.camera ];
        std::cerr << "Processing camera '" << gltf_camera.name << "'\n"
            << "\ttype: " << gltf_camera.type << "\n";
        if( gltf_camera.type != "perspective" )
        {
            std::cerr << "\tskipping non-perpective camera\n";
            return;
        }

        const float3 eye    = make_float3( node_xform*make_float4_from_double( 0.0f, 0.0f,  0.0f, 1.0f ) );
        const float3 up     = make_float3( node_xform*make_float4_from_double( 0.0f, 1.0f,  0.0f, 0.0f ) );
        const float  yfov   = static_cast<float>( gltf_camera.perspective.yfov ) * 180.0f / static_cast<float>( M_PI );

        std::cerr << "\teye   : " << eye.x    << ", " << eye.y    << ", " << eye.z    << std::endl;
        std::cerr << "\tup    : " << up.x     << ", " << up.y     << ", " << up.z     << std::endl;
        std::cerr << "\tfov   : " << yfov     << std::endl;
        std::cerr << "\taspect: " << gltf_camera.perspective.aspectRatio << std::endl;

        Camera camera;
        camera.setFovY       ( yfov                                );
        camera.setAspectRatio( static_cast<float>( gltf_camera.perspective.aspectRatio ) );
        camera.setEye        ( eye                                 );
        camera.setUp         ( up                                  );
        scene.addCamera( camera );
    }
    else if( gltf_node.mesh != -1 )
    {
        auto instance = std::make_shared<Scene::Instance>();
        instance->transform  = node_xform;
        instance->mesh_idx   = gltf_node.mesh;
        instance->world_aabb = scene.meshes()[gltf_node.mesh]->object_aabb;
        instance->world_aabb.transform( node_xform );
        scene.addInstance( instance );
    }

    if( !gltf_node.children.empty() )
    {
        for( int32_t child : gltf_node.children )
        {
            processGLTFNode( scene, model, model.nodes[child], node_xform );
        }
    }
}

} // end anon namespace

template<typename TextureInfo>
void parseTextureInfo( const Scene& scene, const TextureInfo &inTex, MaterialData::Texture &outTex )
{
    if( inTex.index >= 0 )
    {
        outTex.tex = scene.getSampler( inTex.index );
        outTex.texcoord = inTex.texCoord;

        auto itr = inTex.extensions.find( "KHR_texture_transform" );

        float2 offset = { 0, 0 };
        float  rotation = 0;
        float2 scale = { 1, 1 };
        if( itr != inTex.extensions.end() )
        {
            if( itr->second.Has( "offset" ) )
            {
                auto offsetValue = itr->second.Get( "offset" );
                offset.x = ( float )offsetValue.Get( 0 ).GetNumberAsDouble();
                offset.y = ( float )offsetValue.Get( 1 ).GetNumberAsDouble();
            }

            if( itr->second.Has( "rotation" ) )
            {
                auto rotationValue = itr->second.Get( "rotation" );
                rotation = ( float )rotationValue.GetNumberAsDouble();
            }

            if( itr->second.Has( "scale" ) )
            {
                auto scaleValue = itr->second.Get( "scale" );
                scale.x = ( float )scaleValue.Get( 0 ).GetNumberAsDouble();
                scale.y = ( float )scaleValue.Get( 1 ).GetNumberAsDouble();
            }
        }

        outTex.texcoord_offset = offset;
        outTex.texcoord_scale = scale;
        outTex.texcoord_rotation = make_float2( ( float )sinf( rotation ), ( float )cosf( rotation ) );

        if( outTex.texcoord >= ( int )GeometryData::num_texcoords )
        {
            std::cerr << "\tMaximum supported texcoords exceded.\n";
            outTex.texcoord = 0;
        }
    }
    else
    {
        outTex.tex = 0;
        outTex.texcoord = 0;
    }
};

void loadScene( const std::string& filename, Scene& scene )
{
    scene.cleanup();

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret;
    if( filename.size() >= 4 && strncmp( filename.c_str() + filename.size() - 4, ".glb", 4) == 0 )
        ret = loader.LoadBinaryFromFile( &model, &err, &warn, filename );
    else
        ret = loader.LoadASCIIFromFile( &model, &err, &warn, filename );
    if( !warn.empty() )
        std::cerr << "glTF WARNING: " << warn << std::endl;
    if( !ret )
    {
        std::cerr << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        throw Exception( err.c_str() );
    }

    //
    // Process buffer data first -- buffer views will reference this list
    //
    for( const auto& gltf_buffer : model.buffers )
    {
        const uint64_t buf_size = gltf_buffer.data.size();
        std::cerr << "Processing glTF buffer '" << gltf_buffer.name << "'\n"
                  << "\tbyte size: " << buf_size << "\n"
                  << "\turi      : " << gltf_buffer.uri << std::endl;

        scene.addBuffer( buf_size,  gltf_buffer.data.data() );
    }

    //
    // Images -- just load all up front for simplicity
    //
    for( const auto& gltf_image : model.images )
    {
        std::cerr << "Processing image '" << gltf_image.name << "'\n"
                  << "\t(" << gltf_image.width << "x" << gltf_image.height << ")x" << gltf_image.component << "\n"
                  << "\tbits: " << gltf_image.bits << std::endl;

        assert( gltf_image.component == 4 );
        assert( gltf_image.bits      == 8 || gltf_image.bits == 16 );

        scene.addImage(
                gltf_image.width,
                gltf_image.height,
                gltf_image.bits,
                gltf_image.component,
                gltf_image.image.data()
                );
    }

    //
    // Textures -- refer to previously loaded images
    //
    for( const auto& gltf_texture : model.textures )
    {
        if( gltf_texture.sampler == -1 )
        {
            scene.addSampler( cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source );
            continue;
        }

        const auto& gltf_sampler = model.samplers[ gltf_texture.sampler ];

        const cudaTextureAddressMode address_s = gltf_sampler.wrapS == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                                 gltf_sampler.wrapS == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                                                               cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = gltf_sampler.wrapT == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
                                                 gltf_sampler.wrapT == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT ? cudaAddressModeMirror :
                                                                                                               cudaAddressModeWrap;
        const cudaTextureFilterMode  filter    = gltf_sampler.minFilter == TINYGLTF_TEXTURE_FILTER_NEAREST   ? cudaFilterModePoint   :
                                                                                                               cudaFilterModeLinear;
        scene.addSampler( address_s, address_t, filter, gltf_texture.source );
    }

    //
    // Materials
    //
    for( auto& gltf_material : model.materials )
    {
        std::cerr << "Processing glTF material: '" << gltf_material.name << "'\n";
        MaterialData mtl;

        mtl.doubleSided = gltf_material.doubleSided;

        if( gltf_material.alphaMode == "MASK" )
        {
           mtl.alpha_mode = MaterialData::ALPHA_MODE_MASK;
           mtl.alpha_cutoff = (float)gltf_material.alphaCutoff;
        }
        else if( gltf_material.alphaMode == "BLEND" )
        {
            mtl.alpha_mode = MaterialData::ALPHA_MODE_BLEND;
        }
        else if( gltf_material.alphaMode == "OPAQUE" )
        {
            mtl.alpha_mode = MaterialData::ALPHA_MODE_OPAQUE;
        }
        else
        {
            std::cerr << "\tInvalid alpha mode\n";
        }

        {
            const auto base_color_it = gltf_material.values.find( "baseColorFactor" );
            if( base_color_it != gltf_material.values.end() )
            {
                const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
                mtl.pbr.base_color = make_float4_from_double( c[0], c[1], c[2], c[3] );
                std::cerr
                    << "\tBase color: ("
                    << mtl.pbr.base_color.x << ", "
                    << mtl.pbr.base_color.y << ", "
                    << mtl.pbr.base_color.z << ")\n";
            }
            else
            {
                std::cerr << "\tUsing default base color factor\n";
            }
        }

        parseTextureInfo( scene, gltf_material.pbrMetallicRoughness.baseColorTexture, mtl.pbr.base_color_tex );
        parseTextureInfo( scene, gltf_material.pbrMetallicRoughness.metallicRoughnessTexture, mtl.pbr.metallic_roughness_tex );
        parseTextureInfo( scene, gltf_material.normalTexture, mtl.normal_tex );
        parseTextureInfo( scene, gltf_material.emissiveTexture, mtl.emissive_tex );

        {
            const auto roughness_it = gltf_material.values.find( "roughnessFactor" );
            if( roughness_it != gltf_material.values.end() )
            {
                mtl.pbr.roughness = static_cast< float >( roughness_it->second.Factor() );
                std::cerr << "\tRougness:  " << mtl.pbr.roughness << "\n";
            }
            else
            {
                std::cerr << "\tUsing default roughness factor\n";
            }
        }

        {
            const auto metallic_it = gltf_material.values.find( "metallicFactor" );
            if( metallic_it != gltf_material.values.end() )
            {
                mtl.pbr.metallic = static_cast< float >( metallic_it->second.Factor() );
                std::cerr << "\tMetallic:  " << mtl.pbr.metallic << "\n";
            }
            else
            {
                std::cerr << "\tUsing default metallic factor\n";
            }
        }

        {
            const auto emissive_factor_it = gltf_material.additionalValues.find( "emissiveFactor" );
            if( emissive_factor_it != gltf_material.additionalValues.end() )
            {
                const tinygltf::ColorValue c = emissive_factor_it->second.ColorFactor();
                mtl.emissive_factor = make_float3_from_double( c[0], c[1], c[2] );
                std::cerr
                    << "\tEmissive factor: ("
                    << mtl.emissive_factor.x << ", "
                    << mtl.emissive_factor.y << ", "
                    << mtl.emissive_factor.z << ")\n";
            }
            else
            {
                std::cerr << "\tUsing default base color factor\n";
            }
        }

        scene.addMaterial( mtl );
    }

    //
    // Meshes
    //
    for( auto& gltf_mesh : model.meshes )
    {
        std::cerr << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
        std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;

        auto mesh = std::make_shared<Scene::MeshGroup>();
        scene.addMesh( mesh );
        mesh->name = gltf_mesh.name;
        mesh->object_aabb.invalidate();

        for( auto& gltf_primitive : gltf_mesh.primitives )
        {
            if( gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES ) // Ignore non-triangle meshes
            {
                std::cerr << "\tNon-triangle primitive: skipping\n";
                continue;
            }
            
            mesh->indices.push_back( bufferViewFromGLTF<uint32_t>( model, scene, gltf_primitive.indices ) );
            mesh->material_idx.push_back( gltf_primitive.material );
            std::cerr << "\t\tNum triangles: " << mesh->indices.back().count / 3 << std::endl;

            assert( gltf_primitive.attributes.find( "POSITION" ) != gltf_primitive.attributes.end() );
            const int32_t pos_accessor_idx = gltf_primitive.attributes.at( "POSITION" );
            mesh->positions.push_back( bufferViewFromGLTF<float3>( model, scene, pos_accessor_idx ) );

            const auto& pos_gltf_accessor = model.accessors[pos_accessor_idx];

            if( !pos_gltf_accessor.minValues.empty() && !pos_gltf_accessor.maxValues.empty() )
            {
                mesh->object_aabb.include( Aabb(
                    make_float3_from_double(
                        pos_gltf_accessor.minValues[0],
                        pos_gltf_accessor.minValues[1],
                        pos_gltf_accessor.minValues[2]
                    ),
                    make_float3_from_double(
                        pos_gltf_accessor.maxValues[0],
                        pos_gltf_accessor.maxValues[1],
                        pos_gltf_accessor.maxValues[2]
                    ) ) );
            }

            auto normal_accessor_iter = gltf_primitive.attributes.find( "NORMAL" );
            if( normal_accessor_iter != gltf_primitive.attributes.end() )
            {
                std::cerr << "\t\tHas vertex normals: true\n";
                mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, normal_accessor_iter->second ) );
            }
            else
            {
                std::cerr << "\t\tHas vertex normals: false\n";
                mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, -1 ) );
            }

            for( size_t j = 0; j < GeometryData::num_texcoords; ++j )
            {
                char texcoord_str[128];
                snprintf( texcoord_str, 128, "TEXCOORD_%i", (int)j );
                auto texcoord_accessor_iter = gltf_primitive.attributes.find( texcoord_str );
                if( texcoord_accessor_iter != gltf_primitive.attributes.end() )
                {
                    std::cerr << "\t\tHas texcoords_" << j << ": true\n";
                    mesh->texcoords[j].push_back( bufferViewFromGLTF<Vec2f>( model, scene, texcoord_accessor_iter->second ) );
                }
                else
                {
                    std::cerr << "\t\tHas texcoords_" << j << ": false\n";
                    mesh->texcoords[j].push_back( bufferViewFromGLTF<Vec2f>( model, scene, -1 ) );
                }
            }

            auto color_accessor_iter = gltf_primitive.attributes.find( "COLOR_0" );
            if( color_accessor_iter != gltf_primitive.attributes.end() )
            {
                std::cerr << "\t\tHas color_0: true\n";
                mesh->colors.push_back( bufferViewFromGLTF<Vec4f>( model, scene, color_accessor_iter->second ) );
            }
            else
            {
                std::cerr << "\t\tHas color_0: false\n";
                mesh->colors.push_back( bufferViewFromGLTF<Vec4f>( model, scene, -1 ) );
            }
        }
    }

    //
    // Process nodes
    //
    std::vector<int32_t> root_nodes( model.nodes.size(), 1 );
    for( auto& gltf_node : model.nodes )
        for( int32_t child : gltf_node.children )
            root_nodes[child] = 0;

    for( size_t i = 0; i < root_nodes.size(); ++i )
    {
        if( !root_nodes[i] )
            continue;
        auto& gltf_node = model.nodes[i];

        processGLTFNode( scene, model, gltf_node, Matrix4x4::identity() );
    }
}


Scene::Scene( void ) {}


Scene::~Scene( void )
{
    cleanup();
}


void Scene::addBuffer( const uint64_t buf_size, const void* data )
{
        CUdeviceptr buffer = 0;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &buffer ), buf_size ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( buffer ),
                    data,
                    buf_size,
                    cudaMemcpyHostToDevice
                    ) );
        m_buffers.push_back( buffer );
}


void Scene::addImage(
                const int32_t width,
                const int32_t height,
                const int32_t bits_per_component,
                const int32_t num_components,
                const void* data
                )
{
    // Allocate CUDA array in device memory
    int32_t               pitch;
    cudaChannelFormatDesc channel_desc;
    if( bits_per_component == 8 )
    {
        pitch        = width*num_components*sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if( bits_per_component == 16 )
    {
        pitch        = width*num_components*sizeof(uint16_t);
        channel_desc = cudaCreateChannelDesc<ushort4>();
    }
    else
    {
        throw Exception( "Unsupported bits/component in glTF image" );
    }


    cudaArray_t   cuda_array = nullptr;
    CUDA_CHECK( cudaMallocArray(
                &cuda_array,
                &channel_desc,
                width,
                height
                ) );
    CUDA_CHECK( cudaMemcpy2DToArray(
                cuda_array,
                0,     // X offset
                0,     // Y offset
                data,
                pitch,
                pitch,
                height,
                cudaMemcpyHostToDevice
                ) );
    m_images.push_back( cuda_array );
}


 void Scene::addSampler(
         cudaTextureAddressMode address_s,
         cudaTextureAddressMode address_t,
         cudaTextureFilterMode  filter,
         const int32_t          image_idx
         )
{
    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = getImage( image_idx );

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = address_s;
    tex_desc.addressMode[1]      = address_t;
    tex_desc.filterMode          = filter;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK( cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr ) );
    m_samplers.push_back( cuda_tex );
}


CUdeviceptr Scene::getBuffer( int32_t buffer_index ) const
{
    return m_buffers[ buffer_index ];
}


cudaArray_t Scene::getImage( int32_t image_index ) const
{
    return m_images[ image_index ];
}


cudaTextureObject_t Scene::getSampler( int32_t sampler_index ) const
{
    return m_samplers[ sampler_index ];
}


void Scene::finalize( bool create_pipeline, uint32_t ray_type_count )
{
    createContext();
    buildMeshAccels();
    buildInstanceAccel( ray_type_count );

    // Some apps might only need the mesh/instance geometry and not
    // the shader programs or pipeline.
    if( create_pipeline )
    {
        createPTXModule();
        createProgramGroups();
        createPipeline();
        createSBT();
    }

    m_scene_aabb.invalidate();
    for( const auto& instance : m_instances )
        m_scene_aabb.include( instance->world_aabb );

    if( !m_cameras.empty() )
        m_cameras.front().setLookat( m_scene_aabb.center() );
}


void Scene::finalize()
{
    finalize( /*create_pipeline*/ true, whitted::RAY_TYPE_COUNT );
}


void sutil::Scene::cleanup()
{
    // OptiX cleanup
    if( m_pipeline )
    {
        OPTIX_CHECK( optixPipelineDestroy( m_pipeline ) );
        m_pipeline = 0;
    }
    if( m_raygen_prog_group )
    {
        OPTIX_CHECK( optixProgramGroupDestroy( m_raygen_prog_group ) );
        m_raygen_prog_group = 0;
    }
    if( m_radiance_miss_group )
    {
        OPTIX_CHECK( optixProgramGroupDestroy( m_radiance_miss_group ) );
        m_radiance_miss_group = 0;
    }
    if( m_occlusion_miss_group )
    {
        OPTIX_CHECK( optixProgramGroupDestroy( m_occlusion_miss_group ) );
        m_occlusion_miss_group = 0;
    }
    if( m_radiance_hit_group )
    {
        OPTIX_CHECK( optixProgramGroupDestroy( m_radiance_hit_group ) );
        m_radiance_hit_group = 0;
    }
    if( m_occlusion_hit_group )
    {
        OPTIX_CHECK( optixProgramGroupDestroy( m_occlusion_hit_group ) );
        m_occlusion_hit_group = 0;
    }
    if( m_ptx_module )
    {
        OPTIX_CHECK( optixModuleDestroy( m_ptx_module ) );
        m_ptx_module = 0;
    }
    if( m_context )
    {
        OPTIX_CHECK( optixDeviceContextDestroy( m_context ) );
        m_context = 0;
    }

    // Free buffers for mesh (indices, positions, normals, texcoords)
    for( CUdeviceptr& buffer : m_buffers )
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( buffer ) ) );
    m_buffers.clear();

    // Destroy textures (base_color, metallic_roughness, normal)
    for( cudaTextureObject_t& texture : m_samplers )
        CUDA_CHECK( cudaDestroyTextureObject( texture ) );
    m_samplers.clear();

    for( cudaArray_t& image : m_images )
        CUDA_CHECK( cudaFreeArray( image ) );
    m_images.clear();

    if( m_d_ias_output_buffer )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_d_ias_output_buffer ) ) );
        m_d_ias_output_buffer = 0;
    }
    if( m_sbt.raygenRecord )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.raygenRecord ) ) );
        m_sbt.raygenRecord = 0;
    }
    if( m_sbt.missRecordBase )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.missRecordBase ) ) );
        m_sbt.missRecordBase = 0;
    }
    if( m_sbt.hitgroupRecordBase )
    {
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_sbt.hitgroupRecordBase ) ) );
        m_sbt.hitgroupRecordBase = 0;
    }
    for( auto mesh : m_meshes )
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( mesh->d_gas_output ) ) );
    m_meshes.clear();
    m_instances.clear();
}


sutil::Camera sutil::Scene::camera() const
{
    if( !m_cameras.empty() )
    {
        std::cerr << "Returning first camera" << std::endl;
        return m_cameras.front();
    }

    std::cerr << "Returning default camera" << std::endl;
    Camera cam;
    cam.setFovY( 45.0f );
    cam.setLookat( m_scene_aabb.center() );
    cam.setEye   ( m_scene_aabb.center() + make_float3( 0.0f, 0.0f, 1.5f*m_scene_aabb.maxExtent() ) );
    return cam;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void Scene::createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );

    CUcontext          cuCtx = nullptr;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
#if OPTIX_DEBUG_DEVICE_CODE
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
}

void Scene::buildMeshAccels()
{
    // Problem:
    // The memory requirements of a compacted GAS are unknown prior to building the GAS.
    // Hence, compaction of a GAS requires to build the GAS first and allocating memory for the compacted GAS afterwards.
    // This causes a device-host synchronization point, potentially harming performance.
    // This is most likely the case for small GASes where the actual building and compaction of the GAS is very fast.
    // A naive algorithm processes one GAS at a time with the following steps:
    // 1. compute memory sizes for the build process (temporary buffer size and build buffer size)
    // 2. allocate temporary and build buffer
    // 3. build the GAS (with temporary and build buffer) and compute the compacted size
    // If compacted size is smaller than build buffer size (i.e., compaction is worth it):
    // 4. allocate compacted buffer (final output buffer)
    // 5. compact GAS from build buffer into compact buffer
    //
    // Idea of the algorithm:
    // Batch process the building and compaction of multiple GASes to avoid host-device synchronization.
    // Ideally, the number of synchronization points would be linear with the number of batches rather than the number of GASes.
    // The main constraints for selecting batches of GASes are:
    // a) the peak memory consumption when batch processing GASes, and
    // b) the amount of memory for the output buffer(s), containing the compacted GASes. This is also part of a), but is also important after the build process.
    // For the latter we try to keep it as minimal as possible, i.e., the total memory requirements for the output should equal the sum of the compacted sizes of the GASes.
    // Hence, it should be avoided to waste memory by allocating buffers that are bigger than what is required for a compacted GAS.
    //
    // The peak memory consumption effectively defines the efficiency of the algorithm.
    // If memory was unlimited, compaction isn't needed at all.
    // A lower bound for the peak memory consumption during the build is the output of the process, the size of the compacted GASes.
    // Peak memory consumption effectively defines the memory pool available during the batch building and compaction of GASes.
    //
    // The algorithm estimates the size of the compacted GASes by a give compaction ratio as well as the computed build size of each GAS.
    // The compaction ratio is defined as: size of compacted GAS / size of build output of GAS.
    // The validity of this estimate therefore depends on the assumed compaction ratio.
    // The current algorithm assumes a fixed compaction ratio.
    // Other strategies could be:
    // - update the compaction ration on the fly by do statistics on the already processed GASes to have a better guess for the remaining batches
    // - multiple compaction rations by type of GAS (e.g., motion vs static), since the type of GAS impacts the compaction ratio
    // Further, compaction may be skipped for GASes that do not benefit from compaction (compaction ratio of 1.0).
    //
    // Before selecting GASes for a batch, all GASes are sorted by size (their build size).
    // Big GASes are handled before smaller GASes as this will increase the likelihood of the peak memory consumption staying close to the minimal memory consumption.
    // This also increase the benefit of batching since small GASes that benefit most from avoiding synchronizations are built "together".
    // The minimum batch size is one GAS to ensure forward process.
    //
    // Goal:
    // Estimate the required output size (the minimal peak memory consumption) and work within these bounds.
    // Batch process GASes as long as they are expected to fit into the memory bounds (non strict).
    //
    // Assumptions:
    // The inputs to each GAS are already in device memory and are needed afterwards.
    // Otherwise this could be factored into the peak memory consumption.
    // E.g., by uploading the input data to the device only just before building the GAS and releasing it right afterwards.
    //
    // Further, the peak memory consumption of the application / system is influenced by many factors unknown to this algorithm.
    // E.g., if it is known that a big pool of memory is needed after GAS building anyways (e.g., textures that need to be present on the device),
    // peak memory consumption will be higher eventually and the GAS build process could already make use of a bigger memory pool.
    //
    // TODOs:
    // - compaction ratio estimation / updating
    // - handling of non-compactable GASes
    // - integration of GAS input data upload / freeing
    // - add optional hard limits / check for hard memory limits (shrink batch size / abort, ...)
    //////////////////////////////////////////////////////////////////////////

    // Magic constants:

    // see explanation above
    constexpr double initialCompactionRatio = 0.5;

    // It is assumed that trace is called later when the GASes are still in memory.
    // We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
    // Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
    constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

    //////////////////////////////////////////////////////////////////////////

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    struct GASInfo {
        std::vector<OptixBuildInput> buildInputs;
        OptixAccelBufferSizes gas_buffer_sizes;
        std::shared_ptr<MeshGroup> mesh;
    };
    std::multimap<size_t, GASInfo> gases;
    size_t totalTempOutputSize = 0;

    unsigned opaque_triangle_input_flags[2] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
    unsigned mask_triangle_input_flags[2]   = { OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };
    unsigned blend_triangle_input_flags[2]  = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING };

    for( size_t i = 0; i < m_meshes.size(); ++i )
    {
        auto& mesh = m_meshes[i];

        const size_t num_subMeshes = mesh->indices.size();
        std::vector<OptixBuildInput> buildInputs( num_subMeshes );

        assert( mesh->positions.size() == num_subMeshes &&
            mesh->normals.size() == num_subMeshes &&
            mesh->colors.size() == num_subMeshes );

        for( size_t j = 0; j < GeometryData::num_texcoords; ++j )
            assert( mesh->texcoords[j].size() == num_subMeshes );

        for(size_t j = 0; j < num_subMeshes; ++j)
        {
            OptixBuildInput& triangle_input                          = buildInputs[j];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes         =
                mesh->positions[j].byte_stride ?
                mesh->positions[j].byte_stride :
                sizeof(float3);
            triangle_input.triangleArray.numVertices                 = mesh->positions[j].count;
            triangle_input.triangleArray.vertexBuffers               = &(mesh->positions[j].data);
            triangle_input.triangleArray.indexFormat                 =
                mesh->indices[j].elmt_byte_size == 0 ? OPTIX_INDICES_FORMAT_NONE :
                mesh->indices[j].elmt_byte_size == 1 ? OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3 :
                mesh->indices[j].elmt_byte_size == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :
                                                       OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes          =
                mesh->indices[j].byte_stride ?
                mesh->indices[j].byte_stride*3 :
                mesh->indices[j].elmt_byte_size*3;
            triangle_input.triangleArray.numIndexTriplets            = mesh->indices[j].count / 3;
            triangle_input.triangleArray.indexBuffer                 = mesh->indices[j].data;
            triangle_input.triangleArray.numSbtRecords               = 1;

            const int32_t mat_idx = mesh->material_idx[j];
            if( mat_idx >= 0 )
            {
                auto alpha_mode = m_materials[mat_idx].alpha_mode;
                switch( alpha_mode )
                {
                case MaterialData::ALPHA_MODE_MASK:
                    triangle_input.triangleArray.flags = &mask_triangle_input_flags[m_materials[mat_idx].doubleSided];
                    break;
                case MaterialData::ALPHA_MODE_BLEND:
                    triangle_input.triangleArray.flags = &blend_triangle_input_flags[m_materials[mat_idx].doubleSided];
                    break;
                default:
                    triangle_input.triangleArray.flags = &opaque_triangle_input_flags[m_materials[mat_idx].doubleSided];
                    break;
                }
            }
            else
            {
                triangle_input.triangleArray.flags = &opaque_triangle_input_flags[0]; // default is single sided
            }
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &accel_options, buildInputs.data(),
                                                   static_cast<unsigned int>( num_subMeshes ), &gas_buffer_sizes ) );

        totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
        GASInfo g = {std::move( buildInputs ), gas_buffer_sizes, mesh};
        gases.emplace( gas_buffer_sizes.outputSizeInBytes, g );
    }

    size_t totalTempOutputProcessedSize = 0;
    size_t usedCompactedOutputSize = 0;
    double compactionRatio = initialCompactionRatio;

    CuBuffer<char> d_temp;
    CuBuffer<char> d_temp_output;
    CuBuffer<size_t> d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    while( !gases.empty() )
    {
        // The estimated total output size that we end up with when using compaction.
        // It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
        // Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
        size_t remainingEstimatedTotalOutputSize =
            ( size_t )( ( totalTempOutputSize - totalTempOutputProcessedSize ) * compactionRatio );
        size_t availableMemPoolSize = remainingEstimatedTotalOutputSize + additionalAvailableMemory;
        // We need to fit the following things into availableMemPoolSize:
        // - temporary buffer for building a GAS (only during build, can be cleared before compaction)
        // - build output buffer of a GAS
        // - size (actual number) of a compacted GAS as output of a build
        // - compacted GAS

        size_t batchNGASes                    = 0;
        size_t batchBuildOutputRequirement    = 0;
        size_t batchBuildMaxTempRequirement   = 0;
        size_t batchBuildCompactedRequirement = 0;
        for( auto it = gases.rbegin(); it != gases.rend(); it++ )
        {
            batchBuildOutputRequirement += it->second.gas_buffer_sizes.outputSizeInBytes;
            batchBuildCompactedRequirement += ( size_t )( it->second.gas_buffer_sizes.outputSizeInBytes * compactionRatio );
            // roughly account for the storage of the compacted size, although that goes into a separate buffer
            batchBuildOutputRequirement += 8ull;
            // make sure that all further output pointers are 256 byte aligned
            batchBuildOutputRequirement = roundUp<size_t>( batchBuildOutputRequirement, 256ull );
            // temp buffer is shared for all builds in the batch
            batchBuildMaxTempRequirement = std::max( batchBuildMaxTempRequirement, it->second.gas_buffer_sizes.tempSizeInBytes );
            batchNGASes++;
            if( ( batchBuildOutputRequirement + batchBuildMaxTempRequirement + batchBuildCompactedRequirement ) > availableMemPoolSize )
                break;
        }

        // d_temp may still be available from a previous batch, but is freed later if it is "too big"
        d_temp.allocIfRequired( batchBuildMaxTempRequirement );

        // trash existing buffer if it is more than 10% bigger than what we need
        // if it is roughly the same, we keep it
        if( d_temp_output.capacityByteSize() > batchBuildOutputRequirement * 1.1 )
            d_temp_output.free();
        d_temp_output.allocIfRequired( batchBuildOutputRequirement );

        // this buffer is assumed to be very small
        // trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
        if( d_temp_compactedSizes.capacity() > batchNGASes * 2 && d_temp_compactedSizes.capacityByteSize() > 20 * 1024 * 1024 )
            d_temp_compactedSizes.free();
        d_temp_compactedSizes.allocIfRequired( batchNGASes );

        auto it = gases.rbegin();
        for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
        {
            emitProperty.result = d_temp_compactedSizes.getCU( i );
            GASInfo& info = it->second;

            OPTIX_CHECK( optixAccelBuild( m_context, 0,   // CUDA stream
                                            &accel_options,
                                            info.buildInputs.data(),
                                            static_cast<unsigned int>( info.buildInputs.size() ),
                                            d_temp.getCU(),
                                            d_temp.byteSize(),
                                            d_temp_output.getCU( tempOutputAlignmentOffset ),
                                            info.gas_buffer_sizes.outputSizeInBytes,
                                            &info.mesh->gas_handle,
                                            &emitProperty,  // emitted property list
                                            1               // num emitted properties
                                            ) );

            tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
            it++;
        }

        // trash d_temp if it is at least 20MB in size
        if( d_temp.capacityByteSize() > 20 * 1024 * 1024 )
            d_temp.free();

        // download all compacted sizes to allocate final output buffers for these GASes
        std::vector<size_t> h_compactedSizes( batchNGASes );
        d_temp_compactedSizes.download( h_compactedSizes.data() );

        //////////////////////////////////////////////////////////////////////////
        // TODO:
        // Now we know the actual memory requirement of the compacted GASes.
        // Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
        bool canCompact = false;
        it = gases.rbegin();
        for( size_t i = 0; i < batchNGASes; ++i )
        {
            GASInfo& info = it->second;
            if( info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i] )
            {
                canCompact = true;
                break;
            }
            it++;
        }

        // sum of size of compacted GASes
        size_t batchCompactedSize = 0;

        if( canCompact )
        {
            //////////////////////////////////////////////////////////////////////////
            // "batch allocate" the compacted buffers
            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                batchCompactedSize += h_compactedSizes[i];
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &info.mesh->d_gas_output ), h_compactedSizes[i] ) );
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
                it++;
            }

            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                OPTIX_CHECK( optixAccelCompact( m_context, 0, info.mesh->gas_handle, info.mesh->d_gas_output,
                                                h_compactedSizes[i], &info.mesh->gas_handle ) );
                it++;
            }
        }
        else
        {
            it = gases.rbegin();
            for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                info.mesh->d_gas_output = d_temp_output.getCU( tempOutputAlignmentOffset );
                batchCompactedSize += h_compactedSizes[i];
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

                tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
                it++;
            }
            d_temp_output.release();
        }

        usedCompactedOutputSize += batchCompactedSize;

        gases.erase( it.base(), gases.end() );
    }
}


void Scene::buildInstanceAccel( int rayTypeCount )
{
    const size_t num_instances = m_instances.size();

    std::vector<OptixInstance> optix_instances( num_instances );

    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_instances.size(); ++i )
    {
        auto  instance = m_instances[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = m_meshes[instance->mesh_idx]->gas_handle;
        memcpy( optix_instance.transform, instance->transform.getData(), sizeof( float ) * 12 );

        sbt_offset += static_cast<unsigned int>( m_meshes[instance->mesh_idx]->indices.size() ) * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * num_instances;
    CUdeviceptr  d_instances;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_instances ),
                optix_instances.data(),
                instances_size_in_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( num_instances );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                m_context,
                &accel_options,
                &instance_input,
                1, // num build inputs
                &ias_buffer_sizes
                ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_temp_buffer ),
                ias_buffer_sizes.tempSizeInBytes
                ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &m_d_ias_output_buffer ),
                ias_buffer_sizes.outputSizeInBytes
                ) );

    OPTIX_CHECK( optixAccelBuild(
                m_context,
                nullptr,                  // CUDA stream
                &accel_options,
                &instance_input,
                1,                  // num build inputs
                d_temp_buffer,
                ias_buffer_sizes.tempSizeInBytes,
                m_d_ias_output_buffer,
                ias_buffer_sizes.outputSizeInBytes,
                &m_ias_handle,
                nullptr,            // emitted property list
                0                   // num emitted properties
                ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
}

void Scene::createPTXModule()
{

    OptixModuleCompileOptions module_compile_options = {};
#if OPTIX_DEBUG_DEVICE_CODE
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur            = false;
    m_pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues          = whitted::NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues        = 2; // TODO
    m_pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( nullptr, nullptr, "whitted.cu", inputSize );

    m_ptx_module  = {};
    OPTIX_CHECK_LOG( optixModuleCreate(
                m_context,
                &module_compile_options,
                &m_pipeline_compile_options,
                input,
                inputSize,
                LOG, &LOG_SIZE,
                &m_ptx_module
                ) );
}


void Scene::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {};

    //
    // Ray generation
    //
    {

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &raygen_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &m_raygen_prog_group
                    )
                );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_radiance";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &m_radiance_miss_group
                    )
                );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                    m_context,
                    &miss_prog_group_desc,
                    1,                             // num program groups
                    &program_group_options,
                    LOG, &LOG_SIZE,
                    &m_occlusion_miss_group
                    )
                );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hit_prog_group_desc.hitgroup.moduleAH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                         m_context,
                         &hit_prog_group_desc,
                         1,                             // num program groups
                         &program_group_options,
                         LOG, &LOG_SIZE,
                         &m_radiance_hit_group
                         )
                );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleAH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                         m_context,
                         &hit_prog_group_desc,
                         1,                             // num program groups
                         &program_group_options,
                         LOG, &LOG_SIZE,
                         &m_occlusion_hit_group
                         )
                );
    }
}


void Scene::createPipeline()
{
    OptixProgramGroup program_groups[] =
    {
        m_raygen_prog_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = whitted::MAX_TRACE_DEPTH;

    OPTIX_CHECK_LOG( optixPipelineCreate(
                m_context,
                &m_pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                LOG, &LOG_SIZE,
                &m_pipeline
                ) );
}


void Scene::createSBT()
{
    {
        const size_t raygen_record_size = sizeof( EmptyRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_sbt.raygenRecord ), raygen_record_size ) );

        EmptyRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( m_raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.raygenRecord ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );
    }

    {
        const size_t miss_record_size = sizeof( EmptyRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_sbt.missRecordBase ),
                    miss_record_size*whitted::RAY_TYPE_COUNT
                    ) );

        EmptyRecord ms_sbt[ whitted::RAY_TYPE_COUNT ];
        OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_miss_group,  &ms_sbt[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_miss_group, &ms_sbt[1] ) );

        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.missRecordBase ),
                    ms_sbt,
                    miss_record_size*whitted::RAY_TYPE_COUNT,
                    cudaMemcpyHostToDevice
                    ) );
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        m_sbt.missRecordCount     = whitted::RAY_TYPE_COUNT;
    }

    {
        std::vector<HitGroupRecord> hitgroup_records;
        for( const auto& instance : m_instances )
        {
            const auto mesh = m_meshes[instance->mesh_idx];
            for( size_t i = 0; i < mesh->material_idx.size(); ++i )
            {
                HitGroupRecord rec = {};
                OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_hit_group, &rec ) );
                GeometryData::TriangleMesh triangle_mesh = {};
                triangle_mesh.normals   = mesh->normals[i];
                triangle_mesh.positions = mesh->positions[i];
                for( size_t j = 0; j < GeometryData::num_texcoords; ++j )
                    triangle_mesh.texcoords[j] = mesh->texcoords[j][i];
                triangle_mesh.colors    = mesh->colors[i];
                triangle_mesh.indices   = mesh->indices[i];
                rec.data.geometry_data.setTriangleMesh( triangle_mesh );

                const int32_t mat_idx  = mesh->material_idx[i];
                if( mat_idx >= 0 )
                    rec.data.material_data = m_materials[ mat_idx ];
                else
                    rec.data.material_data = MaterialData();
                hitgroup_records.push_back( rec );

                OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_hit_group, &rec ) );
                hitgroup_records.push_back( rec );
            }
        }

        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_sbt.hitgroupRecordBase ),
                    hitgroup_record_size*hitgroup_records.size()
                    ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.hitgroupRecordBase ),
                    hitgroup_records.data(),
                    hitgroup_record_size*hitgroup_records.size(),
                    cudaMemcpyHostToDevice
                    ) );

        m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( hitgroup_record_size );
        m_sbt.hitgroupRecordCount         = static_cast<unsigned int>( hitgroup_records.size() );
    }
}

} // namespace sutil
