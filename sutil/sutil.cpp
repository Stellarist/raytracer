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


#include <sampleConfig.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/PPMLoader.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <tinygltf/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/stb_image_write.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    include<windows.h>
#    include<mmsystem.h>
#else
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif


namespace sutil
{

static void errorCallback( int error, const char* description )
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
}


static void savePPM( const unsigned char* Pix, const char* fname, int wid, int hgt, int chan )
{
    if( Pix == NULL || wid < 1 || hgt < 1 )
        throw Exception( "savePPM: Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw Exception( "savePPM: Attempting to save image with channel count != 1, 3, or 4." );

    std::ofstream OutFile( fname, std::ios::out | std::ios::binary );
    if( !OutFile.is_open() )
        throw Exception( "savePPM: Could not open file for" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ( ( chan == 1 ? ( is_float ? 'Z' : '5' ) : ( chan == 3 ? ( is_float ? '7' : '6' ) : '8' ) ) )
            << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write( reinterpret_cast<char*>( const_cast<unsigned char*>( Pix ) ), wid*hgt*chan*( is_float ? 4 : 1 ) );
    OutFile.close();
}


static bool dirExists( const char* path )
{
#if defined( _WIN32 )
    DWORD attrib = GetFileAttributes( path );
    return ( attrib != INVALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return false;

    closedir( dir );
    return true;
#endif
}

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static std::string existingFilePath( const char* directory, const char* relativeSubDir, const char* relativePath )
{
    std::string path = directory ? directory : "";
    if( relativeSubDir )
    {
        path += '/';
        path += relativeSubDir;
    }
    if( relativePath )
    {
        path += '/';
        path += relativePath;
    }
    return fileExists( path ) ? path : "";
}

std::string getSampleDir()
{
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory && dirExists( directory ) )
            return directory;
    }

    throw Exception( "sutil::getSampleDir couldn't locate an existing sample directory" );
}

const char* sampleFilePath( const char* relativeSubDir, const char* relativePath )
{
    static std::string s;

    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory )
        {
            s = existingFilePath( directory, relativeSubDir, relativePath );
            if( !s.empty() )
            {
                return s.c_str();
            }
        }
    }
    throw Exception( ( std::string{ "sutil::sampleDataFilePath couldn't locate " } +relativePath ).c_str() );
}

const char* sampleDataFilePath( const char* relativePath )
{
    return sampleFilePath( "data", relativePath );
}

void ImageBuffer::destroy()
{
    switch( pixel_format )
    {
        case UNSIGNED_BYTE4:
            delete[] reinterpret_cast<uint4*>( data );
            break;
        case FLOAT4:
            delete[] reinterpret_cast<float4*>( data );
            break;
        case FLOAT3:
            delete[] reinterpret_cast<float3*>( data );
            break;
    }
    data   = nullptr;
    width  = 0;
    height = 0;
}

size_t pixelFormatSize( BufferImageFormat format )
{
    switch( format )
    {
        case BufferImageFormat::UNSIGNED_BYTE4:
            return sizeof( char ) * 4;
        case BufferImageFormat::FLOAT3:
            return sizeof( float ) * 3;
        case BufferImageFormat::FLOAT4:
            return sizeof( float ) * 4;
        default:
            throw Exception( "sutil::pixelFormatSize: Unrecognized buffer format" );
    }
}


Texture loadTexture( const char* fname, float3 default_color, cudaTextureDesc* tex_desc )
{
    const std::string filename( fname );
    bool   isHDR = false;
    size_t len   = filename.length();
    if( len >= 3 )
    {
        isHDR = ( filename[len - 3] == 'H' || filename[len - 3] == 'h' ) &&
                ( filename[len - 2] == 'D' || filename[len - 2] == 'd' ) &&
		        ( filename[len - 1] == 'R' || filename[len - 1] == 'r' );
    }
    if( isHDR )
    {
        std::cerr << "HDR texture loading not yet implemented" << std::endl;
        return {};
    }
    else
    {
        return loadPPMTexture( filename, default_color, tex_desc );
    }
}


ImageBuffer loadImage( const char* fname, int32_t force_components )
{
    const std::string filename( fname );

    if( !fileExists( fname ) )
        throw Exception( ( std::string{ "sutil::loadImage(): File does not exist: " } + filename ).c_str() );

    if( filename.length() < 5 )
        throw Exception( "sutil::loadImage(): Failed to determine filename extension" );

    if( force_components >  4 ||
        force_components == 2 ||
        force_components == 1 )
        throw Exception( "sutil::loadImage(): Invalid force_components value" );

    ImageBuffer image;

    const std::string ext = filename.substr( filename.length()-3 );
    if( ext == "PPM" || ext == "ppm" )
    {
        if( force_components != 4 && force_components != 0 )
            throw Exception( "sutil::loadImage(): PPM loading with force_components not implemented" );

        PPMLoader loader( filename );
        image.width  = loader.width();
        image.height = loader.height();
        image.data   = new uchar4[ image.width*image.height ];
        for( int32_t i = 0; i < static_cast<int32_t>( image.width*image.height ); ++i )
        {
            // convert to rgba
            reinterpret_cast<uchar4*>( image.data )[i].x = loader.raster()[i*3+0];
            reinterpret_cast<uchar4*>( image.data )[i].y = loader.raster()[i*3+1];
            reinterpret_cast<uchar4*>( image.data )[i].z = loader.raster()[i*3+2];
            reinterpret_cast<uchar4*>( image.data )[i].w = 255;
        }
        image.pixel_format = UNSIGNED_BYTE4;
    }
    else if( ext == "png" || ext == "PNG" )
    {
        if( force_components != 4 && force_components != 0 )
            throw Exception( "sutil::loadImage(): PNG loading with force_components not implemented" );

        int32_t w, h, channels;
        uint8_t* data = stbi_load( filename.c_str(), &w, &h, &channels, STBI_rgb_alpha );
        if( !data )
            throw sutil::Exception( "sutil::loadImage( png ): stbi_load failed" );

        image.width  = w;
        image.height = h;
        image.data   = new uchar4[ w*h ];
        image.pixel_format = UNSIGNED_BYTE4;
        memcpy( image.data, data, w*h*STBI_rgb_alpha );

        stbi_image_free( data );
    }
    else if( ext == "exr" || ext == "EXR" )
    {
        if( force_components != 4 && force_components != 0 && force_components != 3 )
            throw Exception( "sutil::loadImage(): PNG loading with force_components not implemented" );

        const char*  err  = nullptr;
        float*       data = nullptr;
        int32_t w, h;
        int32_t res = LoadEXR( &data, &w, &h, filename.c_str(), &err);

        if( res != TINYEXR_SUCCESS)
        {
            if (err)
            {
                sutil::Exception e( ( std::string( "sutil::loadImage( exr ): " ) + err ).c_str() );
                FreeEXRErrorMessage( err );
                throw e;
            }
            else
            {
                throw sutil::Exception( "sutil::loadImage( exr ): failed to load image" );
            }
        }

        image.width  = w;
        image.height = h;
        if( force_components == 4 || force_components == 0 )
        {
            image.data   = new float4[ image.width*image.height ];
            image.pixel_format = FLOAT4;
            memcpy( image.data, data, sizeof(float)*4*w*h );
        }
        else // force_components == 3
        {
            image.data   = new float3[ image.width*image.height ];
            image.pixel_format = FLOAT3;
            for( int32_t i = 0; i < static_cast<int32_t>( image.width*image.height ); ++i )
            {
                reinterpret_cast<float3*>( image.data )[i].x = data[i*4+0];
                reinterpret_cast<float3*>( image.data )[i].y = data[i*4+1];
                reinterpret_cast<float3*>( image.data )[i].z = data[i*4+2];
            }
        }

        free( data );
    }
    else
    {
        throw Exception( ( "sutil::loadImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }

    return image;
}


void initGL()
{
    if( !gladLoadGL() )
        throw Exception( "Failed to initialize GL" );

    GL_CHECK( glClearColor( 0.212f, 0.271f, 0.31f, 1.0f ) );
    GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) );
}

void initGLFW()
{
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw Exception( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_VISIBLE, GLFW_FALSE );

    GLFWwindow* window = glfwCreateWindow( 64, 64, "", nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync
}

GLFWwindow* initGLFW( const char* window_title, int width, int height )
{
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw Exception( "Failed to initialize GLFW" );

    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( width, height, window_title, nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );

    glfwMakeContextCurrent( window );
    glfwSwapInterval( 0 );  // No vsync

    return window;
}


void initImGui( GLFWwindow* window )
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui_ImplGlfw_InitForOpenGL( window, false );
    ImGui_ImplOpenGL3_Init();
    ImGui::StyleColorsDark();

    // Scale font size by the window's DPI.
    float xscale, yscale;
    glfwGetWindowContentScale( window, &xscale, &yscale );

    const float DEFAULT_FONT_SIZE = 13.0f;  // See AddFontDefault in imgui.cpp

    ImFontConfig config{};
    config.SizePixels = DEFAULT_FONT_SIZE * yscale;
    io.Fonts->AddFontDefault( &config );

    ImGui::GetStyle().WindowBorderSize = 0.0f;
}


GLFWwindow* initUI( const char* window_title, int width, int height )
{
    GLFWwindow* window = initGLFW( window_title, width, height );
    initGL();
    initImGui( window );
    return window;
}


void cleanupUI( GLFWwindow* window )
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow( window );
    glfwTerminate();
}


void beginFrameImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}


void endFrameImGui()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}


void displayBufferWindow( const char* title, const ImageBuffer& buffer )
{
    //
    // Initialize GLFW state
    //
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback( errorCallback );
    if( !glfwInit() )
        throw Exception( "Failed to initialize GLFW" );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );  // To make Apple happy -- should not be needed
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    window = glfwCreateWindow( buffer.width, buffer.height, title, nullptr, nullptr );
    if( !window )
        throw Exception( "Failed to create GLFW window" );
    glfwMakeContextCurrent( window );
    glfwSetKeyCallback( window, keyCallback );


    //
    // Initialize GL state
    //
    initGL();
    GLDisplay display( buffer.pixel_format );

    GLuint pbo = 0u;
    GL_CHECK( glGenBuffers( 1, &pbo ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, pbo ) );
    GL_CHECK( glBufferData( GL_ARRAY_BUFFER, pixelFormatSize( buffer.pixel_format ) * buffer.width * buffer.height,
                            buffer.data, GL_STREAM_DRAW ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );

    //
    // Display loop
    //
    int framebuf_res_x = 0, framebuf_res_y = 0;
    do
    {
        glfwWaitEvents();
        glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
        display.display( buffer.width, buffer.height, framebuf_res_x, framebuf_res_y, pbo );
        glfwSwapBuffers( window );
    } while( !glfwWindowShouldClose( window ) );

    glfwDestroyWindow( window );
    glfwTerminate();
}


static float toSRGB( float c )
{
    float invGamma = 1.0f / 2.4f;
    float powed    = std::pow( c, invGamma );
    return c < 0.0031308f ? 12.92f * c : 1.055f * powed - 0.055f;
}

void saveImage( const char* fname, const ImageBuffer& image, bool disable_srgb_conversion )
{
    const std::string filename( fname );
    if( filename.length() < 5 )
        throw Exception( "sutil::saveImage(): Failed to determine filename extension" );

    const std::string ext = filename.substr( filename.length()-3 );
    if( ext == "PPM" || ext == "ppm" )
    {
        //
        // Note -- we are flipping image vertically as we write it into output buffer
        //
        const int32_t width  = image.width;
        const int32_t height = image.height;
        std::vector<unsigned char> pix( width*height*3 );

        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        pix[ dst_idx+0] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+0 ];
                        pix[ dst_idx+1] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+1 ];
                        pix[ dst_idx+2] = reinterpret_cast<uint8_t*>( image.data )[ src_idx+2 ];
                    }
                }
            } break;

            case BufferImageFormat::FLOAT3:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 3*width*j            + 3*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            case BufferImageFormat::FLOAT4:
            {
                for( int j = height - 1; j >= 0; --j )
                {
                    for( int i = 0; i < width; ++i )
                    {
                        const int32_t dst_idx = 3*width*(height-j-1) + 3*i;
                        const int32_t src_idx = 4*width*j            + 4*i;
                        for( int elem = 0; elem < 3; ++elem )
                        {
                            const float   f = reinterpret_cast<float*>( image.data )[src_idx+elem ];
                            const int32_t v = static_cast<int32_t>( 256.0f*(disable_srgb_conversion ? f : toSRGB(f)) );
                            const int32_t c =  v < 0 ? 0 : v > 0xff ? 0xff : v;
                            pix[ dst_idx+elem ] = static_cast<uint8_t>( c );
                        }
                    }
                }
            } break;

            default:
            {
                throw Exception( "sutil::saveImage(): Unrecognized image buffer pixel format.\n" );
            }
        }

        savePPM( pix.data(), filename.c_str(), width, height, 3 );
    }

    else if(  ext == "PNG" || ext == "png" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                stbi_flip_vertically_on_write( true );
                if( !stbi_write_png(
                            filename.c_str(),
                            image.width,
                            image.height,
                            4, // components,
                            image.data,
                            image.width*sizeof( uchar4 ) //stride_in_bytes
                            ) )
                    throw Exception( "sutil::saveImage(): stbi_write_png failed" );
            } break;

            case BufferImageFormat::FLOAT3:
            {
                throw Exception( "sutil::saveImage(): saving of float3 images to PNG not implemented yet" );
            }

            case BufferImageFormat::FLOAT4:
            {
                throw Exception( "sutil::saveImage(): saving of float4 images to PNG not implemented yet" );
            }

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    }

    else if(  ext == "EXR" || ext == "exr" )
    {
        switch( image.pixel_format )
        {
            case BufferImageFormat::UNSIGNED_BYTE4:
            {
                throw Exception( "sutil::saveImage(): saving of uchar4 images to EXR not implemented yet" );
            }

            case BufferImageFormat::FLOAT3:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        3, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );

            } break;

            case BufferImageFormat::FLOAT4:
            {
                const char* err;
                int32_t ret = SaveEXR(
                        reinterpret_cast<float*>( image.data ),
                        image.width,
                        image.height,
                        4, // num components
                        static_cast<int32_t>( true ), // save_as_fp16
                        filename.c_str(),
                        &err );

                if( ret != TINYEXR_SUCCESS )
                    throw Exception( ( "sutil::saveImage( exr ) error: " + std::string( err ) ).c_str() );
            } break;

            default:
            {
                throw Exception( "sutil::saveImage: Unrecognized image buffer pixel format.\n" );
            }
        }
    }
    else
    {
        throw Exception( ( "sutil::saveImage(): Failed unsupported filetype '" + ext + "'" ).c_str() );
    }
}


void displayFPS( unsigned int frame_count )
{
    constexpr std::chrono::duration<double> display_update_min_interval_time( 0.5 );
    static double                           fps              = -1.0;
    static unsigned                         last_frame_count = 0;
    static auto                             last_update_time = std::chrono::steady_clock::now();
    auto                                    cur_time         = std::chrono::steady_clock::now();

    if( cur_time - last_update_time > display_update_min_interval_time )
    {
        fps = ( frame_count - last_frame_count ) / std::chrono::duration<double>( cur_time - last_update_time ).count();
        last_frame_count = frame_count;
        last_update_time = cur_time;
    }
    if( frame_count > 0 && fps >= 0.0 )
    {
        static char fps_text[32];
        sprintf( fps_text, "fps: %7.2f", fps );
        displayText( fps_text, 10.0f, 10.0f );
    }
}


void displayStats( std::chrono::duration<double>& state_update_time,
                          std::chrono::duration<double>& render_time,
                          std::chrono::duration<double>& display_time )
{
    constexpr std::chrono::duration<double> display_update_min_interval_time( 0.5 );
    static int32_t                          total_subframe_count = 0;
    static int32_t                          last_update_frames   = 0;
    static auto                             last_update_time     = std::chrono::steady_clock::now();
    static char                             display_text[128];

    const auto cur_time = std::chrono::steady_clock::now();

    beginFrameImGui();
    last_update_frames++;

    typedef std::chrono::duration<double, std::milli> durationMs;

    if( cur_time - last_update_time > display_update_min_interval_time || total_subframe_count == 0 )
    {
        sprintf( display_text,
                 "%5.1f fps\n\n"
                 "state update: %8.1f ms\n"
                 "render      : %8.1f ms\n"
                 "display     : %8.1f ms\n",
                 last_update_frames / std::chrono::duration<double>( cur_time - last_update_time ).count(),
                 ( durationMs( state_update_time ) / last_update_frames ).count(),
                 ( durationMs( render_time ) / last_update_frames ).count(),
                 ( durationMs( display_time ) / last_update_frames ).count() );

        last_update_time   = cur_time;
        last_update_frames = 0;
        state_update_time = render_time = display_time = std::chrono::duration<double>::zero();
    }
    displayText( display_text, 10.0f, 10.0f );
    endFrameImGui();

    ++total_subframe_count;
}


void displayText( const char* text, float x, float y )
{
    ImGui::SetNextWindowBgAlpha( 0.0f );
    ImGui::SetNextWindowPos( ImVec2( x, y ) );
    ImGui::Begin( "TextOverlayFG", nullptr,
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove
                      | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs );
    ImGui::TextColored( ImColor( 0.7f, 0.7f, 0.7f, 1.0f ), "%s", text );
    ImGui::End();
}

void buildRadioButtons( const char* options[], const int numOptions, float x, float y, int& selected_option )
{
    ImGui::SetNextWindowBgAlpha( 0.0f );
    ImGui::SetNextWindowPos( ImVec2( x, y ) );
    ImGui::Begin( "options", nullptr,
                  ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
                      | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings );

    int current_item = selected_option;
    for (int i = 0; i < numOptions; i++)
    {
        ImGui::RadioButton(options[i], &current_item, i);
        {
            selected_option = current_item;
        }
    }
    ImGui::End();
}


void parseDimensions( const char* arg, int& width, int& height )
{
    // look for an 'x': <width>x<height>
    size_t width_end    = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char* height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }
    const std::string err = "Failed to parse width, height from string '" + std::string( arg ) + "'";
    throw std::invalid_argument( err.c_str() );
}

double currentTime()
{
    return std::chrono::duration_cast< std::chrono::duration< double > >
        ( std::chrono::high_resolution_clock::now().time_since_epoch() ).count();
}


#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) ); \
    } while( 0 )

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

// Returns string of file extension including '.'
static std::string fileExtensionForLoading()
{
    std::string extension;
#if SAMPLES_INPUT_GENERATE_PTX
    extension = ".ptx";
#endif
#if SAMPLES_INPUT_GENERATE_OPTIXIR
    extension = ".optixir";
#endif
    if( const char* ext = getenv("OPTIX_SAMPLES_INPUT_EXTENSION") )
    {
        extension = ext;
        if( extension.size() && extension[0] != '.' )
            extension = "." + extension;
    }
    return extension;
}

#if CUDA_NVRTC_ENABLED

static void getCuStringFromFile( std::string& cu, std::string& location, const char* sampleDir, const char* filename )
{
    std::vector<std::string> source_locations;

    const std::string base_dir = getSampleDir();

    // Potential source locations (in priority order)
    if( sampleDir )
        source_locations.push_back( base_dir + '/' + sampleDir + '/' + filename );
    source_locations.push_back( base_dir + "/cuda/" + filename );

    for( const std::string& loc : source_locations )
    {
        // Try to get source code from file
        if( readSourceFile( cu, loc ) )
        {
            location = loc;
            return;
        }
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error( "Couldn't open source file " + std::string( filename ) );
}

static std::string g_nvrtcLog;

static void getInputFromCuString( std::string&                    input,
                                  const char*                     sample_directory,
                                  const char*                     cu_source,
                                  const char*                     name,
                                  const char**                    log_string,
                                  const std::vector<const char*>& compiler_options )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = getSampleDir();

    // Set sample dir as the primary include path
    std::string sample_dir;
    if( sample_directory )
    {
        sample_dir = std::string( "-I" ) + base_dir + '/' + sample_directory;
        options.push_back( sample_dir.c_str() );
    }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {SAMPLES_ABSOLUTE_INCLUDE_DIRS};
    const char*              rel_dirs[] = {SAMPLES_RELATIVE_INCLUDE_DIRS};

    for( const char* dir : abs_dirs )
    {
        include_dirs.push_back( std::string( "-I" ) + dir );
    }
    for( const char* dir : rel_dirs )
    {
        include_dirs.push_back( "-I" + base_dir + '/' + dir );
    }
    for( const std::string& dir : include_dirs)
    {
        options.push_back( dir.c_str() );
    }

    bool optixir = fileExtensionForLoading() == ".optixir";
    if( optixir )
        options.push_back( "--optix-ir" );

    // Collect NVRTC options
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to OPTIXIR/PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve OPTIXIR/PTX code
    size_t input_size = 0;
    if( optixir )
    {
#if CUDA_VERSION >= 12000
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIRSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIR( prog, &input[0] ) );
#else
        throw std::runtime_error( "OptiX IR support for NVRTC is only available with CUDA 12.0+" );
#endif
    }
    else
    {
        NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &input[0] ) );
    }

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}

#else  // CUDA_NVRTC_ENABLED

static std::string sampleInputFilePath( const char* sampleName, const char* fileName )
{
    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_PTX_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_PTX_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" ),
#if defined(OPTIX_OPTIXIR_BUILD_CONFIGURATION)
        SAMPLES_PTX_DIR "/" OPTIX_OPTIXIR_BUILD_CONFIGURATION,
#else
#error "OPTIX_OPTIXIR_BUILD_CONFIGURATION not defined."
#endif
        SAMPLES_PTX_DIR,
        "."
    };

    // Allow overriding the file extension
    std::string extension = fileExtensionForLoading();

    if( !sampleName )
        sampleName = "sutil";
    std::vector<std::string> locations;
    for( const char* directory : directories )
    {
        if( directory )
        {
            std::string path = directory;
            path += '/';
            path += sampleName;
            path += "_generated_";
            path += fileName;
            path += extension;
            locations.push_back( path );
            if( fileExists( path ) )
                return path;
        }
    }

    std::string error = "sutil::sampleInputFilePath couldn't locate " + std::string( fileName ) + " for sample "
                        + std::string( sampleName ) + " in the following locations:\n";
    for( const auto& path : locations )
        error += "\t" + path + "\n";

    throw Exception( error.c_str() );
}

static void getInputDataFromFile( std::string& inputData, const char* sample_name, const char* filename )
{
    const std::string sourceFilePath = sampleInputFilePath( sample_name, filename );

    // Try to open source file
    if( !readSourceFile( inputData, sourceFilePath ) )
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error( err.c_str() );
    }
}

#endif  // CUDA_NVRTC_ENABLED

struct SourceCache
{
    std::map<std::string, std::string*> map;
    ~SourceCache()
    {
        for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;
    }
};
static SourceCache g_sourceCache;

const char* getInputData( const char*                     sample,
                          const char*                     sampleDir,
                          const char*                     filename,
                          size_t&                         dataSize,
                          const char**                    log,
                          const std::vector<const char*>& compilerOptions )
{
    if( log )
        *log = NULL;

    std::string *                                 inputData, cu;
    std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string*>::iterator elem = g_sourceCache.map.find( key );

    if( elem == g_sourceCache.map.end() )
    {
        inputData = new std::string();
#if CUDA_NVRTC_ENABLED
        std::string location;
        getCuStringFromFile( cu, location, sampleDir, filename );
        getInputFromCuString( *inputData, sampleDir, cu.c_str(), location.c_str(), log, compilerOptions );
#else
        getInputDataFromFile( *inputData, sample, filename );
#endif
        g_sourceCache.map[key] = inputData;
    }
    else
    {
        inputData = elem->second;
    }
    dataSize = inputData->size();
    return inputData->c_str();
}

void ensureMinimumSize( int& w, int& h )
{
    if( w <= 0 )
        w = 1;
    if( h <= 0 )
        h = 1;
}

void ensureMinimumSize( unsigned& w, unsigned& h )
{
    if( w == 0 )
        w = 1;
    if( h == 0 )
        h = 1;
}

void reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
#if defined( _WIN32 ) && defined( RELEASE_PUBLIC )
    {
        char s[2048];
        sprintf( s, "OptiX Error: %s", message );
        MessageBoxA( 0, s, "OptiX Error", MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL );
    }
#endif
}

} // namespace sutil
