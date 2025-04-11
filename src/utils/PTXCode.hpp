#pragma once

#include <string>
#include <fstream>
#include <stdexcept>

struct PTXCode {
    std::string code;

    PTXCode() = default;
    PTXCode(const std::string& ptx_file)
    {
        code = readFromFile(ptx_file);
    }

    void getCode(std::string& out_code) const
    {
        out_code = code;
    }

    std::string getCode() const
    {
        return code;
    }

    static std::string readFromFile(const std::string& ptx_file)
    {
        std::ifstream file(ptx_file);
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }

    static void readFromFile(const std::string& ptx_file, std::string& out_code)
    {
        out_code = readFromFile(ptx_file);
    }
};
