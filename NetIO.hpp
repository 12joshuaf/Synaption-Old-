#pragma once

#include "Net.hpp"
#include <string>
#include <fstream>


namespace nn {

    class NetIO {
    public:
        static void save_to_file(const Net& net, const std::string& filename);
        static void load_from_file(const std::string& filename);
    };

}
