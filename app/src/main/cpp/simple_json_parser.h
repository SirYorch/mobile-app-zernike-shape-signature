#ifndef SIMPLE_JSON_PARSER_H
#define SIMPLE_JSON_PARSER_H

#include <string>
#include <vector>
#include <sstream>
#include <fstream>

// Parser JSON simple para cargar el modelo SVM
class SimpleJSONParser {
public:
    static std::vector<double> parseDoubleArray(const std::string& line) {
        std::vector<double> result;
        std::string cleaned = line;
        
        // Remover corchetes y espacios
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
        
        std::stringstream ss(cleaned);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                result.push_back(std::stod(item));
            }
        }
        
        return result;
    }
    
    static std::vector<std::string> parseStringArray(const std::string& line) {
        std::vector<std::string> result;
        std::string cleaned = line;
        
        // Buscar strings entre comillas
        size_t pos = 0;
        while ((pos = cleaned.find('"', pos)) != std::string::npos) {
            size_t end = cleaned.find('"', pos + 1);
            if (end != std::string::npos) {
                result.push_back(cleaned.substr(pos + 1, end - pos - 1));
                pos = end + 1;
            } else {
                break;
            }
        }
        
        return result;
    }
    
    static double parseDouble(const std::string& line) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string value = line.substr(pos + 1);
            value.erase(std::remove(value.begin(), value.end(), ','), value.end());
            value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
            return std::stod(value);
        }
        return 0.0;
    }
    
    static int parseInt(const std::string& line) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string value = line.substr(pos + 1);
            value.erase(std::remove(value.begin(), value.end(), ','), value.end());
            value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
            return std::stoi(value);
        }
        return 0;
    }
};

#endif // SIMPLE_JSON_PARSER_H