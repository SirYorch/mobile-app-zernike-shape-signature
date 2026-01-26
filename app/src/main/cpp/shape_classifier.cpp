#include "shape_classifier.h"
#include <android/log.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>

#define LOG_TAG "ShapeClassifier"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// Librería JSON simple (o usar nlohmann/json)
#include <nlohmann/json.hpp>
using json = nlohmann::json;

ShapeClassifier::ShapeClassifier() : modelLoaded(false) {}

int ShapeClassifier::factorial(int n) {
    if (n <= 1) return 1;
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

double ShapeClassifier::radialPoly(int n, int m, double rho) {
    if (rho > 1.0) return 0.0;
    
    double sum = 0.0;
    for (int s = 0; s <= (n - abs(m)) / 2; s++) {
        int num = pow(-1, s) * factorial(n - s);
        int den = factorial(s) * factorial((n + abs(m)) / 2 - s) * 
                  factorial((n - abs(m)) / 2 - s);
        sum += (double)num / den * pow(rho, n - 2 * s);
    }
    return sum;
}

double ShapeClassifier::zernikePoly(int n, int m, double rho, double theta) {
    double R = radialPoly(n, abs(m), rho);
    
    if (m >= 0) {
        return R * cos(m * theta);
    } else {
        return R * sin(abs(m) * theta);
    }
}

std::vector<double> ShapeClassifier::extractZernikeMoments(
    const cv::Mat& binary, int degree) {
    
    std::vector<double> moments;
    
    int rows = binary.rows;
    int cols = binary.cols;
    int radius = std::min(rows, cols) / 2;
    
    int cx = cols / 2;
    int cy = rows / 2;
    
    // Normalizar imagen a [0,1]
    cv::Mat normalized;
    binary.convertTo(normalized, CV_32F, 1.0/255.0);
    
    // Calcular momentos de Zernike
    for (int n = 0; n <= degree; n++) {
        for (int m = -n; m <= n; m++) {
            if ((n - abs(m)) % 2 != 0) continue;
            
            double real_sum = 0.0;
            double imag_sum = 0.0;
            int count = 0;
            
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    double dx = x - cx;
                    double dy = y - cy;
                    double rho = sqrt(dx*dx + dy*dy) / radius;
                    
                    if (rho > 1.0) continue;
                    
                    double theta = atan2(dy, dx);
                    double intensity = normalized.at<float>(y, x);
                    
                    double Z = zernikePoly(n, m, rho, theta);
                    
                    real_sum += intensity * Z * cos(m * theta);
                    imag_sum += intensity * Z * sin(m * theta);
                    count++;
                }
            }
            
            if (count > 0) {
                double magnitude = sqrt(real_sum*real_sum + imag_sum*imag_sum);
                magnitude *= (n + 1) / M_PI;
                moments.push_back(magnitude);
            }
        }
    }
    
    LOGD("Momentos de Zernike extraídos: %zu features", moments.size());
    return moments;
}

cv::Mat ShapeClassifier::preprocessImage(const cv::Mat& input) {
    cv::Mat gray, binary;
    
    if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_RGBA2GRAY);
    } else if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_RGB2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Binarización simple (como en Python)
    cv::threshold(gray, binary, 150, 255, cv::THRESH_BINARY_INV);
    
    return binary;
}

bool ShapeClassifier::loadSVMModel(const std::string& modelPath) {
    LOGD("Cargando modelo SVM desde: %s", modelPath.c_str());
    
    std::ifstream file(modelPath);
    if (!file.is_open()) {
        LOGD("Error: No se pudo abrir el archivo del modelo");
        return false;
    }

    json root;
    file >> root;
    
    // Cargar support vectors
    for (const auto& sv : root["support_vectors"]) {
        std::vector<double> vec;
        for (const auto& val : sv) {
            vec.push_back(val.get<double>());
        }
        model.support_vectors.push_back(vec);
    }
    
    // Cargar dual coefficients
    for (const auto& coef_array : root["dual_coef"]) {
        std::vector<double> coefs;
        for (const auto& val : coef_array) {
            coefs.push_back(val.get<double>());

        }
        model.dual_coef.push_back(coefs);
    }
    
    // Cargar intercept
    for (const auto& val : root["intercept"]) {
        model.intercept.push_back(val.get<double>());
    }
    
    // Cargar classes
    for (const auto& cls : root["classes"]) {
        model.classes.push_back(cls.get<std::string>());
    }
    
    // Cargar gamma
    model.gamma = root["gamma"].get<double>();



    // Cargar scaler
    for (const auto& val : root["scaler_mean"]) {
        model.scaler_mean.push_back(val.get<double>());
    }
    for (const auto& val : root["scaler_scale"]) {
        model.scaler_scale.push_back(val.get<double>());
    }

    model.zernike_degree = root["degree"].get<int>();


    modelLoaded = true;
    LOGD("Modelo cargado exitosamente");
    LOGD("  Support Vectors: %zu", model.support_vectors.size());
    LOGD("  Classes: %zu", model.classes.size());
    LOGD("  Gamma: %f", model.gamma);
    
    return true;
}

std::vector<double> ShapeClassifier::normalizeFeatures(
    const std::vector<double>& features) {
    
    std::vector<double> normalized(features.size());
    
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = (features[i] - model.scaler_mean[i]) / model.scaler_scale[i];
    }
    
    return normalized;
}

double ShapeClassifier::rbfKernel(const std::vector<double>& x1,
                                   const std::vector<double>& x2,
                                   double gamma) {
    double sum = 0.0;
    for (size_t i = 0; i < x1.size(); i++) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return exp(-gamma * sum);
}

std::vector<double> ShapeClassifier::decisionFunction(
    const std::vector<double>& features) {
    
    int n_classes = model.classes.size();
    std::vector<double> decisions;
    
    // Para clasificación multiclase (one-vs-one)
    int k = 0;
    for (int i = 0; i < n_classes; i++) {
        for (int j = i + 1; j < n_classes; j++) {
            double decision = 0.0;
            
            // Calcular kernel con cada support vector
            for (size_t sv_idx = 0; sv_idx < model.support_vectors.size(); sv_idx++) {
                double kernel_val = rbfKernel(features, 
                                             model.support_vectors[sv_idx],
                                             model.gamma);
                decision += model.dual_coef[i][sv_idx] * kernel_val;
            }
            
            decision += model.intercept[k];
            decisions.push_back(decision);
            k++;
        }
    }
    
    return decisions;
}

std::string ShapeClassifier::predictSVM(const std::vector<double>& features,
                                        double& confidence) {
    if (!modelLoaded) {
        LOGD("Error: Modelo no cargado");
        return "error";
    }
    
    // Normalizar features
    std::vector<double> normalized = normalizeFeatures(features);
    
    // Calcular decisiones
    std::vector<double> decisions = decisionFunction(normalized);
    
    // Voting (one-vs-one)
    int n_classes = model.classes.size();
    std::vector<int> votes(n_classes, 0);
    
    int k = 0;
    for (int i = 0; i < n_classes; i++) {
        for (int j = i + 1; j < n_classes; j++) {
            if (decisions[k] > 0) {
                votes[i]++;
            } else {
                votes[j]++;
            }
            k++;
        }
    }
    
    // Encontrar clase con más votos
    int max_votes = 0;
    int best_class = 0;
    for (int i = 0; i < n_classes; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            best_class = i;
        }
    }
    
    // Calcular confianza
    confidence = (double)max_votes / (n_classes - 1);
    
    LOGD("Predicción: %s (confianza: %.2f)", 
         model.classes[best_class].c_str(), confidence);
    
    return model.classes[best_class];
}

std::string ShapeClassifier::classify(const cv::Mat& image, float& confidence) {
    if (!modelLoaded) {
        LOGD("Error: Modelo SVM no cargado");
        confidence = 0.0f;
        return "error";
    }
    
    // Preprocesar
    cv::Mat binary = preprocessImage(image);
    
    // Extraer momentos de Zernike
    std::vector<double> zernike = extractZernikeMoments(binary, 
                                                        model.zernike_degree);
    
    if (zernike.empty()) {
        LOGD("Error: No se pudieron extraer momentos de Zernike");
        confidence = 0.0f;
        return "desconocido";
    }
    
    // Clasificar con SVM
    double conf_double = 0.0;
    std::string label = predictSVM(zernike, conf_double);
    confidence = (float)conf_double;
    
    return label;
}