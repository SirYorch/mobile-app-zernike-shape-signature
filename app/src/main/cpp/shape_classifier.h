#ifndef SHAPE_CLASSIFIER_H
#define SHAPE_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <complex>

// Estructura para almacenar el modelo SVM
struct SVMModel {
    std::vector<std::vector<double>> support_vectors;
    std::vector<std::vector<double>> dual_coef;
    std::vector<double> intercept;
    std::vector<std::string> classes;
    double gamma;
    std::vector<int> n_support;
    std::vector<double> scaler_mean;
    std::vector<double> scaler_scale;
    int zernike_degree;
};

class ShapeClassifier {
public:
    ShapeClassifier();
    
    // Cargar modelo SVM desde JSON
    bool loadSVMModel(const std::string& modelPath);
    
    // Clasificar una imagen
    std::string classify(const cv::Mat& image, float& confidence);
    
    // Preprocesar imagen
    cv::Mat preprocessImage(const cv::Mat& input);
    
    // Extraer momentos de Zernike
    std::vector<double> extractZernikeMoments(const cv::Mat& binary, int degree);
    
    // Normalizar features con StandardScaler
    std::vector<double> normalizeFeatures(const std::vector<double>& features);
    
    // Predicción SVM RBF
    std::string predictSVM(const std::vector<double>& features, double& confidence);

private:
    SVMModel model;
    bool modelLoaded;
    
    // Cálculo del kernel RBF
    double rbfKernel(const std::vector<double>& x1, 
                     const std::vector<double>& x2, 
                     double gamma);
    
    // Calcular función de decisión
    std::vector<double> decisionFunction(const std::vector<double>& features);
    
    // Polinomios de Zernike
    double zernikePoly(int n, int m, double rho, double theta);
    double radialPoly(int n, int m, double rho);
    int factorial(int n);
};

#endif // SHAPE_CLASSIFIER_H