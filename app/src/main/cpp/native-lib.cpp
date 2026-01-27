#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

#define LOG_TAG "MomentsNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

extern "C" {

// ===============================
// Procesamiento de Imagen: Invertir Colores
// ===============================
JNIEXPORT void JNICALL
Java_com_timer_moments_ShapeRecognizer_nativeProcessImage(
        JNIEnv* env,
        jobject,
        jobject input,
        jobject output) {

    AndroidBitmapInfo info;
    void* pixelsInput = nullptr;
    void* pixelsOutput = nullptr;

    if (AndroidBitmap_getInfo(env, input, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return;

    if (AndroidBitmap_lockPixels(env, input, &pixelsInput) < 0) return;
    if (AndroidBitmap_lockPixels(env, output, &pixelsOutput) < 0) {
        AndroidBitmap_unlockPixels(env, input);
        return;
    }

    cv::Mat src(info.height, info.width, CV_8UC4, pixelsInput);
    cv::Mat dst(info.height, info.width, CV_8UC4, pixelsOutput);

    // Invertir colores para visualización
    cv::Mat src_rgb;
    cv::cvtColor(src, src_rgb, cv::COLOR_RGBA2RGB); 
    
    cv::Mat inverted_rgb;
    cv::bitwise_not(src_rgb, inverted_rgb);
    
    cv::Mat inverted_rgba;
    cv::cvtColor(inverted_rgb, inverted_rgba, cv::COLOR_RGB2RGBA);
    
    inverted_rgba.copyTo(dst);

    AndroidBitmap_unlockPixels(env, input);
    AndroidBitmap_unlockPixels(env, output);
}

// ===============================
// Cálculo de Descriptores de Fourier (Complex Coordinates)
// ===============================
JNIEXPORT jdoubleArray JNICALL
Java_com_timer_moments_ShapeRecognizer_nativeGetFourierDescriptors(
        JNIEnv* env,
        jobject,
        jobject input) {

    AndroidBitmapInfo info;
    void* pixels = nullptr;

    if (AndroidBitmap_getInfo(env, input, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return env->NewDoubleArray(0);
    }

    if (AndroidBitmap_lockPixels(env, input, &pixels) < 0) {
        return env->NewDoubleArray(0);
    }

    cv::Mat src(info.height, info.width, CV_8UC4, pixels);
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_RGBA2GRAY);
    AndroidBitmap_unlockPixels(env, input);

    // 1. Segmentación: Umbral Adaptativo (para robustez iluminación)
    cv::Mat binary;
    // Uso de ADAPTIVE_THRESH_GAUSSIAN_C, invertido para que el objeto sea blanco
    cv::adaptiveThreshold(gray, binary, 255, 
        cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv::THRESH_BINARY_INV, 
        11, 2);

    // 2. Extraer Contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) {
        LOGD("No se encontraron contornos");
        return env->NewDoubleArray(0);
    }

    // Tomar el contorno más grande (asumiendo que es el dibujo principal)
    size_t largestIndex = 0;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > largestArea) {
            largestArea = area;
            largestIndex = i;
        }
    }
    std::vector<cv::Point> contour = contours[largestIndex];
    size_t N = contour.size();

    // Mínimo de puntos necesarios para DFT útil
    if (N < 4) {
         LOGD("Contorno muy pequeño para DFT");
        return env->NewDoubleArray(0);
    }

    // 3. Crear señal compleja s(n) = (x - xc) + j(y - yc)
    // Calcular centroide (xc, yc)
    cv::Moments m = cv::moments(contour);
    double xc = m.m10 / (m.m00 + 1e-5); 
    double yc = m.m01 / (m.m00 + 1e-5);

    // OpenCV DFT requiere un canal doble (Re, Im) o matriz de complejos
    // Creamos matriz (N, 1) tipo CV_64FC2 (Complejo Double)
    cv::Mat complexSignal(N, 1, CV_64FC2);
    
    for (size_t i = 0; i < N; i++) {
        double real = contour[i].x - xc;
        double imag = contour[i].y - yc;
        complexSignal.at<std::complex<double>>(i, 0) = std::complex<double>(real, imag);
    }

    // 4. DFT
    cv::Mat dftResult;
    // dftResult será también (N, 1) CV_64FC2
    cv::dft(complexSignal, dftResult);

    // 5. Normalización
    // Extraer magnitudes
    std::vector<double> descriptors;
    double firstHarmonicMag = 0.0;
    
    // Devolvemos por ejemplo los primeros 15 descriptores (o N si N < 15)
    size_t numDescriptors = std::min((size_t)15, N);
    
    // Encontrar F(1) para normalizar escala
    // dftResult tiene DC en indice 0, F(1) en 1...
    if (N > 1) {
        std::complex<double> f1 = dftResult.at<std::complex<double>>(1, 0);
        firstHarmonicMag = std::abs(f1);
    }

    if (firstHarmonicMag < 1e-9) firstHarmonicMag = 1.0; // Evitar div/0

    for (size_t i = 0; i < numDescriptors; i++) {
        // Obtenemos descriptor F(i)
        std::complex<double> fi = dftResult.at<std::complex<double>>(i, 0);
        double magnitude = std::abs(fi);
        
        // Normalizar por F(1) para invarianza de escala
        double normalized = magnitude / firstHarmonicMag;
        
        // F(0) suele ser cercano a 0 si restamos el centroide correctamente,
        // pero lo incluimos igual o lo descartamos según preferencia.
        // El usuario pidió: "Dividir por el primer armónico |F(1)|" y "descartar fase".
        
        descriptors.push_back(normalized);
    }

    // Copiar a Java array
    jdoubleArray result = env->NewDoubleArray(descriptors.size());
    env->SetDoubleArrayRegion(result, 0, descriptors.size(), descriptors.data());

    return result;
}

} // extern "C"
