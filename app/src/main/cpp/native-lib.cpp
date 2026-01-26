#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "shape_classifier.h"

#define LOG_TAG "NativeLib"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// ===============================
// Variables globales
// ===============================
static ShapeClassifier* g_classifier = nullptr;
static float g_last_confidence = 0.0f;

extern "C" {

// ===============================
// Inicialización
// ===============================
JNIEXPORT void JNICALL
Java_com_timer_moments_ShapeRecognizer_nativeInit(
        JNIEnv* env,
jobject /* thiz */,
jstring dataPath) {

if (g_classifier != nullptr) {
LOGD("Clasificador ya inicializado");
return;
}

g_classifier = new ShapeClassifier();

const char* path = env->GetStringUTFChars(dataPath, nullptr);
bool ok = g_classifier->loadSVMModel(std::string(path));
env->ReleaseStringUTFChars(dataPath, path);

if (!ok) {
LOGD("Error cargando modelo SVM");
} else {
LOGD("Clasificador inicializado correctamente");
}
}

// ===============================
// Clasificación
// ===============================
JNIEXPORT jstring JNICALL
        Java_com_timer_moments_ShapeRecognizer_nativeClassify(
        JNIEnv* env,
        jobject /* thiz */,
        jobject bitmap) {

if (g_classifier == nullptr) {
LOGD("Error: Clasificador no inicializado");
return env->NewStringUTF("error");
}

AndroidBitmapInfo info;
void* pixels = nullptr;

if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
LOGD("Error obteniendo info del bitmap");
return env->NewStringUTF("error");
}

if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
LOGD("Formato de bitmap no soportado");
return env->NewStringUTF("error");
}

AndroidBitmap_lockPixels(env, bitmap, &pixels);

cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
cv::Mat image = rgba.clone();

AndroidBitmap_unlockPixels(env, bitmap);

float confidence = 0.0f;
std::string label = g_classifier->classify(image, confidence);
g_last_confidence = confidence;

LOGD("Resultado: %s (%.2f%%)", label.c_str(), confidence * 100.0f);

return env->NewStringUTF(label.c_str());
}

// ===============================
// Confianza
// ===============================
JNIEXPORT jfloat JNICALL
Java_com_timer_moments_ShapeRecognizer_nativeGetConfidence(
        JNIEnv*,
        jobject) {
    return g_last_confidence;
}

// ===============================
// Liberar recursos
// ===============================
JNIEXPORT void JNICALL
Java_com_timer_moments_ShapeRecognizer_nativeRelease(
        JNIEnv*,
jobject) {

if (g_classifier != nullptr) {
delete g_classifier;
g_classifier = nullptr;
LOGD("Clasificador liberado");
}
}

} // extern "C"
