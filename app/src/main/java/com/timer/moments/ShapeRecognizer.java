package com.timer.moments;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.IOException;

public class ShapeRecognizer {
    private static final String TAG = "ShapeRecognizer";

    // Flag para indicar si la librería nativa se cargó exitosamente
    private static boolean nativeLibraryLoaded = false;

    // Intentar cargar la librería nativa
    static {
        try {
            System.loadLibrary("moments");
            nativeLibraryLoaded = true;
            Log.d(TAG, "✓ Librería nativa 'moments' cargada exitosamente");
        } catch (UnsatisfiedLinkError e) {
            nativeLibraryLoaded = false;
            Log.e(TAG, "❌ ERROR: No se pudo cargar libmoments.so", e);
            Log.e(TAG, "Verifica que CMakeLists.txt esté configurado correctamente");
            Log.e(TAG, "Revisa Build Output para errores de compilación");
        }
    }

    private boolean initialized = false;
    private float lastConfidence = 0.0f;

    /**
     * Constructor: Inicializa el clasificador cargando el modelo SVM
     * @param context Contexto de la aplicación
     */
    public ShapeRecognizer(Context context) {
        // Verificar PRIMERO si la librería nativa se cargó

        Log.d(TAG, "INSTANCIA ShapeRecognizer: " + this);

        Log.d(TAG, "USANDO INSTANCIA ShapeRecognizer: " + this);


        if (!nativeLibraryLoaded) {
            Log.e(TAG, "❌ No se puede inicializar: librería nativa no cargada");
            initialized = false;
            return;
        }

        try {
            Log.d(TAG, "Inicializando ShapeRecognizer...");

            // 1. Copiar modelo desde assets a almacenamiento interno
            String modelFileName = "svm_model.json";
            File modelFile = new File(context.getFilesDir(), modelFileName);

            // Solo copiar si no existe
            if (!modelFile.exists()) {
                Log.d(TAG, "Copiando modelo desde assets...");
                copyAssetToFile(context, modelFileName, modelFile);
            } else {
                Log.d(TAG, "Modelo ya existe en: " + modelFile.getAbsolutePath());
            }

            // 2. Inicializar la parte nativa con la ruta del modelo
            String modelPath = modelFile.getAbsolutePath();
            Log.d(TAG, "Cargando modelo desde: " + modelPath);

            nativeInit(modelPath);

            initialized = true;
            Log.d(TAG, "✓ ShapeRecognizer inicializado correctamente");

        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "❌ Error JNI al llamar nativeInit", e);
            initialized = false;
        } catch (Exception e) {
            Log.e(TAG, "❌ Error al inicializar ShapeRecognizer", e);
            initialized = false;
        }
    }

    /**
     * Copia un archivo desde assets al almacenamiento interno
     */
    private void copyAssetToFile(Context context, String assetName, File outFile)
            throws IOException {

        InputStream inputStream = null;
        FileOutputStream outputStream = null;

        try {
            // Abrir archivo desde assets
            inputStream = context.getAssets().open(assetName);

            // Crear archivo de salida
            outputStream = new FileOutputStream(outFile);

            // Copiar bytes
            byte[] buffer = new byte[1024];
            int length;
            while ((length = inputStream.read(buffer)) > 0) {
                outputStream.write(buffer, 0, length);
            }

            outputStream.flush();
            Log.d(TAG, "Archivo copiado exitosamente: " + outFile.getAbsolutePath());

        } finally {
            // Cerrar streams
            if (inputStream != null) {
                try { inputStream.close(); } catch (IOException e) { }
            }
            if (outputStream != null) {
                try { outputStream.close(); } catch (IOException e) { }
            }
        }
    }

    /**
     * Clasifica una imagen dibujada
     * @param bitmap Imagen a clasificar
     * @return Etiqueta de la clase predicha
     */
    public String classify(Bitmap bitmap) {
        if (!nativeLibraryLoaded) {
            Log.e(TAG, "Error: Librería nativa no cargada");
            return "error: librería no cargada";
        }

        if (!initialized) {
            Log.e(TAG, "Error: Clasificador no inicializado");
            return "error: no inicializado";
        }

        if (bitmap == null) {
            Log.e(TAG, "Error: Bitmap es null");
            return "error: bitmap null";
        }

        Log.d(TAG, "Clasificando imagen de " + bitmap.getWidth() + "x" + bitmap.getHeight());

        try {
            // Llamar a la función nativa
            String result = nativeClassify(bitmap);
            lastConfidence = nativeGetConfidence();

            Log.d(TAG, "Resultado: " + result + " (confianza: " +
                    String.format("%.2f%%", lastConfidence * 100) + ")");

            return result;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "❌ Error JNI al clasificar", e);
            return "error: JNI";
        }
    }

    /**
     * Obtiene la confianza de la última clasificación
     * @return Valor entre 0 y 1
     */
    public float getLastConfidence() {
        return lastConfidence;
    }

    /**
     * Verifica si el clasificador está inicializado
     */
    public boolean isInitialized() {
        return nativeLibraryLoaded && initialized;
    }

    /**
     * Verifica si la librería nativa está cargada
     */
    public static boolean isNativeLibraryLoaded() {
        return nativeLibraryLoaded;
    }

    /**
     * Libera recursos nativos
     */
    public void release() {
        if (initialized && nativeLibraryLoaded) {
            try {
                nativeRelease();
                initialized = false;
                Log.d(TAG, "Recursos liberados");
            } catch (UnsatisfiedLinkError e) {
                Log.e(TAG, "Error al liberar recursos", e);
            }
        }
    }

    // ==================== Métodos Nativos ====================

    /**
     * Inicializa el clasificador nativo con la ruta del modelo
     */
    private native void nativeInit(String modelPath);

    /**
     * Clasifica una imagen usando el modelo SVM
     */
    private native String nativeClassify(Bitmap bitmap);

    /**
     * Obtiene la confianza de la última predicción
     */
    private native float nativeGetConfidence();

    /**
     * Libera memoria nativa
     */
    private native void nativeRelease();
}