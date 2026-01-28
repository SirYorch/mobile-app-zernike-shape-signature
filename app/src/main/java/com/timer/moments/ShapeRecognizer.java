package com.timer.moments;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

public class ShapeRecognizer {
    private static final String TAG = "ShapeRecognizer";
    private static final String PREFS_NAME = "ShapeDataset";

    // Dataset en memoria: Label -> Lista de descriptores
    // Para simplificar, usaremos una lista de pares (Label, Descriptor)
    private java.util.List<android.util.Pair<String, double[]>> dataset = new java.util.ArrayList<>();
    private Context context;

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
        }
    }

    public ShapeRecognizer(Context context) {
        this.context = context;
        loadDataset();
    }

    /**
     * Agrega una muestra de entrenamiento
     */
    public void addTrainingSample(String label, double[] descriptors) {
        dataset.add(new android.util.Pair<>(label, descriptors));
        saveDataset();
    }

    /**
     * Busca la etiqueta más cercana en el dataset
     * 
     * @return Pair<Etiqueta, Distancia>
     */
    public android.util.Pair<String, Double> predict(double[] descriptors) {
        if (dataset.isEmpty()) {
            return new android.util.Pair<>("Desconocido (Dataset vacío)", 0.0);
        }

        String bestLabel = "Desconocido";
        double minDistance = Double.MAX_VALUE;

        for (android.util.Pair<String, double[]> sample : dataset) {
            double distance = ConfusionMatrixUtils.euclideanDistance(descriptors, sample.second);
            if (distance < minDistance) {
                minDistance = distance;
                bestLabel = sample.first;
            }
        }

        return new android.util.Pair<>(bestLabel, minDistance);
    }

    /**
     * Carga el dataset desde SharedPreferences (JSON simple)
     */
    private void loadDataset() {
        android.content.SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        String json = prefs.getString("dataset", null);
        if (json != null) {
            try {
                org.json.JSONArray array = new org.json.JSONArray(json);
                for (int i = 0; i < array.length(); i++) {
                    org.json.JSONObject obj = array.getJSONObject(i);
                    String label = obj.getString("label");
                    org.json.JSONArray descArray = obj.getJSONArray("descriptors");
                    double[] descriptors = new double[descArray.length()];
                    for (int j = 0; j < descArray.length(); j++) {
                        descriptors[j] = descArray.getDouble(j);
                    }
                    dataset.add(new android.util.Pair<>(label, descriptors));
                }
            } catch (Exception e) {
                Log.e(TAG, "Error cargando dataset", e);
            }
        }
    }

    /**
     * Guarda el dataset en SharedPreferences
     */
    private void saveDataset() {
        try {
            org.json.JSONArray array = new org.json.JSONArray();
            for (android.util.Pair<String, double[]> sample : dataset) {
                org.json.JSONObject obj = new org.json.JSONObject();
                obj.put("label", sample.first);
                org.json.JSONArray descArray = new org.json.JSONArray();
                for (double d : sample.second) {
                    descArray.put(d);
                }
                obj.put("descriptors", descArray);
                array.put(obj);
            }

            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    .edit()
                    .putString("dataset", array.toString())
                    .apply();
        } catch (Exception e) {
            Log.e(TAG, "Error guardando dataset", e);
        }
    }

    /**
     * Procesa la imagen usando la implementación nativa (Invertir colores)
     */
    public Bitmap processImage(Bitmap input) {
        if (!nativeLibraryLoaded) {
            Log.e(TAG, "Error: Librería nativa no cargada");
            return input;
        }
        if (input == null)
            return null;

        try {
            Bitmap output = input.copy(input.getConfig(), true);
            nativeProcessImage(input, output);
            return output;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Error JNI al procesar imagen", e);
            return input;
        }
    }

    /**
     * Obtiene los Descriptores de Fourier (Invariantes a escala, rotación y
     * traslación)
     */
    public double[] getFourierDescriptors(Bitmap input) {
        if (!nativeLibraryLoaded || input == null) {
            return new double[0];
        }
        return nativeGetFourierDescriptors(input);
    }

    /**
     * Libera recursos nativos (si fuera necesario)
     */
    public void release() {
        // Nada que liberar por ahora
    }

    // ==================== Métodos Nativos ====================

    /**
     * Invierte colores usando OpenCV
     */
    private native void nativeProcessImage(Bitmap input, Bitmap output);

    /**
     * Calcula Descriptores de Fourier usando OpenCV
     */
    private native double[] nativeGetFourierDescriptors(Bitmap input);

    /**
     * Calcula la firma de forma (Distancia Centroidal)
     */
    private native double[] nativeGetCentroidDistanceSignature(Bitmap input);

    /**
     * Obtiene la señal de coordenadas complejas (Real, Imag)
     */
    private native double[] nativeGetComplexSignal(Bitmap input);

    /**
     * Wrappers públicos
     */
    public double[] getCentroidDistanceSignature(Bitmap input) {
        if (!nativeLibraryLoaded || input == null)
            return new double[0];
        try {
            return nativeGetCentroidDistanceSignature(input);
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Error invocado nativeGetCentroidDistanceSignature", e);
            return new double[0];
        }
    }

    public double[] getComplexSignal(Bitmap input) {
        if (!nativeLibraryLoaded || input == null)
            return new double[0];
        try {
            return nativeGetComplexSignal(input);
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Error invocado nativeGetComplexSignal", e);
            return new double[0];
        }
    }
}