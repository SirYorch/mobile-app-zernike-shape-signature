package com.timer.moments;

public class ConfusionMatrixUtils {

    /**
     * Calcula la distancia Euclídea entre dos vectores de características.
     * 
     * @param feature1 Primer vector (descriptors)
     * @param feature2 Segundo vector (descriptors)
     * @return La distancia o Double.MAX_VALUE si las dimensiones no coinciden.
     */
    public static double euclideanDistance(double[] feature1, double[] feature2) {
        if (feature1 == null || feature2 == null || feature1.length != feature2.length) {
            return Double.MAX_VALUE;
        }

        double sum = 0.0;
        for (int i = 0; i < feature1.length; i++) {
            double diff = feature1[i] - feature2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }
}
