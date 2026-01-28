package com.timer.moments;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;

public class ResultActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        ImageView ivResult = findViewById(R.id.ivResult);
        TextView tvHuMoments = findViewById(R.id.tvHuMoments);
        TextView tvPrediction = findViewById(R.id.tvPrediction);
        Button btnBack = findViewById(R.id.btnBack);

        btnBack.setOnClickListener(v -> finish());

        String imagePath = getIntent().getStringExtra("image_path");
        if (imagePath != null) {
            File imgFile = new File(imagePath);
            if (imgFile.exists()) {
                Bitmap bitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath());

                // Procesar con JNI
                ShapeRecognizer recognizer = new ShapeRecognizer(this);

                // 1. Invertir colores
                Bitmap processed = recognizer.processImage(bitmap);
                ivResult.setImageBitmap(processed);

                // 2. Calcular Descriptores de Fourier, Firma y Coordenadas
                try {
                    // a) Descriptores de Fourier
                    double[] descriptors = recognizer.getFourierDescriptors(processed);

                    // b) Firma de Forma (Distancia Centroidal)
                    double[] signature = recognizer.getCentroidDistanceSignature(processed);

                    // c) Señal Compleja
                    double[] complexSignal = recognizer.getComplexSignal(processed);

                    StringBuilder sb = new StringBuilder();
                    if (descriptors.length == 0) {
                        sb.append("No se encontró contorno válido.");
                        tvPrediction.setText("Predicción: N/A");
                    } else {
                        // Mostrar predicción
                        android.util.Pair<String, Double> prediction = recognizer.predict(descriptors);
                        tvPrediction.setText(String.format("Predicción: %s\n(Confianza: %.1f%%)",
                                prediction.first, prediction.second));

                        // --- VISUALIZAR FIRMA DE FORMA ---
                        if (signature.length > 0) {
                            ImageView ivShape = findViewById(R.id.ivShapeSignature);
                            Bitmap signatureBitmap = createSignatureGraph(signature);
                            ivShape.setImageBitmap(signatureBitmap);
                        }

                        // --- LISTAR DATOS ---
                        sb.append(String.format("Puntos del Contorno: %d\n", complexSignal.length / 2));
                        sb.append("--- Coordenadas Complejas (Muestra) ---\n");
                        // Mostrar primeros 5 puntos
                        int maxShow = Math.min(5, complexSignal.length / 2);
                        for (int k = 0; k < maxShow; k++) {
                            sb.append(String.format("z(%d) = %.1f + j%.1f\n", k, complexSignal[2 * k],
                                    complexSignal[2 * k + 1]));
                        }
                        if (complexSignal.length / 2 > 5)
                            sb.append("...\n");

                        sb.append("\n--- Descriptores de Fourier ---\n");
                        // Listar descriptores
                        for (int i = 0; i < descriptors.length; i++) {
                            sb.append(String.format("F[%d]: %.6f\n", i, descriptors[i]));
                        }
                    }
                    tvHuMoments.setText(sb.toString());
                } catch (Exception e) {
                    tvHuMoments.setText("Error calculando descriptores: " + e.getMessage());
                }
            } else {
                Toast.makeText(this, "Error: Imagen no encontrada", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Bitmap createSignatureGraph(double[] data) {
        int width = 800;
        int height = 300;
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(bitmap);

        // Fondo
        canvas.drawColor(android.graphics.Color.parseColor("#2C2C2C"));

        if (data.length == 0)
            return bitmap;

        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setColor(android.graphics.Color.CYAN);
        paint.setStyle(android.graphics.Paint.Style.STROKE);
        paint.setStrokeWidth(3f);
        paint.setAntiAlias(true);

        // Encontrar max para normalizar verticalmente
        double maxVal = 0;
        for (double d : data)
            if (d > maxVal)
                maxVal = d;
        if (maxVal == 0)
            maxVal = 1;

        android.graphics.Path path = new android.graphics.Path();

        float xStep = (float) width / data.length;

        for (int i = 0; i < data.length; i++) {
            float x = i * xStep;
            float y = (float) (height - (data[i] / maxVal) * (height * 0.8) - height * 0.1);

            if (i == 0)
                path.moveTo(x, y);
            else
                path.lineTo(x, y);
        }

        canvas.drawPath(path, paint);
        return bitmap;
    }
}
