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

                // 2. Calcular Descriptores de Fourier
                try {
                    double[] descriptors = recognizer.getFourierDescriptors(processed);
                    StringBuilder sb = new StringBuilder();
                    if (descriptors.length == 0) {
                        sb.append("No se encontró contorno válido.");
                        tvPrediction.setText("Predicción: N/A");
                    } else {
                        // Mostrar predicción
                        android.util.Pair<String, Double> prediction = recognizer.predict(descriptors);
                        tvPrediction.setText(String.format("Predicción: %s\n(Confianza: %.1f%%)",
                                prediction.first, prediction.second));

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

}
