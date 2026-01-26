package com.timer.moments;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private DrawView drawView;
    private TextView tvLabel;
    private ShapeRecognizer recognizer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        drawView = findViewById(R.id.drawView);
        tvLabel = findViewById(R.id.tvLabel);
        Button btnClear = findViewById(R.id.btnClear);
        Button btnPredict = findViewById(R.id.btnPredict);

        // Inicializar el reconocedor
        try {
            recognizer = new ShapeRecognizer(this);
            tvLabel.setText("Sistema listo. Dibuja una forma.");
        } catch (Exception e) {
            tvLabel.setText("Error al inicializar: " + e.getMessage());
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_LONG).show();
        }

        btnClear.setOnClickListener(v -> {
            drawView.clearCanvas();
            tvLabel.setText("Canvas limpio. Dibuja una forma.");
        });

        btnPredict.setOnClickListener(v -> {
            if (recognizer == null) {
                tvLabel.setText("Error: Sistema no inicializado");
                return;
            }

            try {
                // Obtener bitmap del canvas
                Bitmap bitmap = drawView.getBitmap();

                // Clasificar
                String result = recognizer.classify(bitmap);

                // Obtener confianza
                float confidence = recognizer.getLastConfidence();

                // Mostrar resultado
                String displayText = "";
                String emoji = "";

                switch (result.toLowerCase()) {
                    case "circle":
                        emoji = "⭕";
                        displayText = String.format("Círculo detectado\n(Confianza: %.1f%%)",
                                confidence * 100);
                        break;
                    case "triangle":
                        emoji = "🔺";
                        displayText = String.format("Triángulo detectado\n(Confianza: %.1f%%)",
                                confidence * 100);
                        break;
                    case "square":
                        emoji = "⬜";
                        displayText = String.format("Cuadrado detectado\n(Confianza: %.1f%%)",
                                confidence * 100);
                        break;
                    case "desconocido":
                        emoji = "❓";
                        displayText = "Forma no reconocida";
                        break;
                    default:
                        displayText = "Resultado: " + result;
                }

                tvLabel.setText(emoji + " " + displayText);

            } catch (Exception e) {
                tvLabel.setText("Error en clasificación: " + e.getMessage());
                e.printStackTrace();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (recognizer != null) {
            recognizer.release();
        }
    }
}