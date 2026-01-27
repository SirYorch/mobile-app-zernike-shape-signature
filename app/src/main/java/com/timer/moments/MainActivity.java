package com.timer.moments;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

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
        } catch (Exception e) {
            tvLabel.setText("Error al inicializar: " + e.getMessage());
            Toast.makeText(this, "Error cargando OpenCV", Toast.LENGTH_LONG).show();
        }

        btnClear.setOnClickListener(v -> {
            drawView.clearCanvas();
        });

        btnPredict.setOnClickListener(v -> {
            Bitmap bitmap = drawView.getBitmap();
            if (bitmap != null) {
                try {
                    // Guardar bitmap en archivo temporal
                    File cachePath = new File(getCacheDir(), "images");
                    cachePath.mkdirs();
                    File imagePath = new File(cachePath, "predict_image.png");
                    FileOutputStream stream = new FileOutputStream(imagePath);
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                    stream.close();

                    // Iniciar actividad de resultados
                    Intent intent = new Intent(MainActivity.this, ResultActivity.class);
                    intent.putExtra("image_path", imagePath.getAbsolutePath());
                    startActivity(intent);

                } catch (IOException e) {
                    e.printStackTrace();
                    Toast.makeText(this, "Error al guardar imagen", Toast.LENGTH_SHORT).show();
                }
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