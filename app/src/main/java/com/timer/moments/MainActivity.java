package com.timer.moments;


import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private DrawView drawView;
    private TextView tvLabel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        drawView = findViewById(R.id.drawView);
        tvLabel = findViewById(R.id.tvLabel);
        Button btnClear = findViewById(R.id.btnClear);
        Button btnPredict = findViewById(R.id.btnPredict);

        btnClear.setOnClickListener(v -> {
            drawView.clearCanvas();
            tvLabel.setText("Canvas limpio");
        });

        btnPredict.setOnClickListener(v -> {
            // Aquí irá la predicción con ML
            tvLabel.setText("Predicción pendiente...");
        });
    }
}