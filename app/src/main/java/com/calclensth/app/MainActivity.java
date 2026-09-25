package com.calclensth.app;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.core.content.FileProvider;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.io.File;

public class MainActivity extends Activity implements OcrBridge.CameraLauncher {
    private static final int REQ_CAMERA = 7001;
    private WebView webView;
    private Uri photoUri;
    private File photoFile;
    private final TextRecognizer recognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        webView = new WebView(this);
        setContentView(webView);

        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(true);

        webView.addJavascriptInterface(new OcrBridge(this), "Android");
        webView.setWebViewClient(new WebViewClient());
        webView.loadUrl("file:///android_asset/index.html");
    }

    @Override
    public void launchCamera() {
        try {
            File dir = new File(getCacheDir(), "images");
            if (!dir.exists()) dir.mkdirs();
            photoFile = File.createTempFile("calclens_", ".jpg", dir);
            photoUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".fileprovider",
                    photoFile
            );

            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                    Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(intent, REQ_CAMERA);
        } catch (Exception e) {
            sendError("เปิดกล้องไม่ได้: " + e.getMessage());
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_CAMERA && resultCode == RESULT_OK && photoUri != null) {
            runOcr(photoUri);
        }
    }

    private void runOcr(Uri uri) {
        try {
            InputImage image = InputImage.fromFilePath(this, uri);
            recognizer.process(image)
                    .addOnSuccessListener(result -> sendOcr(result.getText()))
                    .addOnFailureListener(e -> sendError("OCR อ่านภาพไม่สำเร็จ: " + e.getMessage()));
        } catch (Exception e) {
            sendError("อ่านรูปไม่ได้: " + e.getMessage());
        }
    }

    private void sendOcr(String text) {
        final String safe = jsonEscape(text);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onOcrResult && window.onOcrResult(" + safe + ")", null
        ));
    }

    private void sendError(String text) {
        final String safe = jsonEscape(text);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onNativeError && window.onNativeError(" + safe + ")", null
        ));
    }

    private String jsonEscape(String value) {
        if (value == null) value = "";
        return """ + value
                .replace("\\", "\\\\")
                .replace(""", "\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t") + """;
    }

    @Override
    protected void onDestroy() {
        recognizer.close();
        super.onDestroy();
    }
}
