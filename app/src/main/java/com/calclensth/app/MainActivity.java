package com.calclensth.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
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

import org.json.JSONObject;

import java.io.File;

public class MainActivity extends Activity implements OcrBridge.CameraLauncher {
    private static final int REQ_CAMERA_PERMISSION = 7000;
    private static final int REQ_CAMERA_CAPTURE = 7001;

    private WebView webView;
    private Uri photoUri;

    private final TextRecognizer recognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        webView = new WebView(this);
        setContentView(webView);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);

        webView.addJavascriptInterface(new OcrBridge(this), "Android");
        webView.setWebViewClient(new WebViewClient());
        webView.loadUrl("file:///android_asset/index.html");
    }

    @Override
    public void launchCamera() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M &&
                checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                    new String[]{Manifest.permission.CAMERA},
                    REQ_CAMERA_PERMISSION
            );
            return;
        }

        openSystemCamera();
    }

    private void openSystemCamera() {
        try {
            File dir = new File(getCacheDir(), "images");
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IllegalStateException("สร้างโฟลเดอร์รูปไม่ได้");
            }

            File photoFile = File.createTempFile("calclens_", ".jpg", dir);
            photoUri = FileProvider.getUriForFile(
                    this,
                    getPackageName() + ".fileprovider",
                    photoFile
            );

            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            intent.addFlags(
                    Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
            );

            if (intent.resolveActivity(getPackageManager()) == null) {
                throw new IllegalStateException("ไม่พบแอปกล้องในเครื่อง");
            }

            startActivityForResult(intent, REQ_CAMERA_CAPTURE);
        } catch (Exception e) {
            sendError("เปิดกล้องไม่ได้: " + safeMessage(e));
        }
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQ_CAMERA_PERMISSION) {
            if (grantResults.length > 0 &&
                    grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openSystemCamera();
            } else {
                sendError("ยังไม่ได้อนุญาตใช้กล้อง กรุณาเลือก 'อนุญาต' หรือไปที่ การตั้งค่า > แอป > CalcLens TH > สิทธิ์ > กล้อง");
            }
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQ_CAMERA_CAPTURE &&
                resultCode == RESULT_OK &&
                photoUri != null) {
            runOcr(photoUri);
        }
    }

    private void runOcr(Uri uri) {
        try {
            InputImage image = InputImage.fromFilePath(this, uri);
            recognizer.process(image)
                    .addOnSuccessListener(result -> sendOcr(result.getText()))
                    .addOnFailureListener(e ->
                            sendError("OCR อ่านภาพไม่สำเร็จ: " + safeMessage(e)));
        } catch (Exception e) {
            sendError("อ่านรูปไม่ได้: " + safeMessage(e));
        }
    }

    private void sendOcr(String text) {
        final String safe = JSONObject.quote(text == null ? "" : text);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onOcrResult && window.onOcrResult(" + safe + ")", null
        ));
    }

    private void sendError(String text) {
        final String safe = JSONObject.quote(text == null ? "" : text);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onNativeError && window.onNativeError(" + safe + ")", null
        ));
    }

    private String safeMessage(Exception e) {
        return e.getMessage() == null
                ? e.getClass().getSimpleName()
                : e.getMessage();
    }

    @Override
    protected void onDestroy() {
        recognizer.close();
        super.onDestroy();
    }
}
