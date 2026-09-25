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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity implements OcrBridge.ImageLauncher {
    private static final int REQ_CAMERA_PERMISSION = 7000;
    private static final int REQ_CAMERA_CAPTURE = 7001;
    private static final int REQ_PICK_IMAGE = 7002;
    private static final int REQ_NATIVE_CROP = 7003;

    private WebView webView;
    private Uri photoUri;
    private String ocrMode = "basic";

    private final TextRecognizer textRecognizer =
            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private FormulaOcrEngine formulaEngine;

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

        try {
            formulaEngine = new FormulaOcrEngine(this);
        } catch (Exception e) {
            sendError("เริ่ม Math OCR ไม่สำเร็จ: " + safeMessage(e));
        }
    }

    @Override
    public void launchCamera(String mode) {
        ocrMode = normalizeMode(mode);
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

    @Override
    public void pickImage(String mode) {
        ocrMode = normalizeMode(mode);
        try {
            Intent intent;
            if (android.os.Build.VERSION.SDK_INT >= 33) {
                intent = new Intent(MediaStore.ACTION_PICK_IMAGES);
                intent.setType("image/*");
            } else {
                intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("image/*");
            }
            startActivityForResult(intent, REQ_PICK_IMAGE);
        } catch (Exception e) {
            sendError("เปิดคลังรูปไม่ได้: " + safeMessage(e));
        }
    }

    private String normalizeMode(String mode) {
        if ("calculus".equals(mode) || "basic".equals(mode) ||
                "electrical".equals(mode) || "mechanical".equals(mode)) {
            return mode;
        }
        return "basic";
    }

    private boolean shouldUseFormulaOcr() {
        return "basic".equals(ocrMode) || "calculus".equals(ocrMode);
    }

    private void openSystemCamera() {
        try {
            File dir = getImageCacheDir();
            File photoFile = File.createTempFile("calclens_camera_", ".jpg", dir);
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

    private File getImageCacheDir() {
        File dir = new File(getCacheDir(), "images");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("สร้างโฟลเดอร์รูปไม่ได้");
        }
        return dir;
    }

    private void startCrop(Uri source) {
        try {
            Intent intent = new Intent(this, CropActivity.class);
            intent.putExtra(CropActivity.EXTRA_SOURCE_URI, source.toString());
            startActivityForResult(intent, REQ_NATIVE_CROP);
        } catch (Exception e) {
            sendError("เปิดหน้าครอปไม่ได้: " + safeMessage(e));
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
                sendError("ยังไม่ได้อนุญาตใช้กล้อง กรุณาอนุญาตสิทธิ์กล้องให้ CalcLens TH");
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
            startCrop(photoUri);
            return;
        }

        if (requestCode == REQ_PICK_IMAGE &&
                resultCode == RESULT_OK &&
                data != null &&
                data.getData() != null) {
            startCrop(data.getData());
            return;
        }

        if (requestCode == REQ_NATIVE_CROP &&
                resultCode == RESULT_OK &&
                data != null &&
                data.getData() != null) {
            Uri cropped = data.getData();
            if (shouldUseFormulaOcr()) {
                runFormulaOcr(cropped);
            } else {
                runTextOcr(cropped);
            }
        }
    }

    private void runFormulaOcr(Uri uri) {
        if (formulaEngine == null) {
            sendError("Math OCR ยังเริ่มทำงานไม่ได้");
            return;
        }

        sendModelProgress(0, "กำลังเตรียม Math OCR");

        worker.submit(() -> {
            try {
                formulaEngine.ensureModel(this::sendModelProgress);
                sendModelProgress(100, "กำลังอ่านโครงสร้างสูตร");
                String latex = formulaEngine.recognize(uri);
                if (latex == null || latex.trim().isEmpty()) {
                    throw new IllegalStateException("ไม่พบสูตรในภาพ ลองครอปเฉพาะสูตรให้ชิดขึ้น");
                }
                sendFormulaResult(latex);
            } catch (Exception e) {
                sendError("Math OCR ไม่สำเร็จ: " + safeMessage(e));
            }
        });
    }

    private void runTextOcr(Uri uri) {
        try {
            InputImage image = InputImage.fromFilePath(this, uri);
            textRecognizer.process(image)
                    .addOnSuccessListener(result -> sendOcr(result.getText()))
                    .addOnFailureListener(e ->
                            sendError("OCR อ่านภาพไม่สำเร็จ: " + safeMessage(e)));
        } catch (Exception e) {
            sendError("อ่านรูปไม่ได้: " + safeMessage(e));
        }
    }

    private void sendFormulaResult(String value) {
        final String safe = JSONObject.quote(value == null ? "" : value);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onFormulaResult && window.onFormulaResult(" + safe + ")", null
        ));
    }

    private void sendOcr(String value) {
        final String safe = JSONObject.quote(value == null ? "" : value);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onOcrResult && window.onOcrResult(" + safe + ")", null
        ));
    }

    private void sendModelProgress(int percent, String message) {
        final String safe = JSONObject.quote(message == null ? "" : message);
        runOnUiThread(() -> webView.evaluateJavascript(
                "window.onModelProgress && window.onModelProgress(" +
                        percent + "," + safe + ")", null
        ));
    }

    private void sendError(String value) {
        final String safe = JSONObject.quote(value == null ? "" : value);
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
        textRecognizer.close();
        worker.shutdownNow();
        if (formulaEngine != null) formulaEngine.close();
        super.onDestroy();
    }
}
