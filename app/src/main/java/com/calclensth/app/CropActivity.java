package com.calclensth.app;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class CropActivity extends Activity {
    public static final String EXTRA_SOURCE_URI = "source_uri";
    public static final String EXTRA_MODE = "ocr_mode";

    private CropView cropView;
    private Bitmap sourceBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        String source = getIntent().getStringExtra(EXTRA_SOURCE_URI);
        if (source == null || source.isEmpty()) {
            finishWithError("ไม่พบรูปต้นฉบับ");
            return;
        }

        String mode = getIntent().getStringExtra(EXTRA_MODE);
        buildUi(mode);

        try {
            Uri uri = Uri.parse(source);
            sourceBitmap = loadBitmap(uri, 2200);
            if (sourceBitmap == null) {
                finishWithError("เปิดรูปไม่ได้");
                return;
            }
            cropView.setBitmap(sourceBitmap);
        } catch (Exception e) {
            finishWithError("เปิดรูปไม่ได้: " + safeMessage(e));
        }
    }

    private void buildUi(String mode) {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.BLACK);

        float d = getResources().getDisplayMetrics().density;

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding((int)(10*d), 0, (int)(10*d), 0);
        bar.setBackgroundColor(0xFF111827);

        Button cancel = new Button(this);
        cancel.setText("ยกเลิก");
        cancel.setTextColor(Color.WHITE);
        cancel.setBackgroundColor(Color.TRANSPARENT);
        cancel.setOnClickListener(v -> finish());

        TextView title = new TextView(this);
        boolean mathMode = "basic".equals(mode) || "calculus".equals(mode);
        title.setText(mathMode ? "ครอปโจทย์ 1 ข้อ" : "ครอปเฉพาะโจทย์");
        title.setTextColor(Color.WHITE);
        title.setTextSize(18);
        title.setGravity(Gravity.CENTER);

        Button done = new Button(this);
        done.setText("✓");
        done.setTextSize(26);
        done.setTextColor(Color.WHITE);
        done.setBackgroundColor(0xFF2563EB);
        done.setOnClickListener(v -> saveCrop());

        bar.addView(cancel, new LinearLayout.LayoutParams(0, (int)(56*d), 1f));
        bar.addView(title, new LinearLayout.LayoutParams(0, (int)(56*d), 2f));
        bar.addView(done, new LinearLayout.LayoutParams(0, (int)(56*d), 1f));

        cropView = new CropView(this);
        LinearLayout.LayoutParams cropParams =
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        0,
                        1f
                );

        TextView hint = new TextView(this);
        hint.setText(mathMode
                ? "ครอปให้เหลือโจทย์ 1 ข้อได้ทั้งบรรทัด • ระบบจะหาสูตรบนสุดให้อัตโนมัติ"
                : "ลากมุมสีขาวเพื่อปรับกรอบ • ลากด้านในกรอบเพื่อย้าย");
        hint.setTextColor(Color.WHITE);
        hint.setTextSize(13);
        hint.setGravity(Gravity.CENTER);
        hint.setPadding(8, (int)(8*d), 8, (int)(8*d));
        hint.setBackgroundColor(0xFF111827);

        root.addView(bar);
        root.addView(cropView, cropParams);
        root.addView(hint,
                new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        (int)(42*d)
                ));

        setContentView(root);
    }

    private Bitmap loadBitmap(Uri uri, int maxSide) throws Exception {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(in, null, bounds);
        }

        int sample = 1;
        int largest = Math.max(bounds.outWidth, bounds.outHeight);
        while (largest / sample > maxSide) sample *= 2;

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = Math.max(1, sample);
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap bitmap;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            bitmap = BitmapFactory.decodeStream(in, null, opts);
        }

        if (bitmap == null) return null;

        int rotation = readRotation(uri);
        if (rotation != 0) {
            Matrix m = new Matrix();
            m.postRotate(rotation);
            Bitmap rotated = Bitmap.createBitmap(
                    bitmap, 0, 0,
                    bitmap.getWidth(), bitmap.getHeight(),
                    m, true
            );
            if (rotated != bitmap) bitmap.recycle();
            bitmap = rotated;
        }

        return bitmap;
    }

    private int readRotation(Uri uri) {
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            ExifInterface exif = new ExifInterface(in);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );
            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) return 90;
            if (orientation == ExifInterface.ORIENTATION_ROTATE_180) return 180;
            if (orientation == ExifInterface.ORIENTATION_ROTATE_270) return 270;
        } catch (Exception ignored) {
        }
        return 0;
    }

    private void saveCrop() {
        try {
            Bitmap cropped = cropView.createCroppedBitmap();
            if (cropped == null) {
                Toast.makeText(this, "ครอปรูปไม่ได้", Toast.LENGTH_SHORT).show();
                return;
            }

            File dir = new File(getCacheDir(), "images");
            if (!dir.exists() && !dir.mkdirs()) {
                throw new IllegalStateException("สร้างโฟลเดอร์ไม่ได้");
            }

            File out = File.createTempFile("calclens_crop_", ".jpg", dir);
            try (FileOutputStream fos = new FileOutputStream(out)) {
                if (!cropped.compress(Bitmap.CompressFormat.JPEG, 95, fos)) {
                    throw new IllegalStateException("บันทึกรูปไม่ได้");
                }
            }

            Intent result = new Intent();
            result.setData(Uri.fromFile(out));
            setResult(RESULT_OK, result);
            finish();
        } catch (Exception e) {
            Toast.makeText(
                    this,
                    "บันทึกรูปไม่ได้: " + safeMessage(e),
                    Toast.LENGTH_LONG
            ).show();
        }
    }

    private void finishWithError(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
        setResult(RESULT_CANCELED);
        finish();
    }

    private String safeMessage(Exception e) {
        return e.getMessage() == null
                ? e.getClass().getSimpleName()
                : e.getMessage();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (sourceBitmap != null && !sourceBitmap.isRecycled()) {
            sourceBitmap.recycle();
        }
    }
}
