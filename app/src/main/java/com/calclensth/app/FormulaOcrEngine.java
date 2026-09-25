package com.calclensth.app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class FormulaOcrEngine implements AutoCloseable {
    public interface ProgressCallback {
        void onProgress(int percent, String message);
    }

    private static final int SIZE = 384;
    private static final float MEAN = 0.7931f;
    private static final float STD = 0.1738f;
    private static final String MODEL_NAME = "pp-formulanet-s.onnx";
    private static final String MODEL_URL =
            "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet-s.onnx";
    private static final String MODEL_SHA256 =
            "0ee32c7bfbd9e586364f89f71860476ccb5334e35674a61f3df5e0553d6a6dcc";
    private static final long MODEL_SIZE = 231878904L;

    private final Context context;
    private final File modelFile;
    private final Map<Integer, String> vocab = new HashMap<>();
    private final Set<Integer> specialIds = new HashSet<>();
    private final Map<Integer, Integer> byteDecoder = new HashMap<>();

    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;

    public FormulaOcrEngine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        File dir = new File(this.context.getFilesDir(), "models");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("สร้างโฟลเดอร์โมเดลไม่ได้");
        }
        modelFile = new File(dir, MODEL_NAME);
        loadTokenizer();
        buildByteDecoder();
    }

    public boolean isModelReady() {
        return modelFile.exists() && modelFile.length() == MODEL_SIZE;
    }

    public long getModelSizeBytes() {
        return MODEL_SIZE;
    }

    public synchronized void ensureModel(ProgressCallback callback) throws Exception {
        if (isModelReady()) {
            if (callback != null) callback.onProgress(100, "โมเดล Math OCR พร้อมใช้งาน");
            return;
        }

        if (modelFile.exists()) modelFile.delete();

        File part = new File(modelFile.getParentFile(), MODEL_NAME + ".part");
        if (part.exists()) part.delete();

        if (callback != null) callback.onProgress(0, "กำลังดาวน์โหลด Math OCR ครั้งแรก ~221 MB");

        HttpURLConnection conn = null;
        try {
            URL url = new URL(MODEL_URL);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(20000);
            conn.setReadTimeout(30000);
            conn.setInstanceFollowRedirects(true);
            conn.setRequestProperty("User-Agent", "CalcLensTH/0.4");
            conn.connect();

            int code = conn.getResponseCode();
            if (code < 200 || code >= 300) {
                throw new IllegalStateException("ดาวน์โหลดโมเดลไม่ได้ HTTP " + code);
            }

            long total = conn.getContentLengthLong();
            if (total <= 0) total = MODEL_SIZE;

            try (InputStream in = new BufferedInputStream(conn.getInputStream(), 1024 * 128);
                 BufferedOutputStream out =
                         new BufferedOutputStream(new FileOutputStream(part), 1024 * 128)) {

                byte[] buf = new byte[1024 * 128];
                long done = 0;
                int lastPct = -1;
                int n;
                while ((n = in.read(buf)) >= 0) {
                    if (n == 0) continue;
                    out.write(buf, 0, n);
                    done += n;
                    int pct = (int) Math.min(94, (done * 94L) / Math.max(1L, total));
                    if (pct != lastPct && callback != null) {
                        lastPct = pct;
                        callback.onProgress(
                                pct,
                                "กำลังติดตั้ง Math OCR " +
                                        Math.max(1, done / (1024 * 1024)) + "/" +
                                        Math.max(1, total / (1024 * 1024)) + " MB"
                        );
                    }
                }
            }

            if (callback != null) callback.onProgress(96, "กำลังตรวจสอบไฟล์โมเดล");
            String hash = sha256(part);
            if (!MODEL_SHA256.equalsIgnoreCase(hash)) {
                part.delete();
                throw new IllegalStateException("โมเดลตรวจสอบไม่ผ่าน กรุณาลองใหม่");
            }

            if (!part.renameTo(modelFile)) {
                copyFile(part, modelFile);
                part.delete();
            }

            if (callback != null) callback.onProgress(100, "ติดตั้ง Math OCR เสร็จแล้ว");
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    public synchronized String recognize(Uri uri) throws Exception {
        if (!isModelReady()) {
            throw new IllegalStateException("ยังไม่ได้ติดตั้งโมเดล Math OCR");
        }

        ensureSession();

        Bitmap bitmap;
        try (InputStream in = context.getContentResolver().openInputStream(uri)) {
            bitmap = BitmapFactory.decodeStream(in);
        }
        if (bitmap == null) throw new IllegalStateException("เปิดรูปไม่ได้");

        Bitmap cropped = cropMargin(bitmap);
        Bitmap prepared = resizeAndPad(cropped);
        float[] input = normalize(prepared);

        if (cropped != bitmap && !cropped.isRecycled()) cropped.recycle();
        if (prepared != bitmap && prepared != cropped && !prepared.isRecycled()) prepared.recycle();
        if (!bitmap.isRecycled()) bitmap.recycle();

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(input),
                new long[]{1, 1, SIZE, SIZE});
             OrtSession.Result result =
                     session.run(Collections.singletonMap(inputName, tensor))) {

            long[] tokenIds = null;

            for (Map.Entry<String, OnnxValue> entry : result) {
                if (!(entry.getValue() instanceof OnnxTensor)) continue;
                OnnxTensor output = (OnnxTensor) entry.getValue();
                LongBuffer lb = output.getLongBuffer();
                if (lb != null) {
                    long[] candidate = new long[lb.remaining()];
                    lb.get(candidate);
                    if (candidate.length > 0) {
                        tokenIds = candidate;
                        break;
                    }
                }
            }

            if (tokenIds == null) {
                throw new IllegalStateException("โมเดลไม่ส่งผลลัพธ์ token");
            }

            return decodeTokens(tokenIds);
        }
    }

    private void ensureSession() throws Exception {
        if (session != null) return;

        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(2,
                Math.min(4, Runtime.getRuntime().availableProcessors())));
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(modelFile.getAbsolutePath(), options);
        inputName = session.getInputNames().iterator().next();
    }

    private Bitmap cropMargin(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w < 2 || h < 2) return src;

        int[] pixels = new int[w * h];
        src.getPixels(pixels, 0, w, 0, 0, w, h);

        int min = 255;
        int max = 0;
        int[] gray = new int[pixels.length];

        for (int i = 0; i < pixels.length; i++) {
            int c = pixels[i];
            int g = Math.round(
                    0.299f * Color.red(c) +
                    0.587f * Color.green(c) +
                    0.114f * Color.blue(c));
            gray[i] = g;
            if (g < min) min = g;
            if (g > max) max = g;
        }

        if (max <= min) return src;

        int minX = w, minY = h, maxX = -1, maxY = -1;
        float range = max - min;

        for (int y = 0; y < h; y++) {
            int row = y * w;
            for (int x = 0; x < w; x++) {
                int norm = Math.round((gray[row + x] - min) * 255f / range);
                if (norm < 200) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX <= minX || maxY <= minY) return src;

        int padX = Math.max(4, (maxX - minX) / 40);
        int padY = Math.max(4, (maxY - minY) / 30);
        minX = Math.max(0, minX - padX);
        minY = Math.max(0, minY - padY);
        maxX = Math.min(w - 1, maxX + padX);
        maxY = Math.min(h - 1, maxY + padY);

        return Bitmap.createBitmap(src, minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private Bitmap resizeAndPad(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();

        float scale = SIZE / (float) Math.max(w, h);
        int nw = Math.max(1, Math.min(SIZE, Math.round(w * scale)));
        int nh = Math.max(1, Math.min(SIZE, Math.round(h * scale)));

        Bitmap scaled = Bitmap.createScaledBitmap(src, nw, nh, true);
        Bitmap out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(out);
        canvas.drawColor(Color.BLACK);

        float left = (SIZE - nw) / 2f;
        float top = (SIZE - nh) / 2f;
        Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        canvas.drawBitmap(scaled, left, top, paint);

        if (scaled != src && !scaled.isRecycled()) scaled.recycle();
        return out;
    }

    private float[] normalize(Bitmap bitmap) {
        int[] pixels = new int[SIZE * SIZE];
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);

        float[] out = new float[SIZE * SIZE];
        for (int i = 0; i < pixels.length; i++) {
            int c = pixels[i];
            float gray =
                    (0.299f * Color.red(c) +
                     0.587f * Color.green(c) +
                     0.114f * Color.blue(c)) / 255f;
            out[i] = (gray - MEAN) / STD;
        }
        return out;
    }

    private void loadTokenizer() throws Exception {
        String json;
        try (InputStream in =
                     context.getAssets().open("models/pp-formulanet-tokenizer.json")) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[32768];
            int n;
            while ((n = in.read(buf)) >= 0) {
                if (n > 0) out.write(buf, 0, n);
            }
            json = out.toString(StandardCharsets.UTF_8.name());
        }

        JSONObject root = new JSONObject(json);
        JSONObject vocabJson = root.getJSONObject("model").getJSONObject("vocab");
        for (String token : vocabJson.keySet()) {
            vocab.put(vocabJson.getInt(token), token);
        }

        JSONArray added = root.optJSONArray("added_tokens");
        if (added != null) {
            for (int i = 0; i < added.length(); i++) {
                JSONObject t = added.getJSONObject(i);
                if (t.optBoolean("special", false)) {
                    specialIds.add(t.getInt("id"));
                }
            }
        }
    }

    private String decodeTokens(long[] ids) {
        StringBuilder encoded = new StringBuilder();

        for (long raw : ids) {
            if (raw == 2) break;
            if (raw < 0 || raw >= 50000) break;
            int id = (int) raw;
            if (specialIds.contains(id)) continue;
            String token = vocab.get(id);
            if (token != null) encoded.append(token);
        }

        String decoded = byteLevelDecode(encoded.toString());
        decoded = decoded.replace("\r", "").trim();

        // Light LaTeX cleanup only; never invent missing math structure here.
        decoded = decoded.replaceAll("\\s+", " ");
        decoded = decoded.replaceAll("(\\\\[A-Za-z]+) \\{", "$1{");
        decoded = decoded.replaceAll("\\{ +", "{");
        decoded = decoded.replaceAll(" +\\}", "}");

        return decoded;
    }

    private void buildByteDecoder() {
        Set<Integer> used = new HashSet<>();
        java.util.ArrayList<Integer> bs = new java.util.ArrayList<>();
        java.util.ArrayList<Integer> cs = new java.util.ArrayList<>();

        for (int b = 33; b <= 126; b++) {
            bs.add(b); cs.add(b); used.add(b);
        }
        for (int b = 161; b <= 172; b++) {
            bs.add(b); cs.add(b); used.add(b);
        }
        for (int b = 174; b <= 255; b++) {
            bs.add(b); cs.add(b); used.add(b);
        }

        int n = 0;
        for (int b = 0; b <= 255; b++) {
            if (!used.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < bs.size(); i++) {
            byteDecoder.put(cs.get(i), bs.get(i));
        }
    }

    private String byteLevelDecode(String encoded) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        encoded.codePoints().forEach(cp -> {
            Integer b = byteDecoder.get(cp);
            if (b != null) {
                out.write(b);
            } else {
                byte[] utf = new String(Character.toChars(cp))
                        .getBytes(StandardCharsets.UTF_8);
                out.write(utf, 0, utf.length);
            }
        });

        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }

    private String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buf = new byte[1024 * 128];
            int n;
            while ((n = in.read(buf)) >= 0) {
                if (n > 0) digest.update(buf, 0, n);
            }
        }
        StringBuilder sb = new StringBuilder();
        for (byte b : digest.digest()) {
            sb.append(String.format("%02x", b & 0xff));
        }
        return sb.toString();
    }

    private void copyFile(File from, File to) throws Exception {
        try (InputStream in = new BufferedInputStream(new FileInputStream(from));
             BufferedOutputStream out =
                     new BufferedOutputStream(new FileOutputStream(to))) {
            byte[] buf = new byte[1024 * 128];
            int n;
            while ((n = in.read(buf)) >= 0) {
                if (n > 0) out.write(buf, 0, n);
            }
        }
    }

    @Override
    public synchronized void close() {
        if (session != null) {
            try { session.close(); } catch (Exception ignored) {}
            session = null;
        }
    }
}
