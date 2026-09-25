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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
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

    private static final int LAYOUT_SIZE = 480;
    private static final String LAYOUT_NAME = "pp-doclayout-s.onnx";
    private static final String LAYOUT_URL =
            "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-s.onnx";
    private static final String LAYOUT_SHA256 =
            "c2336493a0a13cd9b9b457ca68aea370b327c362a4a7da4917c2bba96029bceb";
    private static final long LAYOUT_MODEL_SIZE = 4914918L;
    private static final int FORMULA_CLASS_ID = 7;

    private final Context context;
    private final File modelFile;
    private final File layoutModelFile;
    private final Map<Integer, String> vocab = new HashMap<>();
    private final Set<Integer> specialIds = new HashSet<>();
    private final Map<Integer, Integer> byteDecoder = new HashMap<>();

    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;
    private OrtSession layoutSession;
    private String layoutImageInput;
    private String layoutScaleInput;

    public FormulaOcrEngine(Context context) throws Exception {
        this.context = context.getApplicationContext();
        File dir = new File(this.context.getFilesDir(), "models");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("สร้างโฟลเดอร์โมเดลไม่ได้");
        }
        modelFile = new File(dir, MODEL_NAME);
        layoutModelFile = new File(dir, LAYOUT_NAME);
        loadTokenizer();
        buildByteDecoder();
    }

    public boolean isModelReady() {
        return modelFile.exists() && modelFile.length() == MODEL_SIZE
                && layoutModelFile.exists() && layoutModelFile.length() == LAYOUT_MODEL_SIZE;
    }

    public long getModelSizeBytes() {
        return MODEL_SIZE + LAYOUT_MODEL_SIZE;
    }

    public synchronized void ensureModel(ProgressCallback callback) throws Exception {
        boolean formulaReady = modelFile.exists() && modelFile.length() == MODEL_SIZE;
        boolean layoutReady = layoutModelFile.exists() && layoutModelFile.length() == LAYOUT_MODEL_SIZE;

        if (formulaReady && layoutReady) {
            if (callback != null) callback.onProgress(100, "โมเดล Math OCR พร้อมใช้งาน");
            return;
        }

        if (!formulaReady) {
            downloadAndVerify(
                    MODEL_URL,
                    modelFile,
                    MODEL_SIZE,
                    MODEL_SHA256,
                    callback,
                    0,
                    96,
                    "กำลังดาวน์โหลด Math OCR ครั้งแรก ~221 MB"
            );
        }

        if (!layoutReady) {
            downloadAndVerify(
                    LAYOUT_URL,
                    layoutModelFile,
                    LAYOUT_MODEL_SIZE,
                    LAYOUT_SHA256,
                    callback,
                    96,
                    100,
                    "กำลังติดตั้งตัวตรวจหาสูตรในหน้าเอกสาร ~5 MB"
            );
        }

        if (callback != null) callback.onProgress(100, "ติดตั้ง Math OCR เสร็จแล้ว");
    }

    private void downloadAndVerify(
            String urlString,
            File target,
            long expectedSize,
            String expectedSha,
            ProgressCallback callback,
            int progressStart,
            int progressEnd,
            String startMessage
    ) throws Exception {
        if (target.exists() && target.length() == expectedSize) return;
        if (target.exists()) target.delete();

        File part = new File(target.getParentFile(), target.getName() + ".part");
        if (part.exists()) part.delete();

        if (callback != null) callback.onProgress(progressStart, startMessage);

        HttpURLConnection conn = null;
        try {
            URL url = new URL(urlString);
            conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(20000);
            conn.setReadTimeout(30000);
            conn.setInstanceFollowRedirects(true);
            conn.setRequestProperty("User-Agent", "CalcLensTH/0.4.1");
            conn.connect();

            int code = conn.getResponseCode();
            if (code < 200 || code >= 300) {
                throw new IllegalStateException("ดาวน์โหลดโมเดลไม่ได้ HTTP " + code);
            }

            long total = conn.getContentLengthLong();
            if (total <= 0) total = expectedSize;

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

                    int span = Math.max(1, progressEnd - progressStart);
                    int pct = progressStart +
                            (int) Math.min(span - 1L, (done * Math.max(1, span - 1L)) / Math.max(1L, total));

                    if (pct != lastPct && callback != null) {
                        lastPct = pct;
                        callback.onProgress(
                                pct,
                                startMessage + " • " +
                                        Math.max(1, done / (1024 * 1024)) + "/" +
                                        Math.max(1, total / (1024 * 1024)) + " MB"
                        );
                    }
                }
            }

            String hash = sha256(part);
            if (!expectedSha.equalsIgnoreCase(hash)) {
                part.delete();
                throw new IllegalStateException("โมเดลตรวจสอบไม่ผ่าน กรุณาลองใหม่");
            }

            if (!part.renameTo(target)) {
                copyFile(part, target);
                part.delete();
            }
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

        Bitmap formulaRegion = detectTopFormulaRegion(bitmap);
        Bitmap cropped = cropMargin(formulaRegion);
        Bitmap prepared = resizeAndPad(cropped);
        float[] input = normalize(prepared);

        if (formulaRegion != bitmap && !formulaRegion.isRecycled()) formulaRegion.recycle();
        if (cropped != bitmap && cropped != formulaRegion && !cropped.isRecycled()) cropped.recycle();
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

    private Bitmap detectTopFormulaRegion(Bitmap src) throws Exception {
        ensureLayoutSession();

        int srcW = src.getWidth();
        int srcH = src.getHeight();
        if (srcW < 2 || srcH < 2) return src;

        Bitmap scaled = Bitmap.createScaledBitmap(src, LAYOUT_SIZE, LAYOUT_SIZE, true);
        int[] pixels = new int[LAYOUT_SIZE * LAYOUT_SIZE];
        scaled.getPixels(pixels, 0, LAYOUT_SIZE, 0, 0, LAYOUT_SIZE, LAYOUT_SIZE);

        float[] input = new float[3 * LAYOUT_SIZE * LAYOUT_SIZE];
        int plane = LAYOUT_SIZE * LAYOUT_SIZE;
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            input[i] = Color.red(color) / 255f;
            input[plane + i] = Color.green(color) / 255f;
            input[2 * plane + i] = Color.blue(color) / 255f;
        }
        if (scaled != src && !scaled.isRecycled()) scaled.recycle();

        float scaleY = LAYOUT_SIZE / (float) srcH;
        float scaleX = LAYOUT_SIZE / (float) srcW;

        try (OnnxTensor imageTensor = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(input),
                    new long[]{1, 3, LAYOUT_SIZE, LAYOUT_SIZE});
             OnnxTensor scaleTensor = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(new float[]{scaleY, scaleX}),
                    new long[]{1, 2})) {

            Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
            inputs.put(layoutImageInput, imageTensor);
            inputs.put(layoutScaleInput, scaleTensor);

            try (OrtSession.Result result = layoutSession.run(inputs)) {
                List<FormulaBox> boxes = new ArrayList<>();

                for (Map.Entry<String, OnnxValue> entry : result) {
                    if (!(entry.getValue() instanceof OnnxTensor)) continue;
                    OnnxTensor tensor = (OnnxTensor) entry.getValue();
                    long[] shape = tensor.getInfo().getShape();
                    if (shape == null || shape.length < 2 || shape[shape.length - 1] < 6) continue;

                    FloatBuffer fb = tensor.getFloatBuffer();
                    if (fb == null) continue;

                    float[] data = new float[fb.remaining()];
                    fb.get(data);
                    int stride = (int) shape[shape.length - 1];
                    if (stride < 6) continue;

                    for (int off = 0; off + 5 < data.length; off += stride) {
                        int classId = Math.round(data[off]);
                        float score = data[off + 1];
                        if (classId != FORMULA_CLASS_ID || score < 0.25f) continue;

                        float x1 = data[off + 2];
                        float y1 = data[off + 3];
                        float x2 = data[off + 4];
                        float y2 = data[off + 5];

                        if (x2 <= 1.05f && y2 <= 1.05f &&
                                x1 >= -0.05f && y1 >= -0.05f) {
                            x1 *= srcW;
                            x2 *= srcW;
                            y1 *= srcH;
                            y2 *= srcH;
                        }

                        x1 = clamp(x1, 0, srcW - 1);
                        x2 = clamp(x2, 0, srcW);
                        y1 = clamp(y1, 0, srcH - 1);
                        y2 = clamp(y2, 0, srcH);

                        if (x2 - x1 < 8 || y2 - y1 < 8) continue;
                        boxes.add(new FormulaBox(x1, y1, x2, y2, score));
                    }
                }

                if (boxes.isEmpty()) {
                    // A manual crop that is already a single formula should still work.
                    if (looksLikeSingleFormulaCrop(src)) return src;
                    throw new IllegalStateException(
                            "ยังไม่พบสูตรชัดเจนในกรอบ กรุณาครอปให้เหลือโจทย์หนึ่งข้อ"
                    );
                }

                // For a worked example/page, the requested exercise is normally the
                // first formula in reading order; solved steps appear below it.
                boxes.sort(Comparator
                        .comparingDouble((FormulaBox b) -> b.y1)
                        .thenComparing((a, b) -> Float.compare(b.score, a.score)));

                FormulaBox best = boxes.get(0);

                // Merge nearby formula boxes on the same top line, if the detector
                // split one expression into adjacent pieces.
                float lineCenter = (best.y1 + best.y2) * 0.5f;
                float lineHeight = Math.max(1f, best.y2 - best.y1);
                float left = best.x1, top = best.y1, right = best.x2, bottom = best.y2;
                for (int i = 1; i < boxes.size(); i++) {
                    FormulaBox b = boxes.get(i);
                    float center = (b.y1 + b.y2) * 0.5f;
                    if (Math.abs(center - lineCenter) <= lineHeight * 0.7f) {
                        left = Math.min(left, b.x1);
                        top = Math.min(top, b.y1);
                        right = Math.max(right, b.x2);
                        bottom = Math.max(bottom, b.y2);
                    }
                }

                int padX = Math.max(6, Math.round((right - left) * 0.06f));
                int padY = Math.max(6, Math.round((bottom - top) * 0.18f));

                int l = Math.max(0, Math.round(left) - padX);
                int t = Math.max(0, Math.round(top) - padY);
                int r = Math.min(srcW, Math.round(right) + padX);
                int b = Math.min(srcH, Math.round(bottom) + padY);

                if (r <= l || b <= t) return src;
                return Bitmap.createBitmap(src, l, t, r - l, b - t);
            }
        }
    }

    private boolean looksLikeSingleFormulaCrop(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (w <= 0 || h <= 0) return false;
        if (h <= w * 0.58f) return true;

        // Count horizontal ink bands. A full worked solution has many separated
        // lines; one formula usually has only a few (fractions/superscripts allowed).
        int sampleW = Math.min(w, 1000);
        int sampleH = Math.min(h, 1000);
        Bitmap small = Bitmap.createScaledBitmap(src, sampleW, sampleH, true);
        int[] px = new int[sampleW * sampleH];
        small.getPixels(px, 0, sampleW, 0, 0, sampleW, sampleH);
        if (small != src && !small.isRecycled()) small.recycle();

        int bands = 0;
        boolean inBand = false;
        int gap = 0;
        for (int y = 0; y < sampleH; y++) {
            int dark = 0;
            int row = y * sampleW;
            for (int x = 0; x < sampleW; x++) {
                int c = px[row + x];
                int gray = Math.round(
                        0.299f * Color.red(c) +
                        0.587f * Color.green(c) +
                        0.114f * Color.blue(c));
                if (gray < 185) dark++;
            }
            boolean hasInk = dark > Math.max(2, sampleW / 250);
            if (hasInk) {
                if (!inBand) {
                    bands++;
                    inBand = true;
                }
                gap = 0;
            } else if (inBand) {
                gap++;
                if (gap >= 4) {
                    inBand = false;
                    gap = 0;
                }
            }
        }
        return bands <= 4;
    }

    private void ensureLayoutSession() throws Exception {
        if (layoutSession != null) return;

        if (env == null) env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(
                2,
                Math.min(4, Runtime.getRuntime().availableProcessors())
        ));
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        layoutSession = env.createSession(layoutModelFile.getAbsolutePath(), options);

        for (String name : layoutSession.getInputNames()) {
            String low = name.toLowerCase();
            if (low.contains("scale")) layoutScaleInput = name;
            else if (low.contains("image")) layoutImageInput = name;
        }

        if (layoutImageInput == null || layoutScaleInput == null) {
            String[] names = layoutSession.getInputNames().toArray(new String[0]);
            if (names.length >= 2) {
                layoutImageInput = names[0];
                layoutScaleInput = names[1];
            } else {
                throw new IllegalStateException("โครงสร้างโมเดลตรวจหาสูตรไม่ถูกต้อง");
            }
        }
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(value, max));
    }

    private static class FormulaBox {
        final float x1, y1, x2, y2, score;
        FormulaBox(float x1, float y1, float x2, float y2, float score) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.score = score;
        }
    }

    private void ensureSession() throws Exception {
        if (session != null) return;

        if (env == null) env = OrtEnvironment.getEnvironment();
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
        java.util.Iterator<String> keys = vocabJson.keys();
        while (keys.hasNext()) {
            String token = keys.next();
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
        if (layoutSession != null) {
            try { layoutSession.close(); } catch (Exception ignored) {}
            layoutSession = null;
        }
    }
}
