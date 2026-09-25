package com.calclensth.app;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class CropView extends View {
    private Bitmap bitmap;
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF imageRect = new RectF();
    private final RectF cropRect = new RectF();

    private float scale = 1f;
    private int mode = 0; // 0 none, 1 move, 2 TL, 3 TR, 4 BL, 5 BR
    private float lastX, lastY;
    private float handleRadius;
    private float minCrop;

    public CropView(Context context) {
        super(context);
        init();
    }

    public CropView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        float d = getResources().getDisplayMetrics().density;
        handleRadius = 28f * d;
        minCrop = 70f * d;
        setBackgroundColor(Color.BLACK);
    }

    public void setBitmap(Bitmap value) {
        bitmap = value;
        requestLayout();
        invalidate();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        recalcImageRect();
    }

    private void recalcImageRect() {
        if (bitmap == null || getWidth() <= 0 || getHeight() <= 0) return;

        float bw = bitmap.getWidth();
        float bh = bitmap.getHeight();
        scale = Math.min(getWidth() / bw, getHeight() / bh);

        float dw = bw * scale;
        float dh = bh * scale;
        float left = (getWidth() - dw) / 2f;
        float top = (getHeight() - dh) / 2f;

        imageRect.set(left, top, left + dw, top + dh);

        float mx = dw * 0.08f;
        float my = dh * 0.08f;
        cropRect.set(
                imageRect.left + mx,
                imageRect.top + my,
                imageRect.right - mx,
                imageRect.bottom - my
        );
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (bitmap == null) return;

        canvas.drawBitmap(bitmap, null, imageRect, paint);

        paint.setColor(0x99000000);
        canvas.drawRect(imageRect.left, imageRect.top, imageRect.right, cropRect.top, paint);
        canvas.drawRect(imageRect.left, cropRect.bottom, imageRect.right, imageRect.bottom, paint);
        canvas.drawRect(imageRect.left, cropRect.top, cropRect.left, cropRect.bottom, paint);
        canvas.drawRect(cropRect.right, cropRect.top, imageRect.right, cropRect.bottom, paint);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4f * getResources().getDisplayMetrics().density);
        paint.setColor(Color.WHITE);
        canvas.drawRect(cropRect, paint);

        paint.setStyle(Paint.Style.FILL);
        float r = 7f * getResources().getDisplayMetrics().density;
        drawHandle(canvas, cropRect.left, cropRect.top, r);
        drawHandle(canvas, cropRect.right, cropRect.top, r);
        drawHandle(canvas, cropRect.left, cropRect.bottom, r);
        drawHandle(canvas, cropRect.right, cropRect.bottom, r);
    }

    private void drawHandle(Canvas canvas, float x, float y, float r) {
        paint.setColor(Color.WHITE);
        canvas.drawCircle(x, y, r, paint);
    }

    private float dist(float x1, float y1, float x2, float y2) {
        float dx = x1 - x2, dy = y1 - y2;
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (bitmap == null) return false;

        float x = event.getX();
        float y = event.getY();

        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (dist(x, y, cropRect.left, cropRect.top) < handleRadius) mode = 2;
            else if (dist(x, y, cropRect.right, cropRect.top) < handleRadius) mode = 3;
            else if (dist(x, y, cropRect.left, cropRect.bottom) < handleRadius) mode = 4;
            else if (dist(x, y, cropRect.right, cropRect.bottom) < handleRadius) mode = 5;
            else if (cropRect.contains(x, y)) mode = 1;
            else mode = 0;

            lastX = x;
            lastY = y;
            return true;
        }

        if (event.getAction() == MotionEvent.ACTION_MOVE) {
            float dx = x - lastX;
            float dy = y - lastY;

            if (mode == 1) {
                moveCrop(dx, dy);
            } else if (mode >= 2) {
                resizeCrop(x, y);
            }

            lastX = x;
            lastY = y;
            invalidate();
            return true;
        }

        if (event.getAction() == MotionEvent.ACTION_UP ||
                event.getAction() == MotionEvent.ACTION_CANCEL) {
            mode = 0;
            return true;
        }

        return true;
    }

    private void moveCrop(float dx, float dy) {
        float w = cropRect.width();
        float h = cropRect.height();

        float left = cropRect.left + dx;
        float top = cropRect.top + dy;

        left = Math.max(imageRect.left, Math.min(left, imageRect.right - w));
        top = Math.max(imageRect.top, Math.min(top, imageRect.bottom - h));

        cropRect.set(left, top, left + w, top + h);
    }

    private void resizeCrop(float x, float y) {
        x = Math.max(imageRect.left, Math.min(x, imageRect.right));
        y = Math.max(imageRect.top, Math.min(y, imageRect.bottom));

        switch (mode) {
            case 2:
                cropRect.left = Math.min(x, cropRect.right - minCrop);
                cropRect.top = Math.min(y, cropRect.bottom - minCrop);
                break;
            case 3:
                cropRect.right = Math.max(x, cropRect.left + minCrop);
                cropRect.top = Math.min(y, cropRect.bottom - minCrop);
                break;
            case 4:
                cropRect.left = Math.min(x, cropRect.right - minCrop);
                cropRect.bottom = Math.max(y, cropRect.top + minCrop);
                break;
            case 5:
                cropRect.right = Math.max(x, cropRect.left + minCrop);
                cropRect.bottom = Math.max(y, cropRect.top + minCrop);
                break;
        }

        cropRect.left = Math.max(imageRect.left, cropRect.left);
        cropRect.top = Math.max(imageRect.top, cropRect.top);
        cropRect.right = Math.min(imageRect.right, cropRect.right);
        cropRect.bottom = Math.min(imageRect.bottom, cropRect.bottom);
    }

    public Bitmap createCroppedBitmap() {
        if (bitmap == null) return null;

        int left = Math.max(0, Math.round((cropRect.left - imageRect.left) / scale));
        int top = Math.max(0, Math.round((cropRect.top - imageRect.top) / scale));
        int right = Math.min(bitmap.getWidth(), Math.round((cropRect.right - imageRect.left) / scale));
        int bottom = Math.min(bitmap.getHeight(), Math.round((cropRect.bottom - imageRect.top) / scale));

        int width = Math.max(1, right - left);
        int height = Math.max(1, bottom - top);

        return Bitmap.createBitmap(bitmap, left, top, width, height);
    }
}
