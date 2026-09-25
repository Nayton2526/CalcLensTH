package com.calclensth.app;

import android.webkit.JavascriptInterface;

public class OcrBridge {
    public interface CameraLauncher { void launchCamera(); }
    private final CameraLauncher launcher;

    public OcrBridge(CameraLauncher launcher) {
        this.launcher = launcher;
    }

    @JavascriptInterface
    public void takePhoto() {
        launcher.launchCamera();
    }
}
