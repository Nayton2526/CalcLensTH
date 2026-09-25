package com.calclensth.app;

import android.webkit.JavascriptInterface;

public class OcrBridge {
    public interface ImageLauncher {
        void launchCamera();
        void pickImage();
    }

    private final ImageLauncher launcher;

    public OcrBridge(ImageLauncher launcher) {
        this.launcher = launcher;
    }

    @JavascriptInterface
    public void takePhoto() {
        launcher.launchCamera();
    }

    @JavascriptInterface
    public void chooseImage() {
        launcher.pickImage();
    }
}
