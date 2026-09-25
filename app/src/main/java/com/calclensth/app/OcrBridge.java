package com.calclensth.app;

import android.webkit.JavascriptInterface;

public class OcrBridge {
    public interface ImageLauncher {
        void launchCamera(String mode);
        void pickImage(String mode);
    }

    private final ImageLauncher launcher;

    public OcrBridge(ImageLauncher launcher) {
        this.launcher = launcher;
    }

    @JavascriptInterface
    public void takePhoto(String mode) {
        launcher.launchCamera(mode);
    }

    @JavascriptInterface
    public void chooseImage(String mode) {
        launcher.pickImage(mode);
    }
}
