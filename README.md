# Getting TensorFlow Lite Models on Android Apps

[TensorFlow Lite](https://tensorflow.org/lite) makes it very easy to run inference on Android devices. In this guide, you'll learn how to get a pretrained model running in an Android app.

## Setup

First, install [Android Studio](https://developer.android.com/studio/), the official IDE for Android app development. We'll need this to build and run the Android app.

Next, you need to enable developer mode on your Android device:

1. Open the Settings application on your phone.
2. Scroll down to the "About Phone" or "About Device" section.
3. Find the section titled "Build number" and tap it seven times.
4. Now, go back to the main Settings page. You should find "Developer Options" either in the main page, or under the "System" tab.

Now enable USB debugging in the "Developer Options" tab.

## Building the App

Now, let's open up Android Studio and open the project. Click on `Open an existing project` and select `experimental/android_demo/Object detection` as the project folder.

Using a USB cable, plug your Android device into your computer. Then, hit `Build -> Make Project` (or `⌘F9`) to build the project. 

## Running the App on Your Device

Simply hit `Run -> Run 'app'` to run the app on your device. You should now have a working image classifier that performs live person detection.
