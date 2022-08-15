/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.gaze_estimation;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.examples.gaze_estimation.env.BorderedText;
import org.tensorflow.lite.examples.gaze_estimation.env.ImageUtils;
import org.tensorflow.lite.examples.gaze_estimation.env.Logger;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier.Device;
import org.tensorflow.lite.examples.gaze_estimation.tflite.Classifier.Model;

import com.example.gazedemo1.ProcessFactory;
import com.qualcomm.qti.platformvalidator.PlatformValidator;
import com.qualcomm.qti.platformvalidator.PlatformValidatorUtil;
import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;

import static com.example.gazedemo1.ProcessFactory.bitmap2mat;
import static com.example.gazedemo1.ProcessFactory.gaze_postprocess;
import static com.example.gazedemo1.ProcessFactory.gaze_preprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_postprocess;
import static com.example.gazedemo1.ProcessFactory.landmark_preprocess;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.preprocessing;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.scale;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.transpose;
import static org.tensorflow.lite.examples.gaze_estimation.DetectionUtils.postprocessing;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawbox;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawgaze;
import static org.tensorflow.lite.examples.gaze_estimation.DrawUtils.drawlandmark;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(DemoConfig.Preview_H, DemoConfig.Preview_W);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private BorderedText borderedText;
  private SNPE.NeuralNetworkBuilder builder = null;

  private FloatTensor face_detection_tensor = null;
  private FloatTensor landmark_detection_tensor = null;
  private FloatTensor gaze_estimation_face_tensor = null;
  private FloatTensor gaze_estimation_leye_tensor = null;
  private FloatTensor gaze_estimation_reye_tensor = null;

  private NeuralNetwork face_detection_network, landmark_detection_network, gaze_estimation_network;
  private float[] face_detection_input = new float[DemoConfig.getFaceDetectionInputsize()];
  private float[] NNoutput = null;
  Bitmap detection = null;
  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private NeuralNetwork setupNetwork(String model_path, String model_op_out, NeuralNetwork.Runtime runtime) {
    SNPE.NeuralNetworkBuilder builder = null;
    try {
      builder = new SNPE.NeuralNetworkBuilder(getApplication())
              // Allows selecting a runtime order for the network.
              .setRuntimeOrder(runtime)
              // Loads a model from DLC file
              .setModel(new File(model_path));
      builder.setOutputLayers(model_op_out);
    } catch (IOException e) {
      Log.d("weiming", "build network exception " + model_path);
    }
    NeuralNetwork network = builder.build();
    return network;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap =
        Bitmap.createBitmap(
                DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            DemoConfig.crop_W,
            DemoConfig.crop_H,
            DemoConfig.img_orientation,
            MAINTAIN_ASPECT);


    if(DemoConfig.USE_VERTICAL)
      detection = Bitmap.createBitmap(DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);
    else
      detection = Bitmap.createBitmap(DemoConfig.crop_H, DemoConfig.crop_W, Config.ARGB_8888);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    //create snpe
    //This create platform validator object for GPU runtime class
    PlatformValidator pv = new PlatformValidator(PlatformValidatorUtil.Runtime.GPU);
    // To check in general runtime is working use isRuntimeAvailable
    boolean check = pv.isRuntimeAvailable(getApplication());
    // To check SNPE runtime is working use runtimeCheck
    boolean check2 = pv.runtimeCheck(getApplication());
    //To get core version use libVersion api
    String str = pv.coreVersion(getApplication());
    Log.d("weiming", "coreversion = " + str);

    face_detection_network = setupNetwork(DemoConfig.face_detection_model_path, DemoConfig.face_detection_model_op_out, GPU);
    landmark_detection_network = setupNetwork(DemoConfig.landmark_detection_model_path, DemoConfig.landmark_detection_model_op_out, GPU);
    gaze_estimation_network = setupNetwork(DemoConfig.gaze_estimation_model_path, DemoConfig.gaze_estimation_model_op_out, GPU);
  }

  long inferencetime;
  long latency;
  int bitmapset = 0;
  int[] pixs = new int[DemoConfig.crop_H*DemoConfig.crop_W];
  int[] pixs_out = new int[DemoConfig.crop_H*DemoConfig.crop_W];
  @Override

  protected Bitmap processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    //transform and crop the frame
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    /*
    // DEBUG IMAGE START
    Mat mm = bitmap2mat(BitmapFactory.decodeResource(getResources(), R.drawable.sample480x480));
    Imgproc.resize(mm, mm, new org.opencv.core.Size(480, 480));
    croppedBitmap = Bitmap.createBitmap(DemoConfig.crop_W, DemoConfig.crop_H, Config.ARGB_8888);
    Utils.matToBitmap(mm, croppedBitmap);
    Log.d("BITMAP_SIZE", croppedBitmap.getWidth() + " " + croppedBitmap.getHeight());
    // DEBUG IMAGE END
    */

    DetectionUtils.scaleResult scale_result = scale(croppedBitmap, DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_H);
    cropCopyBitmap = scale_result.scaledBitmap;
    float ratioX = scale_result.ratioX;
    float ratioY = scale_result.ratioY;
    //convert to floating point
    long startTime = SystemClock.uptimeMillis();

    // face detection start
    preprocessing(cropCopyBitmap, face_detection_input);
    face_detection_tensor = face_detection_network.createFloatTensor(1, DemoConfig.face_detection_input_H, DemoConfig.face_detection_input_W, DemoConfig.face_detection_input_C);
    face_detection_tensor.write(face_detection_input, 0, DemoConfig.getFaceDetectionInputsize());
    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    Log.d("weiming", "processImage PRE time: " + lastProcessingTimeMs);
    latency = lastProcessingTimeMs;

    //inference with snpe
    final Map<String, FloatTensor> inputsMap = new HashMap<>();
    inputsMap.put("full_image", face_detection_tensor);
    startTime = SystemClock.uptimeMillis();
    final Map<String, FloatTensor> outputsMap = face_detection_network.execute(inputsMap);
    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    inferencetime = lastProcessingTimeMs;
    latency += lastProcessingTimeMs;

    startTime = SystemClock.uptimeMillis();
    float[][] boxes = null;
    for (Map.Entry<String, FloatTensor> output : outputsMap.entrySet()) {
      final FloatTensor tensor = output.getValue();
      if (output.getKey().equals(DemoConfig.face_detection_model_out)) {
        NNoutput = new float[tensor.getSize()];
        tensor.read(NNoutput, 0, NNoutput.length);
        Log.d("Weiming", "processImage get floating point outputs " + tensor.getSize());
        if (output.getKey().equals(DemoConfig.face_detection_model_out)) {
          Log.d("Weiming", "processImage yolox get output");
          boxes = postprocessing(NNoutput, ratioX, ratioY);
          break;
        }
      }
    }
    // face detection end

    Mat img = bitmap2mat(croppedBitmap);

    List<float[]> landmarks = new ArrayList();
    List<float[]> gazes = new ArrayList();
    // landmark detection start
    if (boxes != null && boxes.length != 0 && boxes[0][0] != 0) {
      for (float[] box : boxes) {
        if (box[0] == 0 && box[1] == 0 && box[2] == 0 && box[3] == 0)
          continue;
        ProcessFactory.LandmarkPreprocessResult landmark_preprocess_result = landmark_preprocess(img, box);
        float[] landmark_detection_input = landmark_preprocess_result.input;
        landmark_detection_tensor = landmark_detection_network.createFloatTensor(1, DemoConfig.landmark_detection_input_H, DemoConfig.landmark_detection_input_W, DemoConfig.landmark_detection_input_C);
        landmark_detection_tensor.write(landmark_detection_input, 0, DemoConfig.getLandmarkDetectionInputsize());
        final Map<String, FloatTensor> LandmarkInputsMap = new HashMap<>();
        LandmarkInputsMap.put("face_image", landmark_detection_tensor);
        final Map<String, FloatTensor> LandmarkOutputsMap = landmark_detection_network.execute(LandmarkInputsMap);
        for (Map.Entry<String, FloatTensor> output : LandmarkOutputsMap.entrySet()) {
          final FloatTensor tensor = output.getValue();
          if (output.getKey().equals(DemoConfig.landmark_detection_model_out)) {
            NNoutput = new float[tensor.getSize()];
            tensor.read(NNoutput, 0, NNoutput.length);
            Log.d("landmark_detection", String.valueOf(NNoutput.length));
            float[] landmark = landmark_postprocess(landmark_preprocess_result, NNoutput);
            landmarks.add(landmark);
            ProcessFactory.GazePreprocessResult gaze_preprocess_result = gaze_preprocess(img, landmark);
            gaze_estimation_face_tensor = gaze_estimation_network.createFloatTensor(1, DemoConfig.gaze_estimation_face_input_H, DemoConfig.gaze_estimation_face_input_W, DemoConfig.gaze_estimation_face_input_C);
            gaze_estimation_leye_tensor = gaze_estimation_network.createFloatTensor(1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C);
            gaze_estimation_reye_tensor = gaze_estimation_network.createFloatTensor(1, DemoConfig.gaze_estimation_eye_input_H, DemoConfig.gaze_estimation_eye_input_W, DemoConfig.gaze_estimation_eye_input_C);
            gaze_estimation_face_tensor.write(gaze_preprocess_result.face, 0, DemoConfig.getGazeEstimationFaceInputsize());
            gaze_estimation_leye_tensor.write(gaze_preprocess_result.leye, 0, DemoConfig.getGazeEstimationEyeInputsize());
            gaze_estimation_reye_tensor.write(gaze_preprocess_result.reye, 0, DemoConfig.getGazeEstimationEyeInputsize());

            final Map<String, FloatTensor> GazeInputsMap = new HashMap<>();
            GazeInputsMap.put("left_eye", gaze_estimation_leye_tensor);
            GazeInputsMap.put("right_eye", gaze_estimation_reye_tensor);
            GazeInputsMap.put("face", gaze_estimation_face_tensor);
            final Map<String, FloatTensor> GazeOutputsMap = gaze_estimation_network.execute(GazeInputsMap);
            for (Map.Entry<String, FloatTensor> gaze_output : GazeOutputsMap.entrySet()) {
              final FloatTensor gaze_tensor = gaze_output.getValue();
              if (gaze_output.getKey().equals(DemoConfig.gaze_estimation_model_out)) {
                NNoutput = new float[gaze_tensor.getSize()];
                gaze_tensor.read(NNoutput, 0, NNoutput.length);
                Log.d("gaze_estimation", String.valueOf(NNoutput[0])+" "+String.valueOf(NNoutput[1])+" "+String.valueOf(NNoutput.length));
                float[] gaze_pitchyaw = gaze_postprocess(NNoutput, gaze_preprocess_result.R);
                gazes.add(gaze_pitchyaw);
              }
            }
          }
        }
      }
      // landamrk detection end

      for (int i=0;i<gazes.size();i++) {
        drawgaze(img, gazes.get(i), landmarks.get(i));
      }
      Utils.matToBitmap(img, croppedBitmap);
      croppedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
      drawbox(pixs, boxes, DemoConfig.crop_H, DemoConfig.crop_W);
      for (float[] landmark : landmarks) {
        drawlandmark(pixs, landmark, DemoConfig.crop_H, DemoConfig.crop_W);
      }
    } else {
      croppedBitmap.getPixels(pixs,0, DemoConfig.crop_W, 0, 0,DemoConfig.crop_W, DemoConfig.crop_H);
    }

    Log.d("weiming","load: " + (SystemClock.uptimeMillis() - startTime));
    //transpose
    if(!DemoConfig.USE_VERTICAL) {
      transpose(pixs, pixs_out, DemoConfig.crop_H, DemoConfig.crop_W);
      detection.setPixels(pixs_out, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
    }
    else{
      detection.setPixels(pixs, 0, DemoConfig.crop_H, 0, 0, DemoConfig.crop_H, DemoConfig.crop_W);
    }
    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    latency+=lastProcessingTimeMs;
    Log.d("weiming","POST: " + lastProcessingTimeMs);

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            if (classifier != null) {
              final long startTime = SystemClock.uptimeMillis();
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

              if (bitmapset == 0){
                runOnUiThread(
                        new Runnable() {
                          @Override
                          public void run() {
                            ImageView imageView = (ImageView) findViewById(R.id.imageView2);
                            imageView.setImageBitmap(detection);
                          }
                        });
//                bitmapset = 1;
              }
              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      showFrameInfo(rgbFrameBitmap.getWidth() + "x" + rgbFrameBitmap.getHeight());
                      showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                      showCameraResolution(canvas.getWidth() + "x" + canvas.getHeight());
                      showRotationInfo(String.valueOf(sensorOrientation));
                      showInference(latency + "/" + inferencetime + "ms");
                    }
                  });
            }
            readyForNextImage();
          }
        });
    return croppedBitmap;
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (croppedBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU && model == Model.QUANTIZED) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                .show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      classifier = Classifier.create(this, model, device, numThreads);
    } catch (IOException e) {
      LOGGER.e(e, "Failed to create classifier.");
    }
  }
}
