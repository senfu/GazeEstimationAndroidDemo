package org.tensorflow.lite.examples.gaze_estimation;

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.graphics.Color;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class DrawUtils {

    final static int line_pad = 1;
    public static void drawbox(int[] pixel, float[][] boxes, int res_h, int res_w){
        //color of box
        int G_color = Color.rgb(0,255,0);
        int R_color = Color.rgb(255,0,0);
        int B_color = Color.rgb(255,255,0);
        Log.d("yolox_output_boxes", String.valueOf(boxes.length));
        for(int b = 0; b < boxes.length; b++){
            int box_color = G_color;
            float score = boxes[b][4];
            Log.d("yolox_output_score", String.valueOf(score));
            int x1 = (int)(boxes[b][0]);
            int y1 = (int)(boxes[b][1]);
            int x2 = (int)(boxes[b][2]);
            int y2 = (int)(boxes[b][3]);
            Log.d("yolox_output_coord", String.valueOf(x1)+" "+String.valueOf(y1)+" "+String.valueOf(x2)+" "+String.valueOf(y2));

            for (int l = line_pad * -1; l <= line_pad; l++) {
                x1 = max(min(x1 + l, res_w - 1), 0);
                x2 = max(min(x2 + l, res_w - 1), 0);
                y1 = max(min(y1 + l, res_h - 1), 0);
                y2 = max(min(y2 + l, res_h - 1), 0);
                int x, y;
                //left
                x = x1;
                for (y = y1; y < y2; y++) {
                    pixel[x + y * res_w] = box_color;
                    pixel[max(min((x+1),res_w-1), 0) + y * res_w] = box_color; // line width = 2
                }
                //right
                x = x2;
                for (y = y1; y < y2; y++) {
                    pixel[x + y * res_w] = box_color;
                    pixel[max(min((x+1),res_w-1), 0) + y * res_w] = box_color;
                }
                //top
                y = y2;
                for (x = x1; x < x2; x++) {
                    pixel[x + y * res_w] = box_color;
                    pixel[x + max(min((y+1),res_h-1), 0)  * res_w] = box_color;
                }
                //buttom
                y = y1;
                for (x = x1; x < x2; x++) {
                    pixel[x + y * res_w] = box_color;
                    pixel[x + max(min((y+1),res_h-1), 0) * res_w] = box_color;
                }
            }
        }
    }

    private static void draw_circle(int[] raw_data, int res_h, int res_w, int x, int y){
        int A = 255;
        int R = 255;
        int G = 0;
        int B = 0;
        int box_color = Color.rgb(R,G,B);
        final int box_s = 2;

        for(int i = -box_s; i < box_s; i++){
            for(int j = -box_s; j < box_s; j++){
                if(i + x >= 0 && i + x < res_w && j + y >=0 && j + y < res_h)
                    raw_data[x+i+ (y+j) * res_w] = box_color;
            }
        }
    }

    public static void drawlandmark(int[] pixel, float[] landmark, int res_h, int res_w) {
        if (landmark == null || landmark.length == 0) {
            return;
        }
        for (int i=0;i<landmark.length/2;i++) {
            int x = (int)landmark[i*2];
            int y = (int)landmark[i*2+1];
            if (x < 0 || x >= res_w || y < 0 || y >= res_h)
                continue;
            draw_circle(pixel, res_h, res_w, x, y);
        }
    }

    private final static double length = 100.0f;
    private final static int thickness = 2;
    public static void drawgaze(Mat img, float[] pitchyaw, float[] landmark) {
        double[] eye_pos = new double[]{(landmark[96*2]+landmark[97*2])/2.0, (landmark[96*2+1]+landmark[97*2+1])/2.0};
        int dx = (int)(-length * Math.sin(pitchyaw[1]) * Math.cos(pitchyaw[0]));
        int dy = (int)(-length * Math.sin(pitchyaw[0]));
        Point pt1 = new Point(eye_pos);
        Point pt2 = new Point(eye_pos[0] + dx, eye_pos[1] + dy);
        Imgproc.arrowedLine(img, pt1, pt2, new Scalar(0, 255, 255), thickness);
    }
}
