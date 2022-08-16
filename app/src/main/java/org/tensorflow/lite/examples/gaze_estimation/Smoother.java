package org.tensorflow.lite.examples.gaze_estimation;

import java.util.Vector;

class SingleValueSmoother {
    private double t_prev = -1.0;
    private double dx_prev = 0.0;
    private double x_prev = 0.0;
    private double d_cutoff = 1.0;
    private double beta = 0.01;
    private double min_cutoff = 1.0;

    public SingleValueSmoother(double t0, double x0, double dx0, double min_cutoff, double beta, double d_cutoff) {
        this.min_cutoff = min_cutoff;
        this.beta = beta;
        this.d_cutoff = d_cutoff;
        this.x_prev = x0;
        this.dx_prev = dx0;
        this.t_prev = t0;
    }
    private double smoothing_factor(double t_e, double cutoff) {
        double r = 2.0 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }
    private double exponential_smoothing(double a, double x, double x_prev) {
        return a * x + (1 - a) * x_prev;
    }
    public double record_and_smooth(double x, double t) {
        if (this.t_prev < 0.0) {
            this.t_prev = t;
            this.x_prev = x;
            return x;
        }
        double t_e = t - this.t_prev;

        double a_d = this.smoothing_factor(t_e, this.d_cutoff);
        double dx = (x - this.x_prev) / t_e;
        double dx_hat = this.exponential_smoothing(a_d, dx, this.dx_prev);

        double cutoff = this.min_cutoff + this.beta * Math.abs(dx_hat);
        double a = this.smoothing_factor(t_e, cutoff);
        double x_hat = this.exponential_smoothing(a, x, this.x_prev);

        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}

public class Smoother {
    private Vector<SingleValueSmoother> filters;
    private int num = 0;
    public double[] values;
    public int tick = 0;
    public Smoother(int num, double face_min_cutoff, double face_beta, double lmk_min_cutoff, double lmk_beta, double gaze_min_cutoff, double gaze_beta) {
        this.filters = new Vector<SingleValueSmoother>(num);
        this.num = num;
        this.tick = 0;
        for (int i=0;i<num;i++) {
            if (0 <= i && i < 4) {
                this.filters.addElement(new SingleValueSmoother(-1.0, 0.0, 0.0, face_min_cutoff, face_beta, 1.0));
            } else if (4 <= i && i < 98*2) {
                this.filters.addElement(new SingleValueSmoother(-1.0, 0.0, 0.0, lmk_min_cutoff, lmk_beta, 1.0));
            } else {
                this.filters.addElement(new SingleValueSmoother(-1.0, 0.0, 0.0, gaze_min_cutoff, gaze_beta, 1.0));
            }
        }
        this.values = new double[num];
    }
    public double[] record_and_smooth(double[] values, double t) {
        for (int i=0;i<this.num;i++) {
            this.values[i] = this.filters.elementAt(i).record_and_smooth(values[i], t);
        }
        double[] ret = new double[this.num];
        for (int i=0;i<this.num;i++) {
            ret[i] = this.values[i];
        }
        return ret;
    }
    public final static double[] get_feature(double[] values) {
        double area_sqrt = Math.sqrt(Math.abs(values[2] - values[0]) * Math.abs(values[3] - values[1]));
        double cx = (values[2] + values[0]) / 2;
        double cy = (values[3] + values[1]) / 2;
        return new double[]{area_sqrt, cx, cy};
    }
    public final static double[] concat_prediction(float[] face_bbox, float[] landmark, float[] gaze_pitchyaw) {
        double[] values = new double[4+98*2+2];
        for (int i=0;i<4;i++)
            values[i] = (double)face_bbox[i];
        for (int i=0;i<98*2;i++)
            values[4+i] = (double)landmark[i];
        for (int i=0;i<2;i++)
            values[4+98*2+i] = (double)gaze_pitchyaw[i];
        return values;
    }
    public final static void split_prediction(double[] prediction, float[] face_bbox, float[] landmark, float[] gaze_pitchyaw) {
        for (int i=0;i<4;i++)
            face_bbox[i] = (float)prediction[i];
        for (int i=0;i<98*2;i++)
            landmark[i] = (float)prediction[4+i];
        for (int i=0;i<2;i++)
            gaze_pitchyaw[i] = (float)prediction[4+98*2+i];
    }
}

