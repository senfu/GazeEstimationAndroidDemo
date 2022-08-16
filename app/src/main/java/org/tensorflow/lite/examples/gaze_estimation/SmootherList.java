package org.tensorflow.lite.examples.gaze_estimation;

import static org.tensorflow.lite.examples.gaze_estimation.Smoother.get_feature;

import android.util.Log;

import java.util.Iterator;
import java.util.List;
import java.util.Vector;

public class SmootherList {
    public Vector<Smoother> smoothers = new Vector<Smoother>();

    public void addOne() {
        this.smoothers.addElement(new Smoother(4 + 98 * 2 + 2, 0.0, 1.0, 0.1, 1.0, 0.01, 0.8));
    }

    private static double calc_diff(double[] p1, double[] p2) {
        double x = 0.0;
        for (int i=0;i<p1.length;i++)
            x += Math.abs(p1[i] - p2[i]);
        return x;
    }

    public final static double DIFF_THR = 500.0;
    public final static int MAX_TICK = 20;
    public Vector<double[]> smooth_and_update(Vector<double[]> data, double t) {
        Log.d("SMOOTHING_MATCH", "======================================================");
        if (data.size() != 0) {
            boolean[] data_paired = new boolean[data.size()];
            double[][] data_feature = new double[data.size()][3];
            for (int i=0;i<data.size();i++)
                data_feature[i] = get_feature(data.elementAt(i));

            int s_cnt = -1;
            for (Smoother smoother : smoothers) {
                s_cnt += 1;
                smoother.tick += 1;
                double[] smoother_feature = get_feature(smoother.values);
                int min_idx = 0;
                double min_diff = 10000000.0;
                for (int i=0;i<data.size();i++) {
                    if (data_paired[i])
                        continue;
                    double diff = calc_diff(smoother_feature, data_feature[i]);
                    Log.d("SMOOTHING_DEBUG", String.valueOf(diff));
                    if (diff < min_diff) {
                        min_diff = diff;
                        min_idx = i;
                    }
                }
                Log.d("SMOOTHING_DEBUG", min_diff + " " + smoothers.size());
                if (min_diff <= DIFF_THR) {
                    data.setElementAt(smoother.record_and_smooth(data.elementAt(min_idx), t), min_idx);
                    smoother.tick = 0;
                    data_paired[min_idx] = true;
                    Log.d("SMOOTHING_MATCH", s_cnt+" is matched to "+min_idx+" with a diff of "+min_diff);
                }
            }
            for (int i=0;i<data.size();i++) {
                if (!data_paired[i]) {
                    addOne();
                    Smoother smoother = smoothers.lastElement();
                    smoother.tick = 0;
                    data.setElementAt(smoother.record_and_smooth(data.elementAt(i), t), i);
                }
            }
        }
        Iterator<Smoother> it = smoothers.iterator();
        while (it.hasNext()) {
            Smoother smoother = it.next();
            if (smoother.tick >= MAX_TICK) {
                it.remove();
            }
        }
        return data;
    }
}
