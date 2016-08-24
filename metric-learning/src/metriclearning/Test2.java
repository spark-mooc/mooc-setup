/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.math.array.LinearAlgebra;

/**
 *
 * @author tom
 */
public class Test2 {
    
    public static void main(String[] args) {
        svm_problem problem = new svm_problem();
        double[] pos = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        double[] neg = {1.0, 1.0, 0.9070463215834137, 0.9360493998830014, 1.0, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        problem.l = pos.length+neg.length;
        svm_node[][] x = new svm_node[pos.length+neg.length][2];
        double[] y = new double[pos.length+neg.length];
        for(int j=0; j<2; j++) {
            for(int i=0; i<pos.length; i++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = pos[i];
                y[i] = 1;
            }
            for(int i=pos.length; i<pos.length+neg.length; i++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = neg[i-pos.length];
                y[i] = -1;
            }
        }
        problem.x = x;
        problem.y = y;
        svm_parameter parameter = new svm_parameter();
        parameter.C = 1E+10;
        parameter.svm_type = svm_parameter.C_SVC;
        parameter.kernel_type = svm_parameter.LINEAR;
        parameter.eps = 0.0001;
        svm_model model = svm.svm_train(problem, parameter);
        // sv = ( nSV ; n )
        svm_node[][] sv = model.SV;
        // sv_coef = ( 1 ; nSV )
        double[][] sv_coef = model.sv_coef;
        
        // calculate w and b
        // w = sv' * sv_coef' = (sv_coef * sv)' = ( n ; 1 )
        // b = -rho
        double[][] w = new double[sv[0].length][sv_coef.length];
        int signum = (model.label[0] == -1.0) ? 1 : -1;
        
        w = LinearAlgebra.transpose(LinearAlgebra.times(sv_coef,MetricLearning.toDouble(sv)));
        double b = -model.rho[0];
        
        double[] C = new double[2];
        for(int i=0; i<C.length; i++) {
            C[i] = signum * w[i][0];
            MetricLearning.w("C["+i+"] = "+C[i]);
        }
        
        double bias = signum * b;
        MetricLearning.w("bias = "+bias);

//        C[0] = 0.5;
//        C[1] = 0.5;
//        bias = -1;

        double tp=0,fn=0,tn=0,fp=0;
        for(double p : pos) {
            if(classify(C, p, bias)) {
                MetricLearning.w(p +" -> P -> ok");
                tp++;
            } else {
                MetricLearning.w(p +" -> N -> ko");
                fn++;
            }
        }
        for(double p : neg) {
            if(!classify(C, p, bias)) {
                MetricLearning.w(p +" -> N -> ok");
                tn++;
            } else {
                MetricLearning.w(p +" -> P -> ko");
                fp++;
            }
        }
        double pre = tp+fp != 0 ? tp / (tp + fp) : 0;
        double rec = tp+fn != 0 ? tp / (tp + fn) : 0;
        double f1 = pre+rec != 0 ? 2 * pre * rec / (pre + rec) : 0;
        MetricLearning.w("pre = "+pre);
        MetricLearning.w("rec = "+rec);
        MetricLearning.w("f1 = "+f1+" (tp="+tp+", fp="+fp+", tn="+tn+", fn="+fn+")");


    }
    
    private static boolean classify(double[] C, double x, double b) {
        double sum = 0.0;
        for(int i=0; i<C.length; i++)
            sum += C[i] * x;
        return sum + b <= 0;
    }

}
