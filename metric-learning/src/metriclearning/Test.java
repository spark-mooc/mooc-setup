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
public class Test {
    
    
    public static void main(String[] args) {
/*
        Levenshtein l = new Levenshtein();
        String a = "Metric learning for Link Discovery.";
        String b = "Metriclearning 4 LD";
        float sim = l.getUnNormalisedSimilarity(a, b);
        float nSim = l.getSimilarity(a, b);
        System.out.println(a+","+b+" -> "+sim+" ("+nSim+")");
        System.out.println("alph(a) = "+l.getSourceAlphabet());
        System.out.println("alph(b) = "+l.getTargetAlphabet());
 */
//        updateClassifier();
//        System.out.println(testMe(-20,-28));
//        System.out.println(Double.parseDouble("3.99"));
//        double d = 0.3;
//        Notifier.notify(d, d, d, d, d, d, d, 100);
    }
    
    @SuppressWarnings("unused")
    private static double testMe(double sourceNumericValue, double targetNumericValue) {
                double srcVal, tgtVal;
                if(Math.min(sourceNumericValue, targetNumericValue) < 0) {
                    if(Math.max(sourceNumericValue, targetNumericValue) < 0) {
                        srcVal = -sourceNumericValue;
                        tgtVal = -targetNumericValue;
                    } else {
                        srcVal = sourceNumericValue - Math.min(sourceNumericValue, targetNumericValue);
                        tgtVal = targetNumericValue - Math.min(sourceNumericValue, targetNumericValue);
                    }
                } else {
                    srcVal = sourceNumericValue;
                    tgtVal = targetNumericValue;
                }
                double d;
                if(srcVal == 0.0 && tgtVal == 0.0)
                    d = 1.0;
                else
                    d = 1 - Math.abs(srcVal - tgtVal) / Math.max(srcVal, tgtVal);
                return d;
    }

    @SuppressWarnings("unused")
    private static void updateClassifier() {
        svm_problem problem = new svm_problem();
        final int L = 100;
        problem.l = L;
        svm_node[][] x = new svm_node[L][2];
        double[] y = new double[L];
        for(int i=0; i<x.length; i++) {
            if(i<x.length/2) {
                for(int j=0; j<x[i].length; j++) {
                    x[i][j] = new svm_node();
                    x[i][j].index = j;
                    x[i][j].value = j+Math.random();
//                    s(x[i][j].value+", ", false);
                }
                y[i] = 1;
//                s(y[i]+"", true);
            } else {
                for(int j=0; j<x[i].length; j++) {
                    x[i][j] = new svm_node();
                    x[i][j].index = j;
                    x[i][j].value = Math.random();
//                    s(x[i][j].value+", ", false);
                }
                y[i] = -1;
//                s(y[i]+"", true);
            }
        }
        problem.x = x;
        problem.y = y;
        // svm_type=0,kernel_type=?,gamma=1,cache_size=40,eps=0.001,C=1,nr_weight=0,shrinking=1
        svm_parameter parameter = new svm_parameter();
        parameter.C = 0.5;
        parameter.kernel_type = svm_parameter.LINEAR;
        parameter.eps = 0.001;
        svm_model model = svm.svm_train(problem, parameter);
        svm_node[][] sv = model.SV;
        double[][] sv_coef = model.sv_coef;
        s("hold off;",true);
        for(int j=0; j<x[0].length; j++) {
            s("xp"+(j+1)+" = [",false);
            for(int i=0; i<x.length/2; i++)
                s(x[i][j].value+" ",false);
            s("];",true);
        }
        s("plot(xp1,xp2,'x'); hold on;",true);
        for(int j=0; j<x[0].length; j++) {
            s("xn"+(j+1)+" = [",false);
            for(int i=x.length/2; i<x.length; i++)
                s(x[i][j].value+" ",false);
            s("];",true);
        }
        s("plot(xn1,xn2,'xr');",true);
        s("xl = [0:0.01:2];",true);
        
        // calculating:
        // w = sv' * sv_coef'
        // b = -rho
        double[][] w = new double[sv[0].length][sv_coef.length];
        double[][] sv_tr_conv = transposeAndConvert(sv);
        w = LinearAlgebra.times(sv_tr_conv, LinearAlgebra.transpose(sv_coef));
        double b = -(model.rho[0]);
        int signum = (sv_coef[0][0] < 0) ? 1 : -1;
        double m = signum * w[0][0]/w[1][0];
        double q = -b / w[1][0];
//        for(int i=0; i<w.length; i++)
//            s("w["+i+"] = "+w[i][0],true);
//        s("b = "+q,true);
      
        
        s("yl = xl * "+m+" + "+q+ ";",true);
        s("plot(xl,yl,'k');",true);
        s("axis([0 2 0 2]);", true);
    }

    private static void s(String out, boolean newline) {
        System.out.print(out+(newline ? "\n" : ""));
    }

    private static double[][] transposeAndConvert(svm_node[][] sv) {
        double[][] t = new double[sv[0].length][sv.length];
        for(int i=0; i<sv.length; i++)
            for(int j=0; j<sv[i].length; j++)
                t[j][i] = sv[i][j].value;
        return t;
    }
}
