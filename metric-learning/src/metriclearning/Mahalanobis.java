/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import org.math.array.LinearAlgebra;

/**
 *
 * @author tom
 */
public class Mahalanobis {
    
    // the covariance matrix
    private double[][] S;
    
    public static void main(String[] args) {
        
        double[] x = {2000};
        double[] y = {1999};
        
        Mahalanobis mah = new Mahalanobis(x.length);
        System.out.println(mah.getSimilarity(x, y));
        
    }
    
    public Mahalanobis(int dim) {
        S = new double[dim][dim];
        for(int i=0; i<dim; i++)
            for(int j=0; j<dim; j++)
                if(i == j)
                    S[i][j] = 1.0;
                else
                    S[i][j] = 0.0;
    }
    
    public double getDistance(double[] x, double[] y) {
        double[][] diff = new double[1][x.length];
        for(int i=0; i<x.length; i++)
            diff[0][i] = x[i] - y[i];
        double result[][] = LinearAlgebra.times( diff, LinearAlgebra.inverse(S) );
        result = LinearAlgebra.times( result, LinearAlgebra.transpose(diff) );
        return Math.sqrt( result[0][0] );
    }
    
    public double getSimilarity(double[] x, double[] y) {
        return 1.0 / (1.0 + getDistance(x, y));
    }
     
}
