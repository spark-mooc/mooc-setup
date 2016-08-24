/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import java.util.ArrayList;

/**
 *
 * @author tom
 */
public class Couple1 implements Comparable<Couple1> {
    
    private Resource1 source;
    private Resource1 target;
    
    private ArrayList<Double> similarities;
    private ArrayList<Double> distances;
    
    private double gamma;
    
//    private int[][][] count;
    private ArrayList<Operation1> ops;

    public static final int TP = 1;
    public static final int FP = 2;
    public static final int TN = 3;
    public static final int FN = 4;
    private int classification;

    public int getClassification() {
        return classification;
    }

    public void setClassification(int classification) {
        this.classification = classification;
    }
    
    public void resetCount() {
//        for(int i=0; i<count.length; i++)
//            for(int j=0; j<count[i].length; j++)
//                for(int k=0; k<count[i][j].length; k++)
//                    count[i][j][k] = 0;
    	ops.clear();
    }
    
    public void count(int i, int j, int k) {
//        count[i][j][k] ++;
    	ops.add(new Operation1(i, j, k));
    }
    
    public double getGamma() {
        return gamma;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

    public double getSimMean() {
        double sum = 0.0;
        for(Double sim : similarities)
            sum += sim;
        return sum/similarities.size();
    }

    public Resource1 getSource() {
        return source;
    }

    public Resource1 getTarget() {
        return target;
    }

    public ArrayList<Double> getSimilarities() {
        return similarities;
    }

	public ArrayList<Double> getDistances() {
		return distances;
	}
    
    public void addSimilarity(double s) {
        similarities.add(s);
        double d = s==0 ? Double.POSITIVE_INFINITY : (1.0-s)/s;
        distances.add(d);
    }

    public void addDistance(double d) {
    	double s = d / (1.0 + d);
        similarities.add(s);
        distances.add(d);
    }

    public Couple1(Resource1 source, Resource1 target) {
        this.source = source;
        this.target = target;
        similarities = new ArrayList<Double>();
        distances = new ArrayList<Double>();
        ops = new ArrayList<Operation1>();
    }

    public void clearSimilarities() {
        similarities.clear();
        distances.clear();
    }
    
    public int[] getCountMatrixAsArray(int k) {
        int[] cArr = new int[4096];
        for(Operation1 op : ops) {
        	int n = op.getN();
        	if(k == n) {
	        	int arg1 = op.getArg1();
	        	int arg2 = op.getArg2();
	        	cArr[arg1*64+arg2]++;
        	}
        }        
//        int h = 0;
//        for(int i=0; i<count.length; i++)
//            for(int j=0; j<count[i].length; j++)
//                if(i != j) {
//                    cArr[h] = count[i][j][k];
//                    h++;
//                }
        return cArr;
    }

    @Override
    public int compareTo(Couple1 c) {
    	// maybe we should replace '#' with another symbol...
        String c1 = this.getSource().getID()+"#"+this.getTarget().getID();
        String c2 = c.getSource().getID()+"#"+c.getTarget().getID();
        return c1.compareTo(c2);
    }

}
