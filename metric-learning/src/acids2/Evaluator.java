package acids2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import acids2.output.Fscore;

public class Evaluator {
	
	private double[] wLinear;
	private double theta;
	
	public Evaluator(String testsetFile, String classifierFile) throws FileNotFoundException {

		Scanner in1 = new Scanner(new File(classifierFile));
		if(in1.hasNextLine()) {
			String[] line = in1.nextLine().split(",");
			wLinear = new double[line.length-1];
			for(int i=0; i<wLinear.length; i++)
				wLinear[i] = Double.parseDouble(line[i]);
			theta = Double.parseDouble(line[line.length-1]);
		} else {
			System.err.println("No classifier found in "+classifierFile+".");
			in1.close();
			return;
		}
		in1.close();

		Scanner in2 = new Scanner(new File(testsetFile));
		double tp = 0, tn = 0, fp = 0, fn = 0;
		while(in2.hasNextLine()) {
			String[] line = in2.nextLine().split(",");
			double[] val = new double[line.length-1];
			for(int i=0; i<val.length; i++)
				val[i] = Double.parseDouble(line[i]);
			if(line[line.length-1].equals("1")) {
				if(classify(val))
					tp++;
				else
					fn++;
			} else {
				if(classify(val))
					fp++;
				else
					tn++;
			}
			
		}
		in2.close();

		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();

	}
	
	public static void main(String[] args) throws FileNotFoundException {
		new Evaluator("output/testset.txt", "output/classifier.txt");
	}

	private boolean classify(double[] val) {
		double[] phival = phi(val);
		double sum = 0.0;
		for(int i=0; i<phival.length; i++) {
			sum += phival[i] * wLinear[i];
		}
		return sum >= theta;
	}
	
    private double[] phi(double[] x) {
    	int n = x.length;
    	// assuming DEGREE = 2 and COEF0 = 0
    	int n1 = (int) (Math.pow(n, 2) + n) / 2;
    	double[] phi = new double[n1];
    	for(int i=0; i<n; i++)
    		phi[i] = Math.pow(x[i], 2);
    	double r2 = Math.sqrt(2);
    	int p = n;
    	for(int i=0; i<n-1; i++)
    		for(int j=i+1; j<n; j++) {
        		phi[p] = r2 * x[i] * x[j];
        		p++;
    		}
    	return phi;
    }

	
}
