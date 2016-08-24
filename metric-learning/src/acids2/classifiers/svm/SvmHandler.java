package acids2.classifiers.svm;

import java.util.ArrayList;

import org.math.array.LinearAlgebra;

import utility.SvmUtils;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import acids2.Couple;
import acids2.output.Fscore;

public class SvmHandler {
	
	// SVM parameters
	private svm_model model;
	private svm_problem problem;
	private int kernel;
	private final int DEGREE = 2;
	private final int COEF0 = 0;
	private final double GAMMA = 1;
	private final double C_min = 1E+2;
//	private final double C_max = 1E+2;
	private double C_opt;
	private final double EPS = 1E-3;
	
	// classifier properties
	private double[][] sv_d;
	
	private double[] wLinear;
	
	public void setWLinear(double[] wLinear) {
		this.wLinear = wLinear;
	}

	public double[] getWLinear() {
		return wLinear;
	}

	private double theta;
	private int n;
	
	public SvmHandler(int kernel) {
		this.kernel = kernel;
	}

	public void setN(int n) {
		this.n = n;
	}

	public boolean isTraced() {
		return model != null;
	}
    
	public boolean trace(ArrayList<Couple> poslbl, ArrayList<Couple> neglbl) {
    	
        int size = poslbl.size() + neglbl.size();
        
        // build x,y vectors
        svm_node[][] x = new svm_node[size][n];
        double[] y = new double[size];
        for(int i=0; i<poslbl.size(); i++) {
            ArrayList<Double> arr = poslbl.get(i).getDistances();
            for(int j=0; j<arr.size(); j++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = arr.get(j);
            }
            y[i] = 1;
        }
        for(int i=poslbl.size(); i<size; i++) {
            ArrayList<Double> arr = neglbl.get(i-poslbl.size()).getDistances();
            for(int j=0; j<arr.size(); j++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = arr.get(j);
            }
            y[i] = -1;
        }
        
        // configure model
        // POLY: (gamma*u'*v + coef0)^degree
        problem = new svm_problem();
        problem.l = size;
        problem.x = x;
        problem.y = y;
        svm_parameter parameter = new svm_parameter();
        parameter.svm_type = svm_parameter.C_SVC;
        parameter.kernel_type = kernel;
        if(kernel == svm_parameter.POLY) {
			parameter.degree = DEGREE; // default: 3
			parameter.coef0  = COEF0; // default: 0
			parameter.gamma  = GAMMA; // default: 1/n
        } 
        parameter.eps = EPS;
        
        // parameter optimization
        ArrayList<Couple> couples = new ArrayList<Couple>();
        couples.addAll(poslbl);
        couples.addAll(neglbl);
        double f_max = 0.0;
        C_opt = C_min;
//        for(double C=C_min; C<=C_max; C*=10) {
//            parameter.C = C;
//            model = svm.svm_train(problem, parameter);
//        	Fscore f = evaluateOn(couples);
//        	double f1 = f.getF1();
//        	if(f1 > f_max) {
//        		f_max = f1;
//        		C_opt = C;
//        	}
//        	if(f1 == 1.0)
//        		break; // Ockham's razor.
//        }
        
        // assign best parameter
        System.out.println("Optimization finished: f_max = "+f_max+", C_opt = "+C_opt);
        parameter.C = C_opt;
        model = svm.svm_train(problem, parameter);
        
        // sv = ( nSV ; n )
        svm_node[][] sv = model.SV;
        // no support vectors
        if(sv.length == 0) {
        	System.err.println("No SVMs found.");
        	return false;
        }
 
        // sv_coef = ( 1 ; nSV )
        double[][] sv_coef = model.sv_coef;
        
        sv_d = new double[sv.length][n];
        for(int j=0; j<sv.length; j++)
            for(int i=0; i<sv[j].length; i++)
            	sv_d[j][i] = sv[j][i].value;
        
        int signum = (model.label[0] == -1.0) ? -1 : 1;
        theta = signum * model.rho[0];
        
        switch(kernel) {
        case svm_parameter.LINEAR:
	        double[][] w = new double[sv[0].length][sv_coef.length];
	        signum = (model.label[0] == -1.0) ? 1 : -1;
	        
	        w = LinearAlgebra.transpose(LinearAlgebra.times(sv_coef, SvmUtils.nodeToDouble(sv)));
	        
	        wLinear = new double[w.length];

	        for(int i=0; i<wLinear.length; i++)
	        	wLinear[i] = signum * w[i][0];
	        
	        break;
	        
        case svm_parameter.POLY:
	        // w = sv' * sv_coef' = (sv_coef * sv)' = ( n ; 1 )
	        double[][] phis = new double[sv.length][];
	        for(int i=0; i<phis.length; i++)
	        	phis[i] = phi(sv_d[i]);
	        
	        wLinear = new double[phis[0].length];
	        
	        for(int i=0; i<phis.length; i++)
	        	for(int j=0; j<phis[i].length; j++)
	        		wLinear[j] += sv_coef[0][i] * phis[i][j];
	        
	        break;
	    default:
	    	System.err.println("Kernel not supported: "+kernel);
        }
        
        // theta is normally at the first member in the classification inequality
//        theta_plot = -theta;

    	for(int i=0; i<wLinear.length; i++) {
    		wLinear[i] = signum * wLinear[i] / Math.abs(theta);
    		System.out.println("w_linear["+i+"] = "+wLinear[i]);
    	}
		theta = theta / Math.abs(theta);

        System.out.println("theta = "+theta);
        
        return true;
    }
	
	public Fscore evaluateOn(ArrayList<Couple> couples) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		for(Couple c : couples) {
			if(c.isPositive()) {
				if(this.classify(c))
					tp++;
				else
					fn++;
			} else {
				if(this.classify(c)) {
					fp++;
				} else
					tn++;
			}
		}
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}
    
    public double[] phi(double[] x) {
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

	public double getTheta() {
		return theta;
	}

	public void setTheta(double theta) {
		this.theta = theta;
	}

	public int getN() {
		return n;
	}

	public void initW() {
		wLinear = new double[n];
		for(int i=0; i<n; i++)
			wLinear[i] = 1.0;
	}
	
	public boolean classify(Couple c) {
		if(model == null) {
			// Default classifier is always set to linear.
			double sum = 0.0;
			ArrayList<Double> dist = c.getDistances();
			for(int i=0; i<dist.size(); i++)
				sum += dist.get(i) * wLinear[i];
			return sum >= theta; 
		}
        svm_node[] node = new svm_node[n];
        for(int i=0; i<n; i++) {
        	node[i] = new svm_node();
        	node[i].index = i;
        	node[i].value = c.getDistanceAt(i);
        }
        return classify(node);
	}

	public boolean classify(double[] val) {
        svm_node[] node = new svm_node[n];
        for(int i=0; i<n; i++) {
        	node[i] = new svm_node();
        	node[i].index = i;
        	node[i].value = val[i];
        }
        return classify(node);
	}

	private boolean classify(svm_node[] node) {
		// TODO handle NullPointerException
        if(svm.svm_predict(model, node) == 1.0)
        	return true;
        else return false;
	}
	
	public void initTheta(ArrayList<Couple> couples, int mapSize) {
		theta = n - 0.5;
		int pc = 0, direction = 0;
		double delta = 1;
		while(pc != mapSize) {
			pc = 0;
			for(Couple c : couples)
				if(this.classify(c))
					pc++;
			if(pc < mapSize) {
				theta -= delta;
				if(direction == 1)
					delta /= 10;
				direction = -1;
			}
			if(pc > mapSize) {
				theta += delta;
				if(direction == -1)
					delta /= 10;
				direction = 1;
			}
			if(delta < 1E-8)
				break;
			System.out.println("theta = "+theta+"\tdelta = "+delta+"\tpc = "+pc+"\taimpc = "+mapSize);
		}
	}
	
	public double computeGamma(Couple c, double point) {
		double sum = 0.0;
		ArrayList<Double> dist = c.getDistances();
		for(Double d : dist)
			sum += Math.pow(d - point, 2.0);
		return Math.sqrt(sum);
	}

    public double computeGamma(Couple c) {
		ArrayList<Double> dist = c.getDistances();
		if(model == null || kernel == svm_parameter.LINEAR) {
			// Default classifier...
	        double numer = 0.0, denom = 0.0;
	        for(int i=0; i<dist.size(); i++) {
	            numer += dist.get(i) * wLinear[i];
	            denom += Math.pow(wLinear[i], 2);
	        }
	        numer -= theta;
	        return Math.abs(numer/Math.sqrt(denom));
		}
		// All classifiers...
		double[] x = new double[dist.size()];
		for(int i=0; i<dist.size(); i++)
			x[i] = dist.get(i);
		double[] phi = phi(x);
		int m = phi.length;
		// calculate Q
		double[] q = new double[m];
		for(int i=0; i<m; i++)
			if(wLinear[i] != 0) {
				q[i] = theta / wLinear[i];
				break;
			}
		// calculate unit vector wu
		double norm = 0;
		for(int i=0; i<m; i++)
			norm += Math.pow(wLinear[i], 2);
		norm = Math.sqrt(norm);
		double[] wu = new double[n];
		for(int i=0; i<n; i++)
			wu[i] = wLinear[i] / norm;
		// calculate t = phi(X - Q) . wu
		double[] phixq = new double[m];
		for(int i=0; i<m; i++)
			phixq[i] = phi[i] - q[i];
		double t = 0;
		for(int i=0; i<n; i++)
			t += phixq[i] * wu[i];
		// calculate P'
		double[] p1 = new double[n];
		for(int i=0; i<n; i++)
			p1[i] = phi[i] - t * wu[i];
		// calculate segment XP
		double sum = 0;
		for(int i=0; i<n; i++)
			if(p1[i] >= 0)
				sum += Math.pow(dist.get(i) - phiInverse(p1[i]), 2);
			else // anti-transformation is out of the similarity space
				sum += Math.pow(dist.get(i), 2);
		sum = Math.sqrt(sum);
		if(Double.isNaN(sum))
			System.out.println();
		return sum;
    }

    private double phiInverse(double x) {
		return Math.sqrt(x);
	}


}
