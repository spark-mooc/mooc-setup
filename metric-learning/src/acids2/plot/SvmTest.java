package acids2.plot;

import java.io.IOException;
import java.util.ArrayList;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.math.array.LinearAlgebra;

import acids2.Couple;
import acids2.Resource;

public class SvmTest {

	private static int KERNEL = svm_parameter.POLY;
	private static int DEGREE = 2;
	private static int COEF0 = 0;

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		
		int n = 3;
		double[] w = new double[n];
		double theta;
		
		double[] cx = {1.0, 1.0, 3.0, 2.0};
		double[] cy = {1.0, 3.0, 1.0, 2.0};
		double[] cz = {1.0, 3.0, 1.0, 2.0};
		
		ArrayList<Couple> poslbl = new ArrayList<Couple>();
		ArrayList<Couple> neglbl = new ArrayList<Couple>();
		for(int i=0; i<cx.length; i++) {
			Couple c = new Couple(new Resource("s"+i), new Resource("t"+i));
			c.setDistance(cx[i], i);
			c.setDistance(cy[i], i);
			c.setDistance(cz[i], i);
			if(i<cx.length-1)
				poslbl.add(c);
			else
				neglbl.add(c);
		}
		
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
		svm_model model;
        svm_problem problem = new svm_problem();
        problem.l = size;
        problem.x = x;
        problem.y = y;
        svm_parameter parameter = new svm_parameter();
        parameter.C = 1E+6;
        parameter.svm_type = svm_parameter.C_SVC;
        parameter.kernel_type = KERNEL;
        if(KERNEL == svm_parameter.POLY) {
			parameter.degree = DEGREE; // default: 3
			parameter.coef0  = COEF0; // default: 0
			parameter.gamma  = 1; // default: 1/n
        } 
        parameter.eps = 0.001;
        model = svm.svm_train(problem, parameter);
        // sv = ( nSV ; n )
        svm_node[][] sv = model.SV;
        // sv_coef = ( 1 ; nSV )
        double[][] sv_coef = model.sv_coef;
        
        // vec = sv' * sv_coef' = (sv_coef * sv)' = ( n ; 1 )
        double[][] vec = new double[sv[0].length][sv_coef.length];
        // converting sv to double -> sv_d
        double[][] sv_d = new double[sv.length][sv[0].length];
        for(int i=0; i<sv.length; i++)
            for(int j=0; j<sv[i].length; j++)
                sv_d[i][j] = sv[i][j].value;
        vec = LinearAlgebra.transpose(LinearAlgebra.times(sv_coef, sv_d));
        
        int signum = (model.label[0] == -1.0) ? 1 : -1;
        for(int i=0; i<n; i++) {
            w[i] = signum * vec[i][0];
            System.out.println("w_"+i+" = "+w[i]);
        }
        theta = signum * model.rho[0];
        
        theta = model.rho[0];
        System.out.println("theta = "+theta);
        
		
    	System.out.println("pos:");
        for(Couple c : poslbl)
        	System.out.println(classify(c, model)+"\t"+classify(c, theta, sv_coef, sv_d));
    	System.out.println("neg:");
        for(Couple c : neglbl)
        	System.out.println(classify(c, model)+"\t"+classify(c, theta, sv_coef, sv_d));
        
        Svm3D.draw(model, problem, theta, sv_d, theta);
        
	}

	private static boolean classify(Couple c, svm_model model) {
		int size = c.getDistances().size();
        svm_node[] node = new svm_node[size];
        for(int i=0; i<size; i++) {
        	node[i] = new svm_node();
        	node[i].index = i;
        	node[i].value = c.getDistances().get(i);
        }
        if(svm.svm_predict(model, node) == 1.0)
        	return true;
        else return false;
	}
	
	private static boolean classify(Couple c, double theta, double[][] sv_coef,
			double[][] sv_d) {
		
		double sum = 0.0;
		for(int i=0; i<sv_coef[0].length; i++) {
			double scal = 0.0;
			for(int j=0; j<c.getDistances().size(); j++)
				scal += c.getDistances().get(j) * sv_d[i][j];
			sum += sv_coef[0][i] * Math.pow(scal, 2);
		}
		System.out.println(sum-theta);
		return sum >= theta;
	}
	

}
