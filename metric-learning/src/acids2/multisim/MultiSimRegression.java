package acids2.multisim;

import java.util.ArrayList;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.math.array.DoubleArray;
import org.math.array.LinearAlgebra;

import utility.SvmUtils;

import acids2.Couple;

public class MultiSimRegression {

	// SVM parameters
	private final double C = 1E+2;
	private int KERNEL = svm_parameter.LINEAR;
	
	private svm_model model;
	private int n;

	private svm_problem problem;
	
	private double[][] sv_d;

	// regression properties
	private double theta;
	private double[] w;
	private double[] wLinear;

	public MultiSimRegression() {
	}

	public double[] getW() {
		return w;
	}
	public double[] getWLinear() {
		return wLinear;
	}

	public boolean isTraced() {
		return model != null;
	}

	public void setN(int n) {
		this.n = n;
	}

	public void setW(double[] w) {
		this.w = w;
	}

	public boolean trace(ArrayList<Couple> couples) {

		int size = couples.size();

		// build x vector
		svm_node[][] x = new svm_node[size][n];
		for (int i = 0; i < couples.size(); i++) {
			ArrayList<Double> arr = couples.get(i).getDistances();
			for (int j = 0; j < arr.size(); j++) {
				x[i][j] = new svm_node();
				x[i][j].index = j;
				x[i][j].value = arr.get(j);
			}
		}

		// configure model
		problem = new svm_problem();
		problem.l = size;
		problem.x = x;
		svm_parameter parameter = new svm_parameter();
		parameter.svm_type = svm_parameter.NU_SVR;
		parameter.kernel_type = KERNEL;

		parameter.C = C;
		model = svm.svm_train(problem, parameter);

		// sv = ( nSV ; n )
		svm_node[][] sv = model.SV;
		// no support vectors
		if (sv.length == 0) {
			System.err.println("No SVMs found.");
			return false;
		}

		// sv_coef = ( 1 ; nSV )
		double[][] sv_coef = model.sv_coef;

		sv_d = new double[sv.length][n];
		for (int j = 0; j < sv.length; j++)
			for (int i = 0; i < sv[j].length; i++)
				sv_d[j][i] = sv[j][i].value;

		int signum = (model.label[0] == -1.0) ? -1 : 1;
		theta = signum * model.rho[0];

		double[][] w = new double[sv[0].length][sv_coef.length];
		signum = (model.label[0] == -1.0) ? 1 : -1;

		w = DoubleArray.transpose(LinearAlgebra.times(sv_coef,
				SvmUtils.nodeToDouble(sv)));

		wLinear = new double[w.length];

		for (int i = 0; i < wLinear.length; i++)
			wLinear[i] = signum * w[i][0];

		// theta is normally at the first member in the classification
		// inequality
		// theta_plot = -theta;

		for (int i = 0; i < wLinear.length; i++) {
			wLinear[i] = signum * wLinear[i] / Math.abs(theta);
			System.out.println("w_linear[" + i + "] = " + wLinear[i]);
		}
		theta = theta / Math.abs(theta);

		System.out.println("theta = " + theta);

		return true;
	}

}
