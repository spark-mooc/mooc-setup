package acids2.output;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import filters.StandardFilter;

import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import acids2.test.Test;

public class SageScriptCreator implements ScriptCreator {

	@Override
	public void create(ArrayList<Resource> sources, ArrayList<Resource> targets, ArrayList<Property> props,
			double[] w_linear, double theta) throws IOException  {
		
		int n = props.size();

		// Create file 
		FileWriter fstream0 = new FileWriter("sage/classifier.py");
		BufferedWriter out0 = new BufferedWriter(fstream0);
		FileWriter fstream1 = new FileWriter("sage/positives.py");
		BufferedWriter out1 = new BufferedWriter(fstream1);
		FileWriter fstream2 = new FileWriter("sage/negatives.py");
		BufferedWriter out2 = new BufferedWriter(fstream2);

		out1.append("poss = [");
		out2.append("negs = [");
		
		String[] names = new String[n];
		StandardFilter[] filters = new StandardFilter[n];
		for(int i=0; i<n; i++) {
			filters[i] = props.get(i).getFilter();
			names[i] = props.get(i).getName();
		}

		for(Resource s : sources) {
			for(Resource t : targets) {
				double[] d = new double[n];
				for(int i=0; i<n; i++) {
					double sim = filters[i].getDistance(s.getPropertyValue(names[i]), t.getPropertyValue(names[i]));
					d[i] = Double.isNaN(sim) ? 0 : sim;
				}
				if(Test.askOracle(s.getID()+"#"+t.getID())) {
					out1.append("(");
					for(int i=0; i<n; i++)
						out1.append(d[i]+",");
					out1.append("),");
				} else {
					out2.append("(");
					for(int i=0; i<n; i++)
						out2.append(d[i]+",");
					out2.append("),");
				}
			}
		}
		
		out1.append("]");
		out2.append("]");
		
		out0.append("w = [");
		for(int i=0; i<w_linear.length; i++) {
			out0.append(w_linear[i] + ",");
		}
		out0.append("]\ntheta = "+theta+"\nvar('x,y,z')\n"+
				"p1 = implicit_plot3d((w[0] * x**2 + w[1] * y**2 + w[2] * z**2 + w[3] * sqrt(2) * x*y) + " +
				"(w[4] * sqrt(2) * x*z + w[5] * sqrt(2) * y*z) == theta, (x, 0, 1), (y, 0, 1), (z, 0, 1))\n" +
				"p2 = point3d(negs,size=10,color='red')\n" +
				"p3 = point3d(poss,size=10,color='blue')\n" +
				"show(p1+p2+p3)");
		
//		for(int i=0; i<n; i++) {
//			points += "label"+(i+1)+" = \""+props.get(i).getName()+"\";\n";
//		}

		//Close the output stream
		out0.close();
		out1.close();
		out2.close();
		System.out.println("Script done.");
		
		
	}

	public void create(ArrayList<Couple> couples, ArrayList<Property> props,
			double[] w_linear, double theta) throws IOException  {
		
		int n = props.size();

		// Create file 
		FileWriter fstream0 = new FileWriter("sage/classifier.py");
		BufferedWriter out0 = new BufferedWriter(fstream0);
		FileWriter fstream1 = new FileWriter("sage/positives.py");
		BufferedWriter out1 = new BufferedWriter(fstream1);
		FileWriter fstream2 = new FileWriter("sage/negatives.py");
		BufferedWriter out2 = new BufferedWriter(fstream2);

		out1.append("poss = [");
		out2.append("negs = [");
		
		String[] names = new String[n];
		StandardFilter[] filters = new StandardFilter[n];
		for(int i=0; i<n; i++) {
			filters[i] = props.get(i).getFilter();
			names[i] = props.get(i).getName();
		}

		for(Couple c : couples) {
			double[] d = new double[n];
			for(int i=0; i<n; i++) {
				double sim = c.getDistanceAt(i);
				d[i] = Double.isNaN(sim) ? 0 : sim;
			}
			if(c.isPositive()) {
				out1.append("(");
				for(int i=0; i<n; i++)
					out1.append(d[i]+",");
				out1.append("),");
			} else {
				out2.append("(");
				for(int i=0; i<n; i++)
					out2.append(d[i]+",");
				out2.append("),");
			}
		}
		
		out1.append("]");
		out2.append("]");
		
		out0.append("w = [");
		for(int i=0; i<w_linear.length; i++) {
			out0.append(w_linear[i] + ",");
		}
		out0.append("]\ntheta = "+theta+"\nvar('x,y,z')\n"+
				"p1 = implicit_plot3d((w[0] * x**2 + w[1] * y**2 + w[2] * z**2 + w[3] * sqrt(2) * x*y) + " +
				"(w[4] * sqrt(2) * x*z + w[5] * sqrt(2) * y*z) == theta, (x, 0, 1), (y, 0, 1), (z, 0, 1))\n" +
				"p2 = point3d(negs,size=10,color='red')\n" +
				"p3 = point3d(poss,size=10,color='blue')\n" +
				"show(p1+p2+p3)");
		
//		for(int i=0; i<n; i++) {
//			points += "label"+(i+1)+" = \""+props.get(i).getName()+"\";\n";
//		}

		//Close the output stream
		out0.close();
		out1.close();
		out2.close();
		System.out.println("Script done.");
		
		
	}

}
