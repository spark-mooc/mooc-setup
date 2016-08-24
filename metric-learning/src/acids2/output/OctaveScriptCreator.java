package acids2.output;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import acids2.Property;
import acids2.Resource;
import acids2.test.Test;
import filters.StandardFilter;

public class OctaveScriptCreator implements ScriptCreator {

	@Override
	public void create(ArrayList<Resource> sources, ArrayList<Resource> targets, 
			ArrayList<Property> props, double[] w_linear, double theta) throws IOException {
		
		int n = props.size();
		
		String[] xp = new String[n];
		String[] xn = new String[n];
		for(int i=0; i<n; i++) {
			xp[i] = "x"+i+"p = [";
			xn[i] = "x"+i+"n = [";
		}

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
					for(int i=0; i<n; i++)
						xp[i] += d[i] + " ";
				} else {
					for(int i=0; i<n; i++)
						xn[i] += d[i] + " ";
				}
			}
		}
		
		String points = "";
		for(int i=0; i<n; i++) {
			xp[i] += "];\n";
			xn[i] += "];\n";
			points += xp[i] + xn[i];
		}
		
		points += "w = [";
		for(int i=0; i<w_linear.length; i++) {
			points += w_linear[i] + " ";
		}
		points += "];\ntheta = "+theta+";\n";
		
		for(int i=0; i<n; i++) {
			points += "label"+(i+1)+" = \""+props.get(i).getName()+"\";\n";
		}

		// Create file 
		FileWriter fstream = new FileWriter("octave/output.m");//"octave/points"+System.currentTimeMillis()+".m");
		BufferedWriter out = new BufferedWriter(fstream);
		out.write(points);
		//Close the output stream
		out.close();
		System.out.println("Script done.");

	}

}
