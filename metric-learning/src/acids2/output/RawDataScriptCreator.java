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

public class RawDataScriptCreator implements ScriptCreator {
	
	private String filePrefix;
	private ArrayList<Couple> trainingset;
	private boolean createTestsetFile, createTrainingsetFile, createClassifierFile;
		
	public RawDataScriptCreator(String filePrefix, boolean createTestsetFile, boolean createTrainingsetFile, boolean createClassifierFile) {
		this.filePrefix = filePrefix;
		this.createClassifierFile = createClassifierFile;
		this.createTestsetFile = createTestsetFile;
		this.createTrainingsetFile = createTrainingsetFile;
	}
	
	public ArrayList<Couple> getTrainingset() {
		return trainingset;
	}


	public void setTrainingset(ArrayList<Couple> trainingset) {
		this.trainingset = trainingset;
	}


	@Override
	public void create(ArrayList<Resource> sources,
			ArrayList<Resource> targets, ArrayList<Property> props,
			double[] w_linear, double theta) throws IOException {
		
		int n = props.size();
		
		String[] names = new String[n];
		StandardFilter[] filters = new StandardFilter[n];
		for(int i=0; i<n; i++) {
			filters[i] = props.get(i).getFilter();
			names[i] = props.get(i).getName();
		}
		
		if(createTestsetFile) {
			// Create file 
			FileWriter fstream0 = new FileWriter("output/"+filePrefix+"_testset.txt");
			BufferedWriter out0 = new BufferedWriter(fstream0);
			
			for(Resource s : sources) {
				for(Resource t : targets) {
					double[] d = new double[n];
					for(int i=0; i<n; i++) {
						double sim = filters[i].getDistance(s.getPropertyValue(names[i]), t.getPropertyValue(names[i]));
						d[i] = Double.isNaN(sim) ? 0 : sim;
					}
					for(int i=0; i<n; i++)
						out0.append(d[i]+",");
					if(Test.askOracle(s.getID()+"#"+t.getID())) {
						out0.append("1\n");
					} else {
						out0.append("0\n");
					}
				}
			}
			
			//Close the output stream
			out0.close();
		}

		if(createTrainingsetFile) {
			FileWriter fstream1 = new FileWriter("output/"+filePrefix+"_trainingset.txt");
			BufferedWriter out1 = new BufferedWriter(fstream1);
			
			for(Couple c : trainingset) {
				Resource s = c.getSource(), t = c.getTarget();
				double[] d = new double[n];
				for(int i=0; i<n; i++) {
					double sim = filters[i].getDistance(s.getPropertyValue(names[i]), t.getPropertyValue(names[i]));
					d[i] = Double.isNaN(sim) ? 0 : sim;
				}
				for(int i=0; i<n; i++)
					out1.append(d[i]+",");
				if(Test.askOracle(s.getID()+"#"+t.getID())) {
					out1.append("1\n");
				} else {
					out1.append("0\n");
				}
			}
			
			out1.close();
		}

		if(this.createClassifierFile) {
			FileWriter fstream2 = new FileWriter("output/"+filePrefix+"_classifier.txt");
			BufferedWriter out2 = new BufferedWriter(fstream2);
	
			String clax = "";
			for(double w : w_linear)
				clax += w + ",";
			clax += theta;
			out2.append(clax);
			out2.close();
		}
		
		System.out.println("Script done.");

	}

}
