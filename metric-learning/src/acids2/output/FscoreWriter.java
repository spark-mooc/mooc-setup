package acids2.output;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class FscoreWriter {

	public static void write(String filename, ArrayList<Fscore> fsl) throws IOException {
		// Create file
		FileWriter fstream0 = new FileWriter("output/fscores_"+filename+".txt");
		BufferedWriter out0 = new BufferedWriter(fstream0);
		double avg = 0.0;
		for(Fscore fs : fsl) {
			out0.append(fs.toString()+"\n");
			avg += fs.getF1();
		}
		avg = avg / (double) fsl.size();
		out0.append("\n"+avg+"\n");
		out0.close();
		
		System.out.println("\nAVERAGE: "+avg);
	}

}
