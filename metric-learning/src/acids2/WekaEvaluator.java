package acids2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import acids2.output.Fscore;

public class WekaEvaluator {

	public WekaEvaluator(String fileName) throws FileNotFoundException {
		
		for(double value=0; value<=0.11; value+=0.01) {
			Scanner in = new Scanner(new File(fileName));
			double tp = 0, tn = 0, fp = 0, fn = 0;
			while(in.hasNextLine()) {
				String[] line = removeSpaces(in.nextLine()).split(" ");
				int actual = Integer.parseInt(line[1]);
				double pred = Double.parseDouble(line[2]);
				int pred_class = pred > value ? 1 : 0;
				if(actual == 1) {
					if(actual == pred_class)
						tp++;
					else
						fn++;
				} else {
					if(actual == pred_class)
						tn++;
					else
						fp++;
				}
			}
			in.close();
			Fscore f = new Fscore("", tp, fp, tn, fn);
			System.out.println(value+"\t"+f.getF1()+"\t"+f.getPre()+"\t"+f.getRec());
		}
		
	}
	
	private String removeSpaces(String in) {
		String out = in.trim();
		do {
			in = out;
			out = in.replaceAll("  ", " ");
		} while(!out.equals(in));
		return out;
	}

	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException {
//		new WekaEvaluator("output/linear_regression.txt");
		new WekaEvaluator("output/multilayer_perceptron.txt");
	}

}
