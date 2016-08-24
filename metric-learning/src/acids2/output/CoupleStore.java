package acids2.output;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import acids2.Couple;
import acids2.Resource;

public class CoupleStore {
	
	public static void saveFilteredCouples(ArrayList<Couple> couples, String filePrefix) throws IOException {
		FileWriter fstream0 = new FileWriter("output/"+filePrefix+"_couples.txt");
		BufferedWriter out0 = new BufferedWriter(fstream0);
		
		for(Couple c : couples) {
			for(int i=0; i<c.getDistances().size(); i++)
				out0.append(c.getDistanceAt(i)+",");
			if(c.isPositive()) {
				out0.append("1\n");
			} else {
				out0.append("0\n");
			}
		}
		
		out0.close();
	}
	
	public static ArrayList<Couple> loadFilteredCouples(String filePrefix) throws IOException {
		ArrayList<Couple> couples = new ArrayList<Couple>();
		FileReader fstream0 = new FileReader("output/"+filePrefix+"_couples.txt");
		BufferedReader in = new BufferedReader(fstream0);
		
		String currentLine;
		while((currentLine = in.readLine()) != null) {
			String[] line = currentLine.split(",");
			Couple c = new Couple(new Resource(""+Math.random()), new Resource(""+Math.random()));
			for(int i=0; i<line.length-1; i++)
				c.setDistance(Double.parseDouble(line[i]), i);
			if(line[line.length-1].equals("1"))
				c.setPositive(true);
			else
				c.setPositive(false);
			couples.add(c);
		}
		
		in.close();
		return couples;
	}
	
	public static ArrayList<Couple> filterCouplesFromCartesian(String filePrefix, double threshold) throws IOException {
		ArrayList<Couple> couples = new ArrayList<Couple>();
		FileReader fstream0 = new FileReader("output/"+filePrefix+"_testset.txt");
		BufferedReader in = new BufferedReader(fstream0);

		int cnt = 0;
		System.out.print("Filtering couples");
		String currentLine;
		nextLine: while((currentLine = in.readLine()) != null) {
			String[] line = currentLine.split(",");
			if(++cnt % 100000 == 0)
				System.out.print(".");
			for(int i=0; i<line.length-1; i++) {
				double d = Double.parseDouble(line[i]);
				if(d < threshold)
					continue nextLine;
			}
			Couple c = new Couple(new Resource(""+Math.random()), new Resource(""+Math.random()));
			for(int i=0; i<line.length-1; i++)
				c.setDistance(Double.parseDouble(line[i]), i);
			if(line[line.length-1].equals("1"))
				c.setPositive(true);
			else
				c.setPositive(false);
			couples.add(c);
		}
		System.out.println();
		
		in.close();
		return couples;
	}

}
