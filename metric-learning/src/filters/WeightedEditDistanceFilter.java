package filters;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

import distances.WeightedEditDistanceExtended;

public abstract class WeightedEditDistanceFilter extends StandardFilter {

	public static final double INIT_FULL_WEIGHT = 1.0;
	public static final double INIT_CASE_WEIGHT = 0.5;

	protected HashMap<String, Double> weights = new HashMap<String, Double>();
	
	public WeightedEditDistanceFilter() {
		super();
		
		// load case weights
		for(char c='A'; c<='Z'; c++) {
			weights.put(c+","+(char)(c+32), INIT_CASE_WEIGHT);
			weights.put((char)(c+32)+","+c, INIT_CASE_WEIGHT);
		}
		
		// load confusion matrix
		Scanner in = null;
		try {
			in = new Scanner(new File("data/ConfusionMatrix.txt"));
		} catch (FileNotFoundException e) {
			System.err.println("Missing file `./data/ConfusionMatrix.txt`!");
			return;
		}
		for(char c1='a'; c1<='{'; c1++) {
			for(char c2='a'; c2<='{'; c2++) {
				double d = in.nextDouble();
				if(d != 1) {
					if(c1 == '{') {
						weights.put("ε,"+c2, d);
						weights.put("ε,"+(char)(c2-32), d);
					} else 	if(c2 == '{') {
						weights.put(c1+",ε", d);
						weights.put((char)(c1-32)+",ε", d);
					} else {
						weights.put(c1+","+c2, d);
						weights.put((char)(c1-32)+","+(char)(c2-32), d);
						// crossing weights (e.g., <A,b>, <a,B>)
						double dcross = (1.0 + d) / 2.0;
						weights.put((char)(c1-32)+","+c2, dcross);
						weights.put(c1+","+(char)(c2-32), dcross);
					}
				}
			}
		}
		in.close();
	}
	
	protected WeightedEditDistanceExtended wed = new WeightedEditDistanceExtended() {
		@Override
		public double transposeWeight(char cFirst, char cSecond) {
			return Double.POSITIVE_INFINITY;
		}
		@Override
		public double substituteWeight(char cDeleted, char cInserted) {
			Double d = weights.get(cDeleted+","+cInserted);
//			System.out.println(d+" detected for "+cDeleted+","+cInserted);
			if(d == null)
				return INIT_FULL_WEIGHT;
			else
				return d;
		}
		@Override
		public double matchWeight(char cMatched) {
			return 0.0;
		}
		@Override
		public double insertWeight(char cInserted) {
			Double d = weights.get("ε,"+cInserted);
			if(d == null)
				return INIT_FULL_WEIGHT;
			else
				return d;
		}
		@Override
		public double deleteWeight(char cDeleted) {
			Double d = weights.get(cDeleted+",ε");
			if(d == null)
				return INIT_FULL_WEIGHT;
			else
				return d;
		}
	};

	protected double getMinWeight() {
		double min = Double.MAX_VALUE;
		for(Double d : weights.values())
			if(d < min)
				min = d;
		return min;
	}

	@Override
	public HashMap<String, Double> getWeights() {
		return weights;
	}

	@Override
	public double getDistance(String sp, String tp) {
		// TODO just for testing REEDED against Modified PassJoin
		return wed.proximity(sp, tp);
//		return 1.0 / (1.0 + wed.proximity(sp, tp));
	}
	
	
}
