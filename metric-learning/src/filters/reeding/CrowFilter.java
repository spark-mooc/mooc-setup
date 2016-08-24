package filters.reeding;

import java.util.ArrayList;
import java.util.HashMap;

import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import filters.WeightedNGramFilter;

/**
 * 
 * @author Axel Ngonga <ngonga@informatik.uni-leipzig.de>
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class CrowFilter extends WeightedNGramFilter {
		
	public CrowFilter(Property p) {
		super();
		property = p;
	}
	
	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {
		
		ArrayList<Couple> results = new ArrayList<Couple>();

		long start = System.currentTimeMillis();
		
		double tmu = theta * getMinWeight();
		double upper = (2 - tmu) / tmu;
		double lower = tmu / (2 - tmu);
		double kappa = (2 - theta) / theta;
		int i=0;
		for(Couple c : intersection) {
			Resource s = c.getSource(), t = c.getTarget();
			String src = s.getPropertyValue(propertyName), tgt = t.getPropertyValue(propertyName);
			// definition-based filter
			double ws = weightSum(src), wt = weightSum(tgt);
			double w_plus = kappa * ws, w_minus = ws / kappa;
			if(w_minus <= wt && wt <= w_plus) {
				// index-based filter
				double t0 = sw(tgt), s1 = mw(src), s0 = sw(src), t1 = mw(tgt);
				ArrayList<String> ng0s = getNgrams(s.getPropertyValue(propertyName), n);
				ArrayList<String> ng0t = getNgrams(t.getPropertyValue(propertyName), n);
				int ngs = ng0s.size(), ngt = ng0t.size();
				double k = theta * Math.min(s0, t0) / 2 * (s1 * ngs + t1 * ngt);
				ArrayList<String> share = intersect(ng0s, ng0t);
				if(share.size() >= k) {
					// length-aware filter
					if(ngs <= ngt * upper && ngs >= ngt * lower) {
						// refined length-aware filter
//						double upper2 = (2 * t0 - theta * s1) / (theta * t1);
//						double lower2 = (theta * t1) / (2 * t0 - theta * s1);
//						if(ngs <= ngt * upper2 && ngs >= ngt * lower2) {
							// similarity calculation
							double sim = this.getDistance(src, tgt);
							if(sim >= theta) {
								c.setDistance(sim, this.property.getIndex());
								results.add(c);
							}
//						}
					}
				}
			}
			if(++i % 1000 == 0)
				System.out.print(".");
		}
		System.out.println();
		
		if(this.isVerbose()) {
			double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
			System.out.println("CROW: Join done in "+compTime+" seconds. Couples: "+results.size());
		}	
		
		return results;
	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {

		ArrayList<Couple> results = new ArrayList<Couple>();

		long start = System.currentTimeMillis();
				
		double tmu = theta * getMinWeight();
		double upper = (2 - tmu) / tmu;
		double lower = tmu / (2 - tmu);
		
		HashMap<Resource, ArrayList<String>> ngLs = new HashMap<Resource, ArrayList<String>>();
		HashMap<Resource, Double> ngWs = new HashMap<Resource, Double>();
		for(Resource s : sources) {
			String src = s.getPropertyValue(propertyName);
			ngLs.put(s, getNgrams(src, n));
			ngWs.put(s, weightSum(src));
		}
		HashMap<Resource, ArrayList<String>> ngLt = new HashMap<Resource, ArrayList<String>>();
		HashMap<Resource, Double> ngWt = new HashMap<Resource, Double>();
		HashMap<Resource, Double> t0t = new HashMap<Resource, Double>();
		for(Resource t : targets) {
			String tgt = t.getPropertyValue(propertyName);
			ngLt.put(t, getNgrams(tgt, n));
			ngWt.put(t, weightSum(tgt));
			t0t.put(t, sw(tgt));
		}
		int c1=0,c2=0,c3=0,c4=0;
		double kappa = (2 - theta) / theta;
		for(Resource s : sources) {
			String src = s.getPropertyValue(propertyName);
			ArrayList<String> ng0s = ngLs.get(s);
			int ngs = ng0s.size();
			double s1 = mw(src), s0 = sw(src), ws = ngWs.get(s);
			double w_plus = kappa * ws, w_minus = ws / kappa;
			for(Resource t : targets) {
				String tgt = t.getPropertyValue(propertyName);
				// definition-based filter
				double wt = ngWt.get(t);
				if(w_minus <= wt && wt <= w_plus) {
					c1++;
					// index-based filter
					double t0 = t0t.get(t), t1 = mw(tgt);
					ArrayList<String> ng0t = ngLt.get(t);
					int ngt = ng0t.size();
					double k = theta * Math.min(s0, t0) / 2 * (s1 * ngs + t1 * ngt);
					ArrayList<String> share = intersect(ng0s, ng0t);
					if(share.size() >= k) {
						c4++;
						// length-aware filter
						if(ngs <= ngt * upper && ngs >= ngt * lower) {
							c2++;
							// refined length-aware filter
//							double upper2 = (2 * t0 - theta * s1) / (theta * t1);
//							double lower2 = (theta * t1) / (2 * t0 - theta * s1);
//							if(ngs <= ngt * upper2 && ngs >= ngt * lower2) {
								c3++;
								// similarity calculation
								double sim = this.getDistance(src, tgt);
								if(sim >= theta) {
									Couple c = new Couple(s, t);
									c.setDistance(sim, this.property.getIndex());
									results.add(c);
								}
//							}
						}
					}
				}
			}
		}
		
		System.out.println((sources.size()*targets.size())+"\tDEF="+c1+"\tLEN="+c2+"\tREF="+c3+"\tIND="+c4+"\t"+results.size());
		
		if(this.isVerbose()) {
			double compTime = (System.currentTimeMillis()-start)/1000.0;
			System.out.println("CROW: Join done in "+compTime+" seconds. Couples: "+results.size());
		}
		
		return results;
	}
	
	@SuppressWarnings("unused")
	private ArrayList<String> removeZeros(ArrayList<String> ngrams) {
		ArrayList<String> output = new ArrayList<String>();
		for(String ng : ngrams)
			if(getWeight(ng) > 0)
				output.add(ng);
		return output;
	}


	private double sw(String s) {
		ArrayList<String> ngs = getNgrams(s, n);
		double min = 1;
		for(String ng : ngs) {
			double w = getWeight(ng);
			if(w < min)
				min = w;
		}
		return min;
	}
	
	private double mw(String s) {
		ArrayList<String> ngs = getNgrams(s, n);
		double max = 0;
		for(String ng : ngs) {
			double w = getWeight(ng);
			if(w > max)
				max = w;
		}
		return max;
	}

	private double weightSum(String s) {
		ArrayList<String> ngs = getNgrams(s, n);
		double sum = 0;
		for(String ng : ngs)
			sum += getWeight(ng);
		return sum;
	}

}
