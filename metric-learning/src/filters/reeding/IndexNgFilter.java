package filters.reeding;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

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
public class IndexNgFilter extends WeightedNGramFilter {
		
	public IndexNgFilter(Property p) {
		super();
		property = p;
//		getWeights().put("abc", 0.2);
//		getWeights().put("xyz", 0.4);
	}
	
	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {
		
		ArrayList<Couple> results = new ArrayList<Couple>();

		long start = System.currentTimeMillis();
		
		// TODO algorithm
		
		if(this.isVerbose()) {
			double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
			System.out.println("NewNG: Join done in "+compTime+" seconds.");
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
		
		// build "n-gram to resource" and "length to resource" indexes
		HashMap<String, TreeSet<Resource>> Is = new HashMap<String, TreeSet<Resource>>();
		HashMap<Resource, ArrayList<String>> ngLs = new HashMap<Resource, ArrayList<String>>();
		for(Resource s : sources) {
			String src = s.getPropertyValue(propertyName);
			ArrayList<String> ng0s = removeZeros(getNgrams(src, n));
			ngLs.put(s, ng0s);
			for(String ng : ng0s) {
				TreeSet<Resource> tr = Is.get(ng);
				if(tr == null) {
					tr = new TreeSet<Resource>();
					Is.put(ng, tr);
				}
				tr.add(s);
			}
		}
		HashMap<String, TreeSet<Resource>> It = new HashMap<String, TreeSet<Resource>>();
		HashMap<Resource, ArrayList<String>> ngLt = new HashMap<Resource, ArrayList<String>>();
		for(Resource t : targets) {
			String tgt = t.getPropertyValue(propertyName);
			ArrayList<String> ng0t = removeZeros(getNgrams(tgt, n));
			ngLt.put(t, ng0t);
			for(String ng : ng0t) {
				TreeSet<Resource> tr = It.get(ng);
				if(tr == null) {
					tr = new TreeSet<Resource>();
					It.put(ng, tr);
				}
				tr.add(t);
			}
		}
		
		int count1=0, count2=0, count3=0;
		for(String ng : Is.keySet()) {
			TreeSet<Resource> srcset = Is.get(ng), tgtset = It.get(ng);
			TreeSet<Couple> cand = new TreeSet<Couple>();
			for(Resource s : srcset) {
				for(Resource t : tgtset) {
					ArrayList<String> ng0s = ngLs.get(s), ng0t = ngLt.get(t);
					int ngs = ng0s.size(), ngt = ng0t.size();
					if(ngs <= ngt * upper && ngs >= ngt * lower) {
						Couple c = new Couple(s, t);
						if(!results.contains(c))
							cand.add(c);
					}
				}
			}
			count1 += cand.size();
			for(Couple c : cand) {
				Resource s = c.getSource(), t = c.getTarget();
				String src = s.getPropertyValue(propertyName), tgt = t.getPropertyValue(propertyName);
				ArrayList<String> ng0s = ngLs.get(s), ng0t = ngLt.get(t);
				int ngs = ng0s.size(), ngt = ng0t.size();
				double alpha = mw(src), delta = sw(src), beta = mw(tgt), gamma = sw(tgt);
				double upper2 = (2 * gamma - theta * alpha) / (theta * beta);
				double lower2 = (theta * beta) / (2 * gamma - theta * alpha);
				double k = theta * Math.min(delta, gamma) / 2 * (ngs + ngt);
				if(ngs <= ngt * upper2 && ngs >= ngt * lower2) {
					count2++;
					ArrayList<String> share = intersect(ng0s, ng0t);
					if(share.size() >= k) {
						count3++;
						// similarity calculation
						double sim = this.getDistance(src, tgt);
						if(sim >= theta) {
							c.setDistance(sim, this.property.getIndex());
							results.add(c);
						}
					}
						
				}
			}
		}
		
		if(this.isVerbose()) {
			System.out.println("count1 = "+count1);
			System.out.println("count2 = "+count2);
			System.out.println("count3 = "+count3);
			System.out.println("results = "+results.size());
			System.out.println("out of "+(sources.size()*targets.size()));
			double compTime = (System.currentTimeMillis()-start)/1000.0;
			System.out.println("NewNG: Join done in "+compTime+" seconds.");
		}
		
		return results;
	}
	
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
		double max = 1;
		for(String ng : ngs) {
			double w = getWeight(ng);
			if(w > max)
				max = w;
		}
		return max;
	}


}
