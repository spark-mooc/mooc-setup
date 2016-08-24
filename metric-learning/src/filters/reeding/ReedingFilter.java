package filters.reeding;

import java.util.ArrayList;
import java.util.TreeSet;

import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import filters.WeightedNGramFilter;

/**
 * Rapid Execution of weightED N-Grams (REEDiNG) filter.
 *  
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class ReedingFilter extends WeightedNGramFilter {
	
	public ReedingFilter(Property p) {
		super();
		property = p;
	}
	
	@Override
	public ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {

		ArrayList<Couple> results;

		long start = System.currentTimeMillis();
		
//		for(Resource s : sources)
//			for(Resource t : targets)
//				reedingCore(s, t, propertyName, theta, results);
		
		results = WeightedPPJoinPlus.run(sources, targets, propertyName, theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
		System.out.println("REEDiNG: Join done in "+compTime+" seconds.");
		
		return new ArrayList<Couple>(results);

	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {

		ArrayList<Couple> results = new ArrayList<Couple>();

		long start = System.currentTimeMillis();
		for(Couple c : intersection)
			reedingCore(c.getSource(), c.getTarget(), propertyName, theta, results);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
		System.out.println("REEDiNG: Join done in "+compTime+" seconds.");
		
		return results;

	}
	
	private void reedingCore(Resource s, Resource t, String propertyName, 
			double theta, ArrayList<Couple> results) {
		
		String sp = s.getPropertyValue(propertyName);
		String tp = t.getPropertyValue(propertyName);

		double sigma_max = sigmaMax(sp.length(), tp.length());
		
		if(sigma_max >= theta) {
			
			
			
			double d = this.getDistance(sp, tp);
			if(d <= theta) {
				Couple cpl = new Couple(s, t);
				cpl.setDistance(d, this.property.getIndex());
				results.add(cpl);
			}
		}
		
	}
	
	private double sigmaMax(int sl, int tl) {
		double inters = Math.min(sl, tl) * this.getMaxWeight();
		double ex_disjunction = (Math.abs(sl - tl) / 2.0 + n - 1) * this.getMinWeight();
		return inters / (inters + ex_disjunction);
	}


}
