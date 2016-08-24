package filters.reeded;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import filters.WeightedEditDistanceFilter;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class ReededFilter extends WeightedEditDistanceFilter {
	
	public ReededFilter(Property p) {
		super();
		property = p;
	}
	
	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {
		
		ArrayList<Couple> results = new ArrayList<Couple>();

		double tau = theta / getMinWeight();
		
		long start = System.currentTimeMillis();
		
		HashMap<String, Vector<Character>> index = new HashMap<String, Vector<Character>>();
		for(Couple c : intersection) {
			Resource s = c.getSource();
			String sp = s.getPropertyValue(propertyName);
			Vector<Character> cs;
			if(!index.containsKey(sp)) {
				cs = new Vector<Character>();
				for(int i=0; i<sp.length(); i++)
					cs.add(sp.charAt(i));
				index.put(sp, cs);
			} else
				cs = index.get(sp);
			Resource t = c.getTarget();
			String tp = t.getPropertyValue(propertyName);
			Vector<Character> ct;
			if(!index.containsKey(tp)) {
				ct = new Vector<Character>();
				for(int i=0; i<tp.length(); i++)
					ct.add(tp.charAt(i));
				index.put(tp, ct);
			} else
				ct = index.get(tp);
			reededCore(s, t, sp, tp, cs, ct, tau, theta, results);
		}
		
		if(this.isVerbose()) {
			double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
			System.out.println("REEDED: Join done in "+compTime+" seconds.");
		}	
		
		return results;
	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {
		
		ArrayList<Couple> results = new ArrayList<Couple>();

		double tau = theta / getMinWeight();
		
		long start = System.currentTimeMillis();
		
		HashMap<String, Vector<Character>> index = new HashMap<String, Vector<Character>>();
		for(Resource s : sources) {
			String sp = s.getPropertyValue(propertyName);
			Vector<Character> cs = new Vector<Character>();
			for(int i=0; i<sp.length(); i++)
				cs.add(sp.charAt(i));
			index.put(sp, cs);
		}
		for(Resource t : targets) {
			String tp = t.getPropertyValue(propertyName);
			Vector<Character> ct = new Vector<Character>();
			for(int i=0; i<tp.length(); i++)
				ct.add(tp.charAt(i));
			index.put(tp, ct);
		}
		for(Resource s : sources) {
			for(Resource t : targets) {
				String sp = s.getPropertyValue(propertyName);
				String tp = t.getPropertyValue(propertyName);
				reededCore(s, t, sp, tp, index.get(sp), index.get(tp), tau, theta, results);
			}
		}
				
		if(this.isVerbose()) {
			double compTime = (System.currentTimeMillis()-start)/1000.0;
			System.out.println("REEDED: Join done in "+compTime+" seconds.");
		}
		
		return results;
	}
	
	private double exclDisjSize(Vector<Character> cs, Vector<Character> ct) {
		Vector<Character> cs2 = new Vector<Character>(cs);
		Vector<Character> ct2 = new Vector<Character>(ct);
		for(Character c1 : ct)
			if(cs2.remove(c1))
				ct2.remove(c1);
		return cs2.size()+ct2.size();
	}
	
	private void reededCore(Resource s, Resource t, String sp, String tp, Vector<Character> cs, Vector<Character> ct, 
			double tau, double theta, ArrayList<Couple> results) {
		
		if(Math.abs(sp.length() - tp.length()) <= tau) {
			// (...) + (size % 2);
			if(Math.ceil(exclDisjSize(cs, ct) / 2.0) <= tau) {
				//  Verification.
				double d = this.getDistance(sp, tp);
				if(d <= theta) {
					Couple c = new Couple(s, t);
					c.setDistance(d, this.property.getIndex());
					results.add(c);
				}
			}
		}
		
	}

	@Override
	public void init(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		// TODO Auto-generated method stub
		
	}


}
