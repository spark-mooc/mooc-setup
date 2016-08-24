package filters.edjoin;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeSet;

import utility.SystemOutHandler;
import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import algorithms.edjoin.EdJoinPlus;
import algorithms.edjoin.Entry;
import filters.WeightedEditDistanceFilter;

public class EdJoinFilter extends WeightedEditDistanceFilter {

	public EdJoinFilter(Property p) {
		property = p;
	}
	
	public ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {
		
		TreeSet<Entry> sTree = new TreeSet<Entry>();
        for(Resource s : sources)
            sTree.add(new Entry(s.getID(), s.getPropertyValue(propertyName)));
        TreeSet<Entry> tTree = new TreeSet<Entry>();
        for(Resource s : targets)
            tTree.add(new Entry(s.getID(), s.getPropertyValue(propertyName)));
        
        ArrayList<Couple> results = new ArrayList<Couple>();
		TreeSet<String> temp = new TreeSet<String>();
		
		// ceiling is required because EdJoin works with integers only.
		// e.g., if we need (s,t) with d(s,t) <= 1.8, we must ask for interval [0,2].
		int tau = (int) Math.ceil( theta / getMinWeight() );
		
	    long start = System.currentTimeMillis();

		SystemOutHandler.shutDown();
		temp = EdJoinPlus.runOnEntries(0, tau, sTree, tTree);
		SystemOutHandler.turnOn();

        for(String ids : temp) {
        	String[] id = ids.split("#");
        	Resource s = null;
        	for(Resource src : sources)
        		if(src.getID().equals(id[0])) {
        			s = src;
        			break;
        		}
        	Resource t = null;
        	for(Resource tgt : targets)
        		if(tgt.getID().equals(id[1])) {
        			t = tgt;
        			break;
        		}
        	String sp = s.getPropertyValue(propertyName);
        	String tp = t.getPropertyValue(propertyName);
        	double d = this.getDistance(sp, tp);
        	if(d <= theta) {
            	Couple c = new Couple(s, t);
        		c.setDistance(d, this.property.getIndex());
        		results.add(c);
        	}
        }
        
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
		if(this.isVerbose())
			System.out.print(compTime+"\t");
		
//		System.out.println("count = "+count);
		return results;
	}


	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public HashMap<String, Double> getWeights() {
		return new HashMap<String, Double>();
	}

	@Override
	public void init(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		// TODO Auto-generated method stub
		
	}
}
