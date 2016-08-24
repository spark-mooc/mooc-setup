package filters.mahalanobis;

import java.util.ArrayList;
import java.util.HashMap;

import utility.Transform;
import utility.ValueParser;
import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import filters.StandardFilter;

public class MahalaFilter extends StandardFilter {

	private ArrayList<Double> extrema = new ArrayList<Double>();
	
	public MahalaFilter(Property p) {
		property = p;
	}
	
	@Override
	public ArrayList<Couple> filter(ArrayList<Resource> sources, ArrayList<Resource> targets, 
			String pname, double theta) {
		ArrayList<Couple> results = new ArrayList<Couple>();
		for(Resource s : sources)
			for(Resource t : targets)
				mahalaCore(s, t, pname, theta, results);
		return results;
	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection, String pname, double theta) {
		ArrayList<Couple> results = new ArrayList<Couple>();
		for(Couple c : intersection)
			mahalaCore(c.getSource(), c.getTarget(), pname, theta, results);
		return results;
	}

	private void mahalaCore(Resource s, Resource t, String pname, double theta, ArrayList<Couple> results) {
		String sp = s.getPropertyValue(pname);
		String tp = t.getPropertyValue(pname);
		double d = getDistance(sp, tp);
		double theta_min = Transform.toDistance(theta);
		if(d <= theta_min) {
			Couple c = new Couple(s, t);
			// distance values are then normalized into [0,1]
			c.setDistance(d, this.property.getIndex());
			results.add(c);
		}
	}

	@Override
	public double getDistance(String sp, String tp) {
		double sd = ValueParser.parse(sp);
		double td = ValueParser.parse(tp);
		return normalize(Math.abs(sd-td));
	}

	@Override
	public HashMap<String, Double> getWeights() {
		return new HashMap<String, Double>();
	}
	
	public void setExtrema(ArrayList<Double> ext) {
		this.extrema = ext;
	}

	public ArrayList<Double> getExtrema() {
		return extrema;
	}

	public void computeExtrema(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		extrema.clear();
		String pname = property.getName();
		double maxS = Double.NEGATIVE_INFINITY, minS = Double.POSITIVE_INFINITY;
		for(Resource s : sources) {
			double d = ValueParser.parse( s.getPropertyValue(pname) );
			if(d > maxS) maxS = d;
			if(d < minS) minS = d;
		}
		extrema.add(maxS);
		extrema.add(minS);
		double maxT = Double.NEGATIVE_INFINITY, minT = Double.POSITIVE_INFINITY;
		for(Resource t : targets) {
			double d = ValueParser.parse( t.getPropertyValue(pname) );
			if(d > maxT) maxT = d;
			if(d < minT) minT = d;
		}
		extrema.add(maxT);
		extrema.add(minT);
		System.out.println(extrema.toString());
	}

	private double normalize(double value) {
		// incomplete information means similarity = 0 
		if(Double.isNaN(value))
			return 0.0;
		double maxS = extrema.get(0), minS = extrema.get(1), maxT = extrema.get(2), minT = extrema.get(3);
		double denom = Math.max(maxT - minS, maxS - minT);
		if(denom == 0.0)
			return 1.0;
		else
			return 1.0 - value / denom;
	}

	@Override
	public void init(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		// this is for tf-idf indexing. leave blank.
	}

}
