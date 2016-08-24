package acids2.multisim;

import java.util.ArrayList;
import java.util.HashMap;

import utility.ValueParser;
import acids2.Resource;

public class MultiSimNumericSimilarity extends MultiSimSimilarity {
	
	private HashMap<String, Double> extrema = new HashMap<String, Double>();
	private MultiSimNumericFilter filter = new MultiSimNumericFilter(this);

	public MultiSimNumericSimilarity(MultiSimProperty property, int index) {
		super(property, index);
		computeExtrema();
	}

	@Override
	public String getName() {
		return "Numeric Similarity";
	}
	
	@Override
	public int getDatatype() {
		return MultiSimDatatype.TYPE_NUMERIC;
	}
	
	@Override
	public double getSimilarity(String a, String b) {
		double sd = Math.log10(ValueParser.parse(a));
		double td = Math.log10(ValueParser.parse(b));
		return Math.pow(normalize(Math.abs(sd-td)), 3);
	}

	public HashMap<String, Double> getWeights() {
		return new HashMap<String, Double>();
	}
	
	public void setExtrema(HashMap<String, Double> ext) {
		this.extrema = ext;
	}

	public HashMap<String, Double> getExtrema() {
		return extrema;
	}

	public void computeExtrema() {
		ArrayList<Resource> sources = property.getMeasures().getSetting().getSources();
		ArrayList<Resource> targets = property.getMeasures().getSetting().getTargets();
		extrema.clear();
		String pname = property.getName();
		double maxS = Double.NEGATIVE_INFINITY, minS = Double.POSITIVE_INFINITY;
		for(Resource s : sources) {
			double d = Math.log10(ValueParser.parse( s.getPropertyValue(pname) ));
			if(d > maxS) maxS = d;
			if(d < minS) minS = d;
		}
		extrema.put("maxS", maxS);
		extrema.put("minS", minS);
		double maxT = Double.NEGATIVE_INFINITY, minT = Double.POSITIVE_INFINITY;
		for(Resource t : targets) {
			double d = Math.log10(ValueParser.parse( t.getPropertyValue(pname) ));
			if(d > maxT) maxT = d;
			if(d < minT) minT = d;
		}
		extrema.put("maxT", maxT);
		extrema.put("minT", minT);
		System.out.println(extrema.toString());
	}

	private double normalize(double value) {
		// incomplete information means similarity = 0
		if(Double.isNaN(value))
			return Double.NaN;
		double denom = Math.max(extrema.get("maxT") - extrema.get("minS"), 
				extrema.get("maxS") - extrema.get("minT"));
		if(denom == 0.0)
			return 1.0;
		else
			return 1.0 - value / denom;
	}

	@Override
	public MultiSimFilter getFilter() {
		return filter;
	}

}
