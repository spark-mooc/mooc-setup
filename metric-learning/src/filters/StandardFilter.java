package filters;

import java.util.ArrayList;
import java.util.HashMap;

import acids2.Couple;
import acids2.Property;
import acids2.Resource;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public abstract class StandardFilter {
	
	public abstract ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta);
	
	public abstract ArrayList<Couple> filter(ArrayList<Couple> intersection, String propertyName, double theta);
	
	public abstract double getDistance(String sp, String tp);
	
	public abstract HashMap<String, Double> getWeights();

	public abstract void init(ArrayList<Resource> sources, ArrayList<Resource> targets);
	
	protected boolean verbose = true;
	
	protected Property property = null;

	public Property getProperty() {
		return property;
	}

	public void setProperty(Property property) {
		this.property = property;
	}

	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
	
}
