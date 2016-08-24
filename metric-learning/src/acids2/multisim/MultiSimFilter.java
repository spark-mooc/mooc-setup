package acids2.multisim;

import java.util.ArrayList;

import acids2.Couple;
import acids2.Resource;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public abstract class MultiSimFilter {
	
	protected MultiSimSimilarity similarity;
	
	public MultiSimFilter(MultiSimSimilarity similarity) {
		this.similarity = similarity;
	}
	
	public abstract ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta);
	
	public abstract ArrayList<Couple> filter(ArrayList<Couple> intersection, String propertyName, double theta);
	
	protected boolean verbose = true;
	
	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}
	
}
