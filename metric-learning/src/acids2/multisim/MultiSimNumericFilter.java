package acids2.multisim;

import java.util.ArrayList;

import acids2.Couple;
import acids2.Resource;

public class MultiSimNumericFilter extends MultiSimFilter {

	private MultiSimNumericSimilarity nsim;
	
	public MultiSimNumericFilter(MultiSimSimilarity similarity) {
		super(similarity);
		nsim = (MultiSimNumericSimilarity) similarity;
	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {
		
		ArrayList<Couple> result = new ArrayList<Couple>();
		
		System.out.print("Numeric filtering");
		int i = 0;
		for(Resource s : sources) {
			for(Resource t : targets) {
				double d = nsim.getSimilarity(s.getPropertyValue(propertyName), 
						t.getPropertyValue(propertyName));
				if(d >= theta) {
					Couple c = new Couple(s, t);
					c.setDistance(d, nsim.getIndex());
					result.add(c);
				}
				if(++i % 100000 == 0)
					System.out.print(".");
			}
		}
		System.out.println();
		
		return result;
	}

	@Override
	public ArrayList<Couple> filter(ArrayList<Couple> intersection,
			String propertyName, double theta) {
		
		ArrayList<Couple> result = new ArrayList<Couple>();
		
		System.out.print("Numeric filtering");
		int i = 0;
		for(Couple c : intersection) {
			double d = nsim.getSimilarity(c.getSource().getPropertyValue(propertyName), 
					c.getTarget().getPropertyValue(propertyName));
			if(d >= theta) {
				c.setDistance(d, nsim.getIndex());
				result.add(c);
			}
			if(++i % 100000 == 0)
				System.out.print(".");
		}
		System.out.println();
		
		return result;
	}

}
