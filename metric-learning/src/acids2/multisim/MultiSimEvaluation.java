package acids2.multisim;

import java.util.ArrayList;

import acids2.Couple;
import acids2.Resource;
import acids2.classifiers.svm.SvmHandler;
import acids2.output.Fscore;

public class MultiSimEvaluation {
	
	private SvmHandler svmHandler;
	private MultiSimSetting setting;

	public MultiSimEvaluation(MultiSimSetting setting) {
		super();
		this.setting = setting;
		this.svmHandler = setting.getSvmHandler();
	}

	public void labelAll(ArrayList<Couple> couples) {
		for(Couple c : couples)
			c.setPositive(setting.getOracle().ask(c));
	}
	
	public Fscore evaluateOn(ArrayList<Couple> couples) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		for(Couple c : couples) {
			if(c.isPositive()) {
				if(svmHandler.classify(c))
					tp++;
				else
					fn++;
			} else {
				if(svmHandler.classify(c))
					fp++;
				else
					tn++;
			}
		}
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}

	public Fscore fastEvaluation(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		
		ArrayList<MultiSimSimilarity> sims = setting.getMeasures().getAllSimilarities();
		ArrayList<String> mapping = setting.getOracle().getOraclesAnswers();
		
		double tp = 0, tn = 0, fp = 0, fn = 0;
		
		for(String map : mapping) {
			String[] ids = map.split("#");
			Resource src = null, tgt = null;
			for(Resource s : sources)
				if(s.getID().equals(ids[0])) {
					src = s;
					break;
				}
			for(Resource t : targets)
				if(t.getID().equals(ids[1])) {
					tgt = t;
					break;
				}
			Couple c = new Couple(src, tgt);
			
			for(MultiSimSimilarity sim : sims) 
				c.setDistance(sim.getSimilarity(src.getPropertyValue(sim.getProperty().getName()),
						tgt.getPropertyValue(sim.getProperty().getName())), sim.getIndex());
			
			setting.getMeasures().estimateMissingValues(c);
			
			if(svmHandler.classify(c))
				tp++;
			else
				fn++;
		}

		ArrayList<Couple> intersection = new ArrayList<Couple>();
		boolean allInfinite = true, performCartesianP = true;
		for(int index=0; index<sims.size(); index++) {
			MultiSimSimilarity sim = sims.get(index);
			double theta_i = setting.getMeasures().computeThreshold(sim);
			System.out.println("Property: "+sim.getProperty().getName()+"\ttheta_"+index+" = "+theta_i);
			if(sim.isComputed() && sim.getFilter() != null) {
				allInfinite = false;
				if(performCartesianP) { // first property works on the entire Cartesian product.
					if(sim.getProperty().getDatatype() == MultiSimDatatype.TYPE_STRING)
						intersection = sim.getFilter().filter(sources, targets, sim.getProperty().getName(), theta_i);
					else
						intersection = sim.getFilter().filter(sources, targets, sim.getProperty().getName(), theta_i);
					performCartesianP = false;
				} else {
					if(sim.getProperty().getDatatype() == MultiSimDatatype.TYPE_STRING)
						intersection = sim.getFilter().filter(intersection, sim.getProperty().getName(), theta_i);
					else
						intersection = sim.getFilter().filter(intersection, sim.getProperty().getName(), theta_i);
				}
			}
			System.out.println("intersection size: "+intersection.size());
		}
		if(allInfinite) {
			System.out.println("No thresholds available, switching to naive evaluation.");
			return naiveEvaluation(sources, targets);
		}
		
		for(MultiSimSimilarity sim : sims)
			if(!sim.isComputed() || sim.getFilter() == null)
				for(Couple c : intersection)
					c.setDistance(sim.getSimilarity(c.getSource().getPropertyValue(sim.getProperty().getName()),
							c.getTarget().getPropertyValue(sim.getProperty().getName())), sim.getIndex());
		
		for(Couple c : intersection) 
			setting.getMeasures().estimateMissingValues(c);
		
		for(Couple c : intersection)
			if(!setting.getOracle().ask(c)) {
				if(!svmHandler.classify(c))
					tn++;
				else
					fp++;
			}
		
		tn = tn + (sources.size() * targets.size() - mapping.size() - (tn + fp));
		
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}
		
	public Fscore naiveEvaluation(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		
		ArrayList<MultiSimSimilarity> sims = setting.getMeasures().getAllSimilarities();
		ArrayList<String> mapping = setting.getOracle().getOraclesAnswers();

		double tp = 0, tn = 0, fp = 0, fn = 0;
		double[] val = new double[sims.size()];
		
		System.out.print("Evaluating");
		System.out.println();
		int count = 0;		
		for(Resource s : sources) {
			for(Resource t : targets) {
				for(int j=0; j<sims.size(); j++) {
					MultiSimSimilarity sim = sims.get(j);
					String p = sim.getProperty().getName();
					val[j] = sim.getSimilarity(s.getPropertyValue(p), t.getPropertyValue(p));
				}
				val = setting.getMeasures().estimateMissingValues(val);
				if(mapping.contains(s.getID()+"#"+t.getID())) {
					if(svmHandler.classify(val))
						tp++;
					else {
						fn++;
//						System.out.println("false neg: "+s.getID()+"#"+t.getID());
					}
				} else {
					if(svmHandler.classify(val)) {
						fp++;
//						System.out.println("false pos: "+s.getID()+"#"+t.getID());
					} else
						tn++;
				}
				if(++count % 100000 == 0)
					System.out.print(".");
			}
		}
		System.out.println();
        
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}

	
	
}
