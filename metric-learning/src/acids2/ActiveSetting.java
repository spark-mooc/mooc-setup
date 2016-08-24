package acids2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import libsvm.svm_parameter;

import utility.GammaComparator;
import acids2.classifiers.svm.SvmHandler;
import acids2.output.CoupleStore;
import acids2.output.Fscore;
import acids2.output.RawDataScriptCreator;
import acids2.output.SageScriptCreator;
import acids2.test.Test;
import filters.mahalanobis.MahalaFilter;
import filters.reeding.CrowFilter;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class ActiveSetting extends TestUnit {

	private ArrayList<Resource> sources, targets;
	private int k;
	private boolean tfidf;
	
	private final int dk = 5;

	private ArrayList<Property> props = new ArrayList<Property>();
	private SvmHandler m; 
	
	public ActiveSetting(ArrayList<Resource> sources, ArrayList<Resource> targets, int k, boolean tfidf, String dataset) {
		
		this.sources = sources;
		this.targets = targets;
		this.k = k;
		this.tfidf = tfidf;
		
		m = new SvmHandler(svm_parameter.POLY);
		
		initialization();
		
		ArrayList<Couple> couples = new ArrayList<Couple>();
		int mapSize = Math.min(sources.size(), targets.size());
		
		for(int i=0; i<props.size() && couples.isEmpty(); i++) {
			System.out.println("Processing property: "+props.get(i).getName());
			for(double threshold=0.5; ; threshold-=0.1) {
				System.out.println("threshold = "+threshold);
				Property p = props.get(i);
				couples = p.getFilter().filter(sources, targets, p.getName(), threshold);
				System.out.println("size = "+couples.size());
				if(couples.size() >= mapSize) {
					p.setFiltered(true);
					break;
				}
			}
		}
		
		for(Property p : props) {
			if(!p.isFiltered()) {
				for(Couple c : couples) {
					Resource s = c.getSource();
					Resource t = c.getTarget();
					double d = p.getFilter().getDistance(s.getPropertyValue(p.getName()), t.getPropertyValue(p.getName()));
					c.setDistance(d, p.getIndex());
				}
			}
		}

		
//		try {
//			for(double thr=0.5; couples.size() < mapSize; thr-=0.1)
//				couples = CoupleStore.filterCouplesFromCartesian(dataset, thr);
//			CoupleStore.saveFilteredCouples(couples, dataset);
//			couples = CoupleStore.loadFilteredCouples(dataset);
//		} catch (IOException e2) {
//			e2.printStackTrace();
//		}
		
//		for(Couple c : couples) {
//			c.setDistance(0.0, 1);
//			c.setDistance(0.0, 2);
//		}
//		double[] w = {1.0, 0.0, 0.0};
//		m.setW(w);
				
		m.initTheta(couples, mapSize);
//		m.setTheta(m.getN() - 0.5);
		
		ArrayList<Couple> labelled = new ArrayList<Couple>();

		for(int i=0; labelled.size() < k; i++) {
			
			System.out.println("\n### Iteration = "+i+" ###");
			
			ArrayList<Couple> posInformative = new ArrayList<Couple>();
			ArrayList<Couple> negInformative = new ArrayList<Couple>();
			
			for(Couple c : couples) {
		        c.setGamma( m.computeGamma(c) ); // TODO Change to m.computeGamma(c);
				if(m.classify(c))
					posInformative.add(c);
				else
					negInformative.add(c);
			}
			
			System.out.println("theta = "+m.getTheta()+"\tpos = "+posInformative.size()+"\tneg = "+negInformative.size());
			
			Collections.sort(posInformative, new GammaComparator());
			Collections.sort(negInformative, new GammaComparator());
			
			ArrayList<Couple> poslbl = new ArrayList<Couple>();
			ArrayList<Couple> neglbl = new ArrayList<Couple>();
			
			for(Couple c : labelled)
				if(c.isPositive())
					poslbl.add(c);
				else
					neglbl.add(c);
			
			if(i == 0) {
				// required for right classifier orientation
				orientate(poslbl, neglbl, labelled);
				// train with dk most likely positive examples
				processTraining(posInformative, negInformative, labelled, poslbl, neglbl, this.dk, false);
			}
			
			// train with dk most informative positive examples
			processTraining(posInformative, negInformative, labelled, poslbl, neglbl, this.dk, true);
			// train with dk most informative negative examples
			processTraining(negInformative, posInformative, labelled, poslbl, neglbl, this.dk, true);
			
//			// train with dk random examples
//			for(int j=0; j<this.dk; j++) {
//				Resource s = sources.get((int) (sources.size() * Math.random()));
//				Resource t = targets.get((int) (targets.size() * Math.random()));
//				Couple c = new Couple(s, t);
//				for(Property p : props) {
//					double d = p.getFilter().getDistance(s.getPropertyValue(p.getName()), t.getPropertyValue(p.getName()));
//					c.setDistance(d, p.getIndex());
//				}
//				labelled.add(c);
//				if(askOracle(c)) {
//					c.setPositive(true);
//					poslbl.add(c);
//				} else {
//					c.setPositive(false);
//					neglbl.add(c);
//				}
//			}
			
			System.out.println("Labeled pos: "+poslbl.size());
			System.out.println("Labeled neg: "+neglbl.size());
			
			boolean svmResponse = m.trace(poslbl, neglbl);
			if(!svmResponse) {
				// ask for more
				continue;
			}
			evaluateOn(labelled);

//			RawDataScriptCreator raw = new RawDataScriptCreator(dataset+"_"+this.k+"_iter"+i, false, false, true);
//			try {
//				raw.create(sources, targets, props, m.getWLinear(), m.getTheta());
//				new Evaluator("output/"+dataset+"_testset.txt", "output/"+dataset+"_"+this.k+"_iter"+i+"_classifier.txt");
//			} catch (Exception e) {
//				e.printStackTrace();
//			}
			
			// last iteration
			if(labelled.size() >= k) {
				System.out.println("Converging...");
				for(int j=0; m.getTheta() == -1.0 && j < 10; j++) {
					processTraining(posInformative, negInformative, labelled, poslbl, neglbl, 1, true);
					m.trace(poslbl, neglbl);
				}
			}
		}

//		// add 10,000 random examples to the plot
//		for(int j=0; j<10000; j++) {
//			Resource s = sources.get((int) (sources.size() * Math.random()));
//			Resource t = targets.get((int) (targets.size() * Math.random()));
//			Couple c = new Couple(s, t);
//			for(Property p : props) {
//				double d = p.getFilter().getDistance(s.getPropertyValue(p.getName()), t.getPropertyValue(p.getName()));
//				c.setDistance(d, p.getIndex());
//			}
//			if(askOracle(c)) {
//				c.setPositive(true);
//			} else {
//				c.setPositive(false);
//			}
//			couples.add(c);
//		}
		
		System.out.println("Evaluating on filtered subset:");
		evaluateOn(couples);
		
		SageScriptCreator ssc = new SageScriptCreator();
		try {
			ssc.create(couples, props, m.getWLinear(), m.getTheta());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		RawDataScriptCreator raw = new RawDataScriptCreator(dataset+"_"+this.k, false, false, true);
		try {
			raw.create(sources, targets, props, m.getWLinear(), m.getTheta());
			new Evaluator("output/"+dataset+"_testset.txt", "output/"+dataset+"_"+this.k+"_classifier.txt");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
//		System.out.println("\nEVALUATION:");
//		
//		fs = fastEvaluation(sources, targets);
//		
//		RawDataScriptCreator rd = new RawDataScriptCreator(""+System.currentTimeMillis(), true, false, false);
////		rd.setTrainingset(labelled);
//		try {
//			rd.create(sources, targets, props, m.getWLinear(), m.getTheta());
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

	}

	private Fscore evaluateOn(ArrayList<Couple> couples) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		for(Couple c : couples) {
			if(c.isPositive()) {
				if(m.classify(c))
					tp++;
				else
					fn++;
			} else {
				if(m.classify(c)) {
					fp++;
				} else
					tn++;
			}
		}
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}

	private Fscore fastEvaluation(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		int n = m.getN();
		
		ArrayList<String> mapping = Test.getOraclesAnswers();
		
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
			for(int i=0; i<n; i++) {
				Property p = props.get(i);
				double d = p.getFilter().getDistance(src.getPropertyValue(p.getName()), tgt.getPropertyValue(p.getName()));
				c.setDistance(d, p.getIndex());
			}
			if(m.classify(c))
				tp++;
			else
				fn++;
		}

		ArrayList<Couple> intersection = new ArrayList<Couple>();
		boolean allInfinite = true, performCartesianP = true;
		for(int i=0; i<n; i++) {
			Property p = props.get(i);
			double theta_i = computeMonteCarlo(i); // computeThreshold(i);
			if(!Double.isInfinite(theta_i))
				allInfinite = false;
			System.out.println("Property: "+p.getName()+"\ttheta_"+i+" = "+theta_i);
			if(!p.isNoisy()) {
				if(performCartesianP) { // first property works on the entire Cartesian product.
					if(p.getDatatype() == Property.TYPE_STRING) {
						CrowFilter ngf = new CrowFilter(p);
						ngf.setWeights(p.getFilter().getWeights());
						intersection = ngf.filter(sources, targets, p.getName(), theta_i);
					} else {
						intersection = p.getFilter().filter(sources, targets, p.getName(), theta_i);
					}
					performCartesianP = false;
				} else {
					if(p.getDatatype() == Property.TYPE_STRING) {
						CrowFilter ngf = new CrowFilter(p);
						ngf.setWeights(p.getFilter().getWeights());
						merge(intersection, ngf.filter(intersection, p.getName(), theta_i), i);
					} else {
						merge(intersection, p.getFilter().filter(intersection, p.getName(), theta_i), i);
					}
				}
			}
			System.out.println("intersection size: "+intersection.size());
		}
		if(allInfinite) {
			System.out.println("Cannot evaluate precision, no thresholds available.");
			return null;
		}

		for(int i=0; i<n; i++) {
			Property p = props.get(i);
			if(p.isNoisy()) {
				for(Couple c : intersection) {
					Resource s = c.getSource();
					Resource t = c.getTarget();
					double d = props.get(i).getFilter().getDistance(s.getPropertyValue(p.getName()), t.getPropertyValue(p.getName()));
					c.setDistance(d, p.getIndex());
				}
			}
		}

		double negIn = 0;
		for(Couple c : intersection)
			if(!askOracle(c.toString())) {
				if(!m.classify(c))
					tn++;
				else {
					fp++;
				}
				negIn++;
			}
		
		double negOut = sources.size() * targets.size() - mapping.size() - negIn;
		
		tn = tn + negOut;
		
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}
	
	private double computeMonteCarlo(int j) {
		double min = 1;
		for(int a=0; a<10000; a++) {
			double[] x = new double[m.getN()];
			for(int i=0; i<x.length; i++)
				x[i] = Math.random();
			if(m.classify(x)) {
				if(x[j] < min)
					min = x[j];
			}
		}
		min = (int)(min*10) / 10.0;
		// XXX 0.01 is arbitrary... 
		if(min < 0.01) {
			min = Double.NEGATIVE_INFINITY;
			props.get(j).setNoisy(true);
		} else {
			props.get(j).setNoisy(false);
		}
		System.out.println("MC method for "+j+" = "+min);
		return min;
	}

	private void merge(ArrayList<Couple> intersection, ArrayList<Couple> join, int index) {
	    Iterator<Couple> e = intersection.iterator();
	    while (e.hasNext()) {
	    	Couple c = e.next();
	        if (!join.contains(c))
		        e.remove();
	        else {
	        	for(Couple cj : join)
	        		if(cj.equals(c)) {
	        			c.setDistance(cj.getDistanceAt(index), index);
	        			break;
	        		}
	        }	
	    }
	}


	private boolean askOracle(Couple c) {
    	boolean b = askOracle(c.getID());
    	if(b)
    		c.setPositive(true);
    	else
    		c.setPositive(false);
		return b;
	}

	private boolean askOracle(String ids) {
		return Test.askOracle(ids); // TODO remove me & add interaction
	}
	
	/**
	 * Selects and trains <i>k</i> couples randomly from the joint of the primary and secondary sets.
	 * @param primary Set of couples, usually the set of positive or negative examples.
	 * @param secondary Set of couples, usually the complementary of primary.
	 * @param labelled Set of labeled couples so far.
	 * @param poslbl Set of positive-labeled couples.
	 * @param neglbl Set of negative-labeled couples.
	 * @param size Query size.
	 */
	@SuppressWarnings("unused")
	private void processRandomTraining(ArrayList<Couple> primary, ArrayList<Couple> secondary, 
			ArrayList<Couple> labelled, ArrayList<Couple> poslbl, ArrayList<Couple> neglbl, int size) {
		
		ArrayList<Couple> couples = new ArrayList<Couple>(primary);
		couples.addAll(secondary);
		
		ArrayList<Couple> temp = new ArrayList<Couple>();
		for(int i=0; temp.size() < size && i < couples.size(); i++) {
			Couple c = couples.get((int) (couples.size() * Math.random()));
			if(!labelled.contains(c)) {
				temp.add(c);
				if(c.isPositive()) // TODO Here use askOracle.
					poslbl.add(c);
				else
					neglbl.add(c);
			}
		}
		labelled.addAll(temp);
		// needed to prevent looping if called again.
		primary.removeAll(temp);
		secondary.removeAll(temp);
	}
	
	/**
	 * Selects and trains <i>size</i> couples from the primary set ordered by their gamma values. Ascending or descending order depends on nearest.
	 * @param primary Set of couples, usually the set of positive or negative examples.
	 * @param secondary Set of couples, usually the complementary of primary.
	 * @param labelled Set of labeled couples so far.
	 * @param poslbl Set of positive-labeled couples.
	 * @param neglbl Set of negative-labeled couples.
	 * @param size Query size.
	 * @param nearest Select the nearest couples (points) to the classifier.
	 */
	private void processTraining(ArrayList<Couple> primary, ArrayList<Couple> secondary, 
			ArrayList<Couple> labelled, ArrayList<Couple> poslbl, ArrayList<Couple> neglbl, int size, boolean nearest) {
		
		ArrayList<Couple> temp = new ArrayList<Couple>();
		for(int i=0; temp.size() < size && i < primary.size(); i++) {
			Couple c = nearest ? primary.get(i) : primary.get(primary.size()-i-1);
			if(!labelled.contains(c)) {
				temp.add(c);
				if(c.isPositive()) // TODO Here use askOracle.
					poslbl.add(c);
				else
					neglbl.add(c);
			}
		}
		labelled.addAll(temp);
		// needed to prevent looping if called again.
		primary.removeAll(temp);
		secondary.removeAll(temp);
	}
	
	/**
	 * Adds one fake positive ([1, ..., 1]) and one fake negative example ([0, ..., 0]) to the corresponding sets.
	 * This gives semantic orientation to the classifier and prevents SVM from failing.
	 * @param poslbl
	 * @param neglbl
	 * @param labelled
	 */
	private void orientate(ArrayList<Couple> poslbl, ArrayList<Couple> neglbl,
			ArrayList<Couple> labelled) {
		Couple c1 = new Couple(new Resource("1"), new Resource("1"));
		Couple c2 = new Couple(new Resource("0"), new Resource("0"));
		for(Property p : props) {
			c1.setDistance(1.0, p.getIndex());
			c2.setDistance(0.0, p.getIndex());
		}
		c1.setPositive(true);
		c2.setPositive(false);
		poslbl.add(c1);
		neglbl.add(c2);
		labelled.add(c1);
		labelled.add(c2);
	}

	/**
	 * Initializes the properties checking their data types. Eventually calls weights and extrema computation.
	 */
	private void initialization() {
		ArrayList<String> propertyNames;
		try {
			propertyNames = sources.get(0).getPropertyNames();
		} catch (Exception e) {
			System.err.println("Source set is empty!");
			return;
		}
		
		for(int i=0; i<propertyNames.size(); i++) {
			String pn = propertyNames.get(i);
			int type = Property.TYPE_NUMERIC;
			for(Resource s : sources) {
				if(s.checkDatatype(pn) == Property.TYPE_STRING) {
					type = Property.TYPE_STRING;
					break;
				}
			}
			Property p = new Property(pn, type, i);
			props.add(p);
			if(tfidf)
				p.getFilter().init(sources, targets);
		}
		
		for(Property p : props) {
			System.out.println(p.getName()+"\t"+p.getDatatypeAsString());
			if(p.getDatatype() == Property.TYPE_NUMERIC)
				((MahalaFilter) p.getFilter()).computeExtrema(sources, targets);
		}
		
		m.setN(propertyNames.size());
		m.initW();
	}


    
}
