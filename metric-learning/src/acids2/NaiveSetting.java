package acids2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Scanner;

import libsvm.svm_parameter;

import utility.GammaComparator;
import acids2.classifiers.svm.SvmHandler;
import acids2.output.CoupleStore;
import acids2.output.Fscore;
import acids2.output.RawDataScriptCreator;
import acids2.test.Test;
import filters.mahalanobis.MahalaFilter;
import filters.reeding.CrowFilter;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class NaiveSetting extends TestUnit {

	private ArrayList<Resource> sources, targets;
	private int k;
	private boolean tfidf;

	private ArrayList<Property> props = new ArrayList<Property>();
	private SvmHandler m; 
	
	public NaiveSetting(ArrayList<Resource> sources, ArrayList<Resource> targets, int k, boolean tfidf, String dataset) {
		
		this.sources = sources;
		this.targets = targets;
		this.k = k;
		this.tfidf = tfidf;
		
		m = new SvmHandler(svm_parameter.POLY);
		
		initialization();
		
		ArrayList<Couple> couples = new ArrayList<Couple>();
		int mapSize = Math.min(sources.size(), targets.size());
		
//		for(double threshold=0.4; couples.size() < mapSize; threshold-=0.1) {
//			System.out.println("threshold = "+threshold);
//			for(int i=0; i<props.size(); i++) {
//				Property p = props.get(i);
//				if(i == 0)
//					couples = p.getFilter().filter(sources,  targets, p.getName(), threshold);
//				else
//					merge(couples, p.getFilter().filter(couples, p.getName(), threshold), i);
//			}
//		}
		
		int[] ks = {5, 10, 20, 50, 100};

		try {
			for(double thr=0.4; couples.size() < mapSize; thr-=0.1)
				couples = CoupleStore.filterCouplesFromCartesian(dataset, thr);
			CoupleStore.saveFilteredCouples(couples, dataset);
		} catch (IOException e2) {
			e2.printStackTrace();
		}

		m.initTheta(couples, mapSize);
		double theta = m.getTheta();

		for(int i=0; i<ks.length; i++) {
			this.k = ks[i];
			System.out.println("\n#### k = "+this.k+" ####");
			
//			try {
//				couples = CoupleStore.loadFilteredCouples(dataset);
//			} catch (IOException e1) {
//				e1.printStackTrace();
//			} finally {
//				System.out.println("Done.");
//			}

			m = new SvmHandler(svm_parameter.POLY);
			m.setN(props.size());
			m.initW();
			m.setTheta(theta);

			ArrayList<Couple> labelled = new ArrayList<Couple>();
			
			ArrayList<Couple> posInformative = new ArrayList<Couple>();
			ArrayList<Couple> negInformative = new ArrayList<Couple>();
			
			for(Couple c : couples) {
		        c.setGamma( computeGammaLinear(c) );
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
			
			// train with k most likely positive examples
			processTraining(posInformative, negInformative, labelled, poslbl, neglbl, false);
			// train with k most informative positive examples
			processTraining(posInformative, negInformative, labelled, poslbl, neglbl, true);
			// train with k most informative negative examples
			processTraining(negInformative, posInformative, labelled, poslbl, neglbl, true);
							
			System.out.println("Labeled pos: "+poslbl.size());
			System.out.println("Labeled neg: "+neglbl.size());
			
			m.trace(poslbl, neglbl);
			evaluateOn(labelled);
			
			RawDataScriptCreator raw = new RawDataScriptCreator(dataset+"_"+this.k, false, false, true);
			try {
				raw.create(sources, targets, props, m.getWLinear(), m.getTheta());
				new Evaluator("output/"+dataset+"_testset.txt", "output/"+dataset+"_"+this.k+"_classifier.txt");
			} catch (Exception e) {
				e.printStackTrace();
			}
			
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
//			if(askOracle(c)) {
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

	private double computeGammaLinear(Couple c) {
		double[] w = m.getWLinear();
		ArrayList<Double> dist = c.getDistances();
		double numer = 0.0, denom = 0.0;
		for(int i=0; i<dist.size(); i++) {
			numer += dist.get(i) * w[i];
			denom += Math.pow(w[i], 2);
		}
		numer -= m.getTheta();
		return Math.abs(numer/Math.sqrt(denom));
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
	 */
	@SuppressWarnings("unused")
	private void processRandomTraining(ArrayList<Couple> primary, ArrayList<Couple> secondary, 
			ArrayList<Couple> labelled, ArrayList<Couple> poslbl, ArrayList<Couple> neglbl) {
		ArrayList<Couple> couples = new ArrayList<Couple>(primary);
		couples.addAll(secondary);
		
		ArrayList<Couple> temp = new ArrayList<Couple>();
		for(int i=0; temp.size() < k && i != couples.size(); i++) {
			Couple c = couples.get((int) (couples.size() * Math.random()));
			if(!labelled.contains(c)) {
				temp.add(c);
				if(askOracle(c))
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
	 * Selects and trains <i>k</i> couples from the primary set ordered by their gamma values. Ascending or descending order depends on nearest.
	 * @param primary Set of couples, usually the set of positive or negative examples.
	 * @param secondary Set of couples, usually the complementary of primary.
	 * @param labelled Set of labeled couples so far.
	 * @param poslbl Set of positive-labeled couples.
	 * @param neglbl Set of negative-labeled couples.
	 * @param nearest Selects the nearest couples (points) to the classifier.
	 */
	private void processTraining(ArrayList<Couple> primary, ArrayList<Couple> secondary, 
			ArrayList<Couple> labelled, ArrayList<Couple> poslbl, ArrayList<Couple> neglbl, boolean nearest) {
		
		ArrayList<Couple> temp = new ArrayList<Couple>();
		for(int i=0; temp.size() < k && i != primary.size(); i++) {
			Couple c = nearest ? primary.get(i) : primary.get(primary.size()-i-1);
			if(!labelled.contains(c)) {
				temp.add(c);
//				if(askOracle(c))
//					poslbl.add(c);
//				else
//					neglbl.add(c);
				if(c.isPositive())
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
