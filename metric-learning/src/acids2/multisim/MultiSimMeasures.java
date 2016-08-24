package acids2.multisim;

import java.util.ArrayList;
import java.util.Collections;

import libsvm.svm_parameter;
import utility.GammaComparator;
import utility.Statistics;
import acids2.Couple;
import acids2.Resource;
import acids2.classifiers.svm.SvmHandler;

public class MultiSimMeasures {
	
	private ArrayList<MultiSimProperty> props = new ArrayList<MultiSimProperty>();
	
	private SvmHandler svmHandler;
	private MultiSimSetting setting;
	
	public MultiSimMeasures(MultiSimSetting setting) {
		
		super();
		this.setting = setting;
		this.svmHandler = setting.getSvmHandler();
		
		initialize();
		firstClassifier();
	}

	public double[] estimateMissingValues(double[] val) {
		/*
		double[] res = new double[val.length];
		double[] means = this.getMeans();
		for(int i=0; i<val.length; i++) {
			if(Double.isNaN(val[i])) {
				int n = 0;
				double sum = 0;
				for(int j=0; j<val.length; j++) {
					double d2 = val[j];
					if(i != j && !Double.isNaN(d2)) {
						sum += d2 / means[j];
						n++;
					}
				}
				double est = sum * means[i] / n;
				if(est > 1.0)
					est = 1.0;
				res[i] = est;
//				System.out.print(est+" predicted; ");
			} else {
				res[i] = val[i];
			}
		}
//		for(double v : val)
//			System.out.print(v+", ");
//		System.out.println();
		return res;
		*/ 
		for(int i=0; i<val.length; i++)
			if(Double.isNaN(val[i]))
				val[i] = 0.0;
		return val;
	}
	
	public void estimateMissingValues(Couple c) {
/*		// first method (not symmetric)
		ArrayList<MultiSimSimilarity> sims = getAllSimilarities();
		for(MultiSimSimilarity sim : sims) {
			double d = c.getDistanceAt(sim.getIndex());
			if(Double.isNaN(d)) {
				int n = 0;
				double sum = 0;
				for(MultiSimSimilarity sim2 : sims) {
					double d2 = c.getDistanceAt(sim2.getIndex());
					if(sim != sim2 && !Double.isNaN(d2)) {
						sum += d2 / sim2.getStats().getMean();
						n++;
					}
				}
				double est = sum * sim.getStats().getMean() / n;
				if(est > 1.0)
					est = 1.0;
				c.setDistance(est, sim.getIndex());
//				System.out.println(c.getDistances()+" -> "+est);
			}
		}
*/
		for(MultiSimSimilarity sim : getAllSimilarities())
			if(Double.isNaN(c.getDistanceAt(sim.getIndex())))
				c.setDistance(0.0, sim.getIndex());
	}

	public ArrayList<MultiSimSimilarity> getAllSimilarities() {
		ArrayList<MultiSimSimilarity> sims = new ArrayList<MultiSimSimilarity>();
		for(MultiSimProperty p : props)
			sims.addAll(p.getSimilarities());
		return sims;
	}
	
	public int computeN() {
		int dim = 0;
		for(MultiSimProperty p : props)
			dim += p.getSize();
		return dim;
	}
	
	public double computeThreshold(MultiSimSimilarity similarity) {
		double[] w = svmHandler.getWLinear();
		double theta = svmHandler.getTheta();
		int index = similarity.getIndex();
		
		if(w[index] <= 0.0) {
			similarity.setComputed(false);
			return 0.0;
		}
		
		double sum = 0;
		for(int i=0; i<w.length; i++)
			if(i != index && w[i] > 0)
				sum += w[i];
		double d = (theta - sum) / w[index];
		
		if(d <= 0) {
			similarity.setComputed(false);
			return 0.0;
		}
		
		similarity.setComputed(true);
		return d;
	}
	
	/**
	 * Initializes the properties checking their data types. Eventually calls weights and extrema computation.
	 */
	private void initialize() {
		ArrayList<Resource> sources = setting.getSources();
		ArrayList<Resource> targets = setting.getTargets();
		
		ArrayList<String> propertyNames;		
		try {
			propertyNames = sources.get(0).getPropertyNames();
		} catch (Exception e) {
			System.err.println("Source set is empty!");
			return;
		}
		
		int index = 0;
		for(String pn : propertyNames) {
			int type = MultiSimDatatype.TYPE_NUMERIC;
			for(Resource s : sources) {
				if(s.checkDatatype(pn) == MultiSimDatatype.TYPE_STRING) {
					type = MultiSimDatatype.TYPE_STRING;
					break;
				}
			}
			if(type == MultiSimDatatype.TYPE_NUMERIC) {
				for(Resource t : targets) {
					if(t.checkDatatype(pn) == MultiSimDatatype.TYPE_STRING) {
						type = MultiSimDatatype.TYPE_STRING;
						break;
					}
				}
			}
			MultiSimProperty p = new MultiSimProperty(pn, type, index, this);
			props.add(p);
			index += p.getSize();
		}
		
		for(MultiSimProperty p : props) {
			for(MultiSimSimilarity sim : p.getSimilarities()) {
				System.out.println(p.getName() + "\t" + sim.getName() + "\t" + MultiSimDatatype.asString(sim.getDatatype()));
			}
		}
		
		svmHandler.setN(this.getAllSimilarities().size());
//		svmHandler.initW();
	}
	
	/**
	 * Finds first classifier.
	 */
	private void firstClassifier() {
		ArrayList<Resource> sources = setting.getSources();
		ArrayList<Resource> targets = setting.getTargets();
		
		ArrayList<Couple> couples = new ArrayList<Couple>();
		ArrayList<MultiSimSimilarity> sims = this.getAllSimilarities();
		SvmHandler support = new SvmHandler(svm_parameter.LINEAR);
		
		// number of negatives for each positive
		int rate = Math.max(sources.size(), targets.size());
		
		// choosing 10 virtually positive examples
		int nvp = 10;
		int ntot = rate * nvp;
		
		System.out.print("Computing first classifier");
		double[][] simArray = new double[sims.size()][ntot];
		bigloop: for(int i=0; i<ntot; i++) {
			Resource s = sources.get((int) (sources.size()*Math.random()));
			Resource t = targets.get((int) (targets.size()*Math.random()));
			Couple c = new Couple(s, t);
			
			for(MultiSimSimilarity sim : sims) {
				double d = sim.getSimilarity(s.getPropertyValue(sim.getProperty().getName()), 
						t.getPropertyValue(sim.getProperty().getName()));
				if(Double.isNaN(d)) {
					i--;
					continue bigloop;
				}
				c.setDistance(d, sim.getIndex());
				simArray[sim.getIndex()][i] = d;
			}

//			System.out.println(c.getDistances());
			
			couples.add(c);
			c.setGamma(support.computeGamma(c, 1.0));

			if(i % rate == 0)
				System.out.print(".");
		}
		System.out.println(" done.");
		Collections.sort(couples, new GammaComparator());
		
		ArrayList<Couple> pos = new ArrayList<Couple>();
		ArrayList<Couple> neg = new ArrayList<Couple>();
		
		for(int i=0; i<couples.size(); i++) {
			Couple c = couples.get(i);
			if(i < nvp) {
				pos.add(c);
				c.setPositive(true);
			} else {
				neg.add(c);
				c.setPositive(false);
			}
		}
		
		System.out.println("\npositives:");
		for(Couple c : pos)
			c.info();
		
		support.setN(sims.size());
		support.trace(pos, neg);
		
		support.evaluateOn(couples);
		
		svmHandler.setWLinear(support.getWLinear());
		svmHandler.setTheta(support.getTheta());
		
		for(int j=0; j<simArray.length; j++) {
			Statistics stat = new Statistics(simArray[j]);
			double perc = stat.getPercentile(1.0 - 1.0 / rate);
			sims.get(j).setEstimatedThreshold(perc);
			sims.get(j).setStats(stat);
			System.out.println("Stats(sim_"+j+") = {"+stat.getMean()+", "+perc+"}");
		}

	}
	
	public ArrayList<MultiSimProperty> getProps() {
		return props;
	}

	public SvmHandler getSvmHandler() {
		return svmHandler;
	}

	public MultiSimSetting getSetting() {
		return setting;
	}

	public double computeMonteCarlo(MultiSimSimilarity sim) {
		int index = sim.getIndex();
		double min = 1;
		// TODO With linear classifiers we could just check [y,0,...,0] to [y,1,...,1] with `y` as random number in position `index`.
		for(int a=0; a<100000; a++) {
			double[] x = new double[svmHandler.getN()];
			for(int i=0; i<x.length; i++)
				x[i] = Math.random();
			if(svmHandler.classify(x)) {
				if(x[index] < min)
					min = x[index];
			}
		}
		min = (int)(min*10) / 10.0;
		// XXX 0.01 is arbitrary...
		sim.setComputed(min >= 0.01 ? true : false);
		System.out.println("MC method for "+index+" = "+min);
		return min;
	}

	public double[] getMeans() {
		ArrayList<MultiSimSimilarity> sims = this.getAllSimilarities();
		double[] means = new double[sims.size()];
		for(int i=0; i<means.length; i++)
			means[i] = sims.get(i).getStats().getMean();
		return means;
	}
	
}
