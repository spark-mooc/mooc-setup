package acids2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.TreeSet;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;			
import libsvm.svm_problem;
import utility.GammaComparator;
import utility.SystemOutHandler;
import acids2.output.Fscore;
import acids2.output.OctaveScriptCreator;
import acids2.output.RawDataScriptCreator;
import acids2.output.SageScriptCreator;
import acids2.plot.Svm3D;
import acids2.test.Test;
import filters.StandardFilter;
import filters.WeightedNGramFilter;
import filters.mahalanobis.MahalaFilter;
import filters.reeding.CrowFilter;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class RandomSetting extends TestUnit {
		
	// SVM parameters
	private svm_model model;
	private svm_problem problem;
	private int KERNEL = svm_parameter.POLY;
	private final int DEGREE = 2;
	private final int COEF0 = 0;
	private final double GAMMA = 1;
	
	// classifier properties
	private double[][] sv_d;
	private double[] w;
	private double[] w_linear;
	private double theta0;
	private double theta;
//	private double theta_plot;
	private double beta;
	private int n;
	/*
	 * TODO create Settings class
	 */
	
	// total number of queries
	private int max_queries;
	// maximum number of iterations for perceptron
	private int max_iter_perceptron;
	// queries per iteration per class (pos|neg)
	private int k = 5;

	// number of random examples
	private int H;

	// support variables
	private int query_counter = 0;
	private svm_node[][] orig_x; 

	private ArrayList<Property> props = new ArrayList<Property>();
    
    public RandomSetting(ArrayList<Resource> sources, ArrayList<Resource> targets, int _queries, double _beta,
    		int _mip, int _H, boolean tfidf) {
		//  {
    	
		long t0 = System.currentTimeMillis(); 
				
    	max_queries = _queries;
    	max_iter_perceptron = _mip;
    	H = _H;
    	
    	props.clear();
    	
		// initialization
		
		ArrayList<String> propertyNames;
		try {
			propertyNames = sources.get(0).getPropertyNames();
		} catch (Exception e) {
			System.err.println("Source set is empty!");
			return;
		}
		
		int index = 0;
		for(String pn : propertyNames) {
			int cnt = 0, type = Property.TYPE_NUMERIC;
			for(Resource s : sources) {
				if(s.checkDatatype(pn) == Property.TYPE_STRING) {
					type = Property.TYPE_STRING;
					break;
				}
				cnt++;
				if(cnt == 10) {
					type = Property.TYPE_NUMERIC;
					break;
				}
			}
			Property p = new Property(pn, type, index);
			props.add(p);
			if(tfidf)
				p.getFilter().init(sources, targets);
			
			index++;
		}
		
		for(Property p : props) {
			System.out.println(p.getName()+"\t"+p.getDatatypeAsString());
			if(p.getDatatype() == Property.TYPE_NUMERIC)
				((MahalaFilter) p.getFilter()).computeExtrema(sources, targets);
		}
		
		n = propertyNames.size();
		w = new double[n];
		for(int i=0; i<n; i++)
			w[i] = 1.0;
		theta0 = (double) n - 0.5;
		theta = theta0;
		
		long prTime = System.currentTimeMillis() - t0, alTime = 0, mlTime = 0;
		
		// algorithm
		
		ArrayList<Couple> labelled = new ArrayList<Couple>();

		for(int i_loop=0; query_counter<max_queries || theta == -1.0; i_loop++) { // huge loop
			
			System.out.println("\nGrand loop: iteration #"+(i_loop+1));
			
			long pr0 = System.currentTimeMillis();
			
			ArrayList<Couple> intersection = new ArrayList<Couple>();
			beta = _beta;
			
			max_queries = H;
			labelled.clear();
			query_counter = 0;
			
			while(intersection.size() < H/2) {
				Resource s = sources.get((int) (sources.size() * Math.random()));
				Resource t = targets.get((int) (targets.size() * Math.random()));
				Couple c = new Couple(s, t);
				if(intersection.contains(c))
					continue;
				intersection.add(c);
			}
			
			while(intersection.size() < H) {
				String oa = Test.getOraclesAnswers().get((int) (Test.getOraclesAnswers().size() * Math.random()));
				String[] oas = oa.split("#");
				String src = oas[0], tgt = oas[1];
				Resource s = null, t = null;
				for(Resource so : sources)
					if(so.getID().equals(src)) {
						s = so;
						break;
					}
				for(Resource ta : targets)
					if(ta.getID().equals(tgt)) {
						t = ta;
						break;
					}
				Couple c = new Couple(s, t);
				if(intersection.contains(c))
					continue;
				c.setPositive(true);
				intersection.add(c);
			}
			
			query_counter = H;
			labelled.addAll(intersection);
			
			for(Property p : props)
				p.setNoisy(true);
			
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
			
			long al0 = System.currentTimeMillis();
			prTime += al0 - pr0;
			
			// active learning phase
			ArrayList<Couple> poslbl = new ArrayList<Couple>();
			ArrayList<Couple> neglbl = new ArrayList<Couple>();
			
			for(Couple c : labelled)
				if(c.isPositive())
					poslbl.add(c);
				else
					neglbl.add(c);
			
			System.out.println("Labeled pos: "+poslbl.size());
			System.out.println("Labeled neg: "+neglbl.size());
			
			labelled.clear();
			labelled.addAll(poslbl);
			labelled.addAll(neglbl);
			
			System.out.println("Pos Labeled:");
			for(Couple c : poslbl) {
				for(double d : c.getDistances())
					System.out.print(d+", ");
				System.out.println("\t"+c);
			}
			System.out.println("Neg Labeled:");
			for(Couple c : neglbl) {
				for(double d : c.getDistances())
					System.out.print(d+", ");
				System.out.println("\t"+c);
			}
	
			System.out.println("Labeled pos: "+poslbl.size());
			System.out.println("Labeled neg: "+neglbl.size());
	        System.out.println("Questions submitted: "+query_counter);
			
			long ml0 = System.currentTimeMillis();
			alTime += ml0 - al0;
	        
			traceSvm(poslbl, neglbl);
				        
	        mlTime += System.currentTimeMillis() - ml0;
	        
		}
        
        for(Property p : props)
        	if(p.getDatatype() == Property.TYPE_STRING)
        		System.out.println("WEIGHTS ("+p.getName()+"): "+p.getFilter().getWeights());
        
		System.out.println("== EXECUTION TIME (seconds) ==");
		System.out.println("Preparation     \t" + prTime/1000.0);
		System.out.println("Active learning \t" + alTime/1000.0);
		System.out.println("SVM + Perceptron\t" + mlTime/1000.0);
		System.out.println("Total exec. time\t" + (prTime+alTime+mlTime)/1000.0);

		// testing...
		
		System.out.println("");

		fs = fastEvaluation(sources, targets);
		if(fs == null)
			fs = evaluation(sources, targets);
		
//			subsetEvaluation(sources, targets);
		
		
//		try {
////			new OctaveScriptCreator().create(sources, targets, props, w_linear, theta);
////			new SageScriptCreator().create(sources, targets, props, w_linear, theta);
//			RawDataScriptCreator rd = new RawDataScriptCreator();
//			rd.setTrainingset(labelled);
//			rd.create(sources, targets, props, w_linear, _beta);
//		} catch (IOException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
		
//		try {
//			Svm3D.draw(model, problem, theta_plot, sv_d, theta0);
//		} catch (ArrayIndexOutOfBoundsException e) {
//			System.out.println("3D: no plot (n < 3).");
//		}
		
	}

	private Fscore fastEvaluation(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		
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
			if(classify(c))
				tp++;
			else
				fn++;
		}

		ArrayList<Couple> intersection = new ArrayList<Couple>();
		beta = 0.1;
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
				if(!classify(c))
					tn++;
				else {
					fp++;
				}
				negIn++;
			}
		
		double negOut = sources.size() * targets.size() - mapping.size() - negIn;
		
		tn = tn + negOut;
		
		System.out.println();
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}

	private void updateSimilarities(TreeSet<Couple> fpC,
			TreeSet<Couple> fnC) {
		for(Property p : props) {
			if(p.getDatatype() == Property.TYPE_STRING) {
				StandardFilter filter = p.getFilter();
				String pname = p.getName();
				int index = p.getIndex();
				for(Couple c : fpC)
					c.setDistance(filter.getDistance(c.getSource().getPropertyValue(pname),
							c.getTarget().getPropertyValue(pname)), index);
				for(Couple c : fnC)
					c.setDistance(filter.getDistance(c.getSource().getPropertyValue(pname),
							c.getTarget().getPropertyValue(pname)), index);
			}
		}
	}

	private void updateWeights(TreeSet<Couple> fpC, TreeSet<Couple> fnC) {
		for(Property p : props) {
			if(p.getDatatype() == Property.TYPE_STRING) {
				// TODO find alternative to casting: method .prepare() in StandardFilter
				WeightedNGramFilter rf = (WeightedNGramFilter)(p.getFilter());
				String pname = p.getName();
				for(Couple c : fpC)
					rf.prepareNgCache(c.getSource().getPropertyValue(pname), 
							c.getTarget().getPropertyValue(pname), false, 3);
				for(Couple c : fnC)
					rf.prepareNgCache(c.getSource().getPropertyValue(pname), 
							c.getTarget().getPropertyValue(pname), true, 3);
				rf.updateWeights();
			}
		}
	}

	private boolean perceptronEndCondition(boolean cl, int i_perc, double f1) {
		return !cl || i_perc >= max_iter_perceptron || f1 == 1.0;
	}

    private double computeThreshold(int j) {
        // Default linear classifier...
        if(model == null) {
			double sum = 0;
			for(int i=0; i<n; i++)
				if(i != j)
					sum += w[i];
			double d = (theta - sum) / w[j] - beta;
			if(d <= 0)
				props.get(j).setNoisy(true);
			else
				props.get(j).setNoisy(false);
			return d;
        }
        // Polynomial homogeneous II-degree classifiers...
        // if weight is not positive, property is noisy (no lower bound)
        if(w_linear[j] <= 0) {
                props.get(j).setNoisy(true);
                return 0.0;
        }
        ArrayList<String> suffix = buildPhiSuffix();
        double r2 = Math.sqrt(2);
        // optimal values for other coordinates
        double[] x = new double[n];
        for(int i=0; i<n; i++)
//              if(w_linear[i] > 0)
                        x[i] = 1;
//              else
//                      x[i] = 0;
        // calculate b, c
        double b = 0.0, c = 0.0;
        for(int i=0; i<suffix.size(); i++) {
                String[] s = suffix.get(i).split(",");
                if(s[0].equals(""+j))
                        b += w_linear[n + i] * x[Integer.parseInt(s[1])];
                else if(s[1].equals(""+j))
                        b += w_linear[n + i] * x[Integer.parseInt(s[0])];
                else
                        c += w_linear[n + i] * x[Integer.parseInt(s[0])] * x[Integer.parseInt(s[1])];
        }
        b = b * r2 / w_linear[j];
        c = c * r2;
        // quadratic terms
        for(int i=0; i<n; i++)
                if(i != j)
                        c += w_linear[i]; // * Math.pow(x[i], 2);
        // is "theta-beta" right for all theta?
        if(theta > 0)
                c = (c - (theta)) / w_linear[j];
        else
                c = (c - (theta)) / w_linear[j];
        // II-degree equation formula (a=1)
        double x1 = (- b + Math.sqrt(Math.pow(b, 2) - 4 * c)) / 2;
        double x2 = (- b - Math.sqrt(Math.pow(b, 2) - 4 * c)) / 2;
        if(x2 < 0 && 0 < x1 && x1 <= 1) {
                props.get(j).setNoisy(false);
                return x1 - beta;
        }
        props.get(j).setNoisy(true);
        return Double.NEGATIVE_INFINITY;
    }

	private double computeMonteCarlo(int j) {
		if(model == null)
			return computeThreshold(j);
		double min = 1;
		for(int a=0; a<10000; a++) {
			double[] x = new double[n];
			for(int i=0; i<n; i++)
				x[i] = Math.random();
			if(classify(x)) {
				if(x[j] < min)
					min = x[j];
			}
		}
		min = (int)((min-beta)*10) / 10.0;
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


	private Fscore evaluation(ArrayList<Resource> sources, ArrayList<Resource> targets) {
		double tp = 0, tn = 0, fp = 0, fn = 0;
		double[] val = new double[n];
		
		for(Resource s : sources) {
			for(Resource t : targets) {
				for(int j=0; j<props.size(); j++) {
					Property prop = props.get(j);
					String p = prop.getName();
					val[j] = prop.getFilter().getDistance(s.getPropertyValue(p), t.getPropertyValue(p));
				}
				if(askOracle(s.getID()+"#"+t.getID())) {
					if(classify(val))
						tp++;
					else
						fn++;
				} else {
					if(classify(val)) {
						fp++;
					} else
						tn++;
				}
			}
		}
		
		System.out.println();
		Fscore f = new Fscore("", tp, fp, tn, fn);
		f.print();
		return f;
	}
	
	private double getFScore(double tp, double fp, double tn, double fn) {
        double pre = tp+fp != 0 ? tp / (tp + fp) : 0;
        double rec = tp+fn != 0 ? tp / (tp + fn) : 0;
        double f1 = pre+rec != 0 ? 2 * pre * rec / (pre + rec) : 0;
        System.out.println("pre = "+pre+", rec = "+rec);
        System.out.println("f1 = "+f1+" (tp="+tp+", fp="+fp+", tn="+tn+", fn="+fn+")");
        return f1;
	}


	private void subsetEvaluation(TreeSet<Resource> sources, TreeSet<Resource> targets) {
//		double tp = tp0, tn = tn0, fp = fp0, fn = fn0;
		double tp = 0, tn = 0, fp = 0, fn = 0;
		
		System.out.print("\nEvaluation esteem");
		
		final int BREAK_AT = 100000;
        svm_node[][] x2 = new svm_node[problem.l+BREAK_AT][n];
        for(int i=0; i<problem.x.length; i++)
        	for(int j=0; j<problem.x[i].length; j++)
        		x2[i][j] = problem.x[i][j];
        double[] y2 = new double[problem.l+BREAK_AT];
        for(int i=0; i<problem.y.length; i++)
    		y2[i] = problem.y[i];
        
        ArrayList<Resource> src = new ArrayList<Resource>(sources);
        ArrayList<Resource> tgt = new ArrayList<Resource>(targets);
        ArrayList<String> ids = new ArrayList<String>();
        
        for(int i=0; i<BREAK_AT; i++) {
        	Resource s = src.get( (int)(src.size()*Math.random()) );
        	Resource t = tgt.get( (int)(tgt.size()*Math.random()) );
        	if(ids.contains(s.getID()+"#"+t.getID())) {
        		i--;
        	} else {
				double val;
				for(int j=0; j<props.size(); j++) {
					Property prop = props.get(j);
					String p = prop.getName();
					val = prop.getFilter().getDistance(s.getPropertyValue(p), t.getPropertyValue(p));
	                x2[problem.l+i][j] = new svm_node();
	                x2[problem.l+i][j].index = j;
                	x2[problem.l+i][j].value = val;
				}
				ids.add(s.getID()+"#"+t.getID());
				if(i % 1000 == 0)
					System.out.print(".");
        	}
		}
		System.out.println("");
        
		// recover old values
		for(int j=0; j<props.size(); j++)
			if(props.get(j).getDatatype() == Property.TYPE_NUMERIC)
		        for(int i=0; i<orig_x.length; i++)
		    		x2[i][j].value = orig_x[i][j].value;
		
		// predicts the test set
		for(int i=0; i<ids.size(); i++) {
			if(askOracle(ids.get(i))) {
				if(classify(x2[problem.l+i])) tp++;
				else { fn++; System.out.println("FN: "+ids.get(i)); }
				y2[problem.l+i] = 1;
			} else {
				if(classify(x2[problem.l+i])) { fp++; System.out.println("FP: "+ids.get(i)); }
				else tn++;
				y2[problem.l+i] = -1;
			}
		}
        
		problem.x = x2;
		problem.y = y2;
		problem.l += ids.size();
		
		getFScore(tp, fp, tn, fn);
	}


	private boolean blockingEndCondition(int intersSize, int cartesianP) {
		return intersSize >= 2*k+query_counter || cartesianP < max_queries+query_counter;
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

	private boolean classify(Couple c) {
		if(model == null) {
			// Default classifier is always set to linear.
			double sum = 0.0;
			ArrayList<Double> dist = c.getDistances();
			for(int i=0; i<dist.size(); i++)
				sum += dist.get(i) * w[i];
			return sum >= theta; 
		}
        svm_node[] node = new svm_node[n];
        for(int i=0; i<n; i++) {
        	node[i] = new svm_node();
        	node[i].index = i;
        	node[i].value = c.getDistanceAt(i);
        }
        return classify(node);
	}

	private boolean classify(double[] val) {
        svm_node[] node = new svm_node[n];
        for(int i=0; i<n; i++) {
        	node[i] = new svm_node();
        	node[i].index = i;
        	node[i].value = val[i];
        }
        return classify(node);
	}
	
	private boolean classify(svm_node[] node) {
        if(svm.svm_predict(model, node) == 1.0)
        	return true;
        else return false;
	}
	

	// gamma = distance from classifier
    private double computeGamma(ArrayList<Double> dist) {
		if(model == null) {
			// Default classifier...
	        double numer = 0.0, denom = 0.0;
	        for(int i=0; i<dist.size(); i++) {
	            numer += dist.get(i) * w[i];
	            denom += Math.pow(w[i], 2);
	        }
	        numer -= theta;
	        return Math.abs(numer/Math.sqrt(denom));
		}
		// All classifiers...
		double[] x = new double[dist.size()];
		for(int i=0; i<dist.size(); i++)
			x[i] = dist.get(i);
		double[] phi = phi(x);
		int m = phi.length;
		// calculate Q
		double[] q = new double[m];
		for(int i=0; i<m; i++)
			if(w_linear[i] != 0) {
				q[i] = theta / w_linear[i];
				break;
			}
		// calculate unit vector wu
		double norm = 0;
		for(int i=0; i<m; i++)
			norm += Math.pow(w_linear[i], 2);
		norm = Math.sqrt(norm);
		double[] wu = new double[n];
		for(int i=0; i<n; i++)
			wu[i] = w_linear[i] / norm;
		// calculate t = phi(X - Q) . wu
		double[] phixq = new double[m];
		for(int i=0; i<m; i++)
			phixq[i] = phi[i] - q[i];
		double t = 0;
		for(int i=0; i<n; i++)
			t += phixq[i] * wu[i];
		// calculate P'
		double[] p1 = new double[n];
		for(int i=0; i<n; i++)
			p1[i] = phi[i] - t * wu[i];
		// calculate segment XP
		double sum = 0;
		for(int i=0; i<n; i++)
			if(p1[i] >= 0)
				sum += Math.pow(dist.get(i) - phiInverse(p1[i]), 2);
			else // anti-transformation is out of the similarity space
				sum += Math.pow(dist.get(i), 2);
		sum = Math.sqrt(sum);
		if(Double.isNaN(sum))
			System.out.println();
		return sum;
    }
    
    private ArrayList<String> buildPhiSuffix() {
    	ArrayList<String> suffix = new ArrayList<String>();
    	for(int i=0; i<n-1; i++)
    		for(int j=i+1; j<n; j++)
    			if(i != j)
    				suffix.add(i+","+j);
    	return suffix;
    }
    
    private double[] phi(double[] x) {
    	// assuming DEGREE = 2 and COEF0 = 0
    	int n1 = (int) (Math.pow(n, 2) + n) / 2;
    	double[] phi = new double[n1];
    	for(int i=0; i<n; i++)
    		phi[i] = Math.pow(x[i], 2);
    	double r2 = Math.sqrt(2);
    	int p = n;
    	for(int i=0; i<n-1; i++)
    		for(int j=i+1; j<n; j++) {
        		phi[p] = r2 * x[i] * x[j];
        		p++;
    		}
    	return phi;
    }

    private double phiInverse(double x) {
		return Math.sqrt(x);
	}

	public boolean askOracle(Couple c) {
    	query_counter++;
    	String ids = c.getSource().getID()+"#"+c.getTarget().getID();
    	boolean b = askOracle(ids);
    	if(b)
    		c.setPositive(true);
    	else
    		c.setPositive(false);
		return b;
	}

    private boolean askOracle(String ids) {
		return Test.askOracle(ids); // TODO remove me & add interaction
	}


    private boolean traceSvm(ArrayList<Couple> poslbl, ArrayList<Couple> neglbl) {
    	
        int size = poslbl.size() + neglbl.size();

        // build x,y vectors
        svm_node[][] x = new svm_node[size][n];
        double[] y = new double[size];
        for(int i=0; i<poslbl.size(); i++) {
            ArrayList<Double> arr = poslbl.get(i).getDistances();
            for(int j=0; j<arr.size(); j++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = arr.get(j);
            }
            y[i] = 1;
        }
        for(int i=poslbl.size(); i<size; i++) {
            ArrayList<Double> arr = neglbl.get(i-poslbl.size()).getDistances();
            for(int j=0; j<arr.size(); j++) {
                x[i][j] = new svm_node();
                x[i][j].index = j;
                x[i][j].value = arr.get(j);
            }
            y[i] = -1;
        }
        
        // configure model
        // POLY: (gamma*u'*v + coef0)^degree
        problem = new svm_problem();
        problem.l = size;
        problem.x = x;
        problem.y = y;
        orig_x = x.clone();
        svm_parameter parameter = new svm_parameter();
        parameter.C = 1E+4;
        parameter.svm_type = svm_parameter.C_SVC;
        parameter.kernel_type = KERNEL;
        if(KERNEL == svm_parameter.POLY) {
			parameter.degree = DEGREE; // default: 3
			parameter.coef0  = COEF0; // default: 0
			parameter.gamma  = GAMMA; // default: 1/n
        } 
        parameter.eps = 1E-4;
        model = svm.svm_train(problem, parameter);
        // sv = ( nSV ; n )
        svm_node[][] sv = model.SV;
        // no support vectors
        if(sv.length == 0)
        	return false;
        // sv_coef = ( 1 ; nSV )
        double[][] sv_coef = model.sv_coef;
        
        sv_d = new double[sv.length][n];
        for(int j=0; j<sv.length; j++)
            for(int i=0; i<sv[j].length; i++)
            	sv_d[j][i] = sv[j][i].value;
        
        // w = sv' * sv_coef' = (sv_coef * sv)' = ( n ; 1 )
        double[][] phis = new double[sv.length][];
        for(int i=0; i<phis.length; i++)
        	phis[i] = phi(sv_d[i]);
        
        w_linear = new double[phis[0].length];
        
        for(int i=0; i<phis.length; i++)
        	for(int j=0; j<phis[i].length; j++)
        		w_linear[j] += sv_coef[0][i] * phis[i][j];
        
        int signum = (model.label[0] == -1.0) ? -1 : 1;
        theta = signum * model.rho[0];
        
        // theta is normally at the first member in the classification inequality
//        theta_plot = -theta;

    	for(int i=0; i<w_linear.length; i++) {
    		w_linear[i] = signum * w_linear[i] / Math.abs(theta);
    		System.out.println("w_linear["+i+"] = "+w_linear[i]);
    	}
		theta = theta / Math.abs(theta);

        System.out.println("theta = "+theta);
        
        return true;
    }

}
