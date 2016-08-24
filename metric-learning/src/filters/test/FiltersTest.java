package filters.test;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.TreeSet;

import utility.SystemOutHandler;
import acids2.Couple;
import acids2.Property;
import acids2.Resource;
import algorithms.edjoin.EdJoinPlus;
import algorithms.edjoin.Entry;
import au.com.bytecode.opencsv.CSVReader;
import distances.WeightedEditDistanceExtended;
import filters.edjoin.EdJoinFilter;
import filters.passjoin.PassJoin;
import filters.reeded.ReededFilter;
import filters.reeding.CrowFilter;
import filters.reeding.IndexNgFilter;


public class FiltersTest {

	private static ArrayList<Resource> sources = new ArrayList<Resource>();
	private static ArrayList<Resource> targets = new ArrayList<Resource>();
	
	private static String sys_out = "\n";
	private static double THETA_MIN;

    private static void loadKnowledgeBases(String sourcePath, String targetPath) throws IOException {
    	loadKnowledgeBases(sourcePath, targetPath, 0, Integer.MAX_VALUE);
    }
    
    private static void clearKnowledgeBases() {
    	sources.clear();
    	targets.clear();
    	System.gc();
    }
    
    private static void loadKnowledgeBases(String sourcePath, String targetPath, int startOffset, int endOffset) throws IOException {
    	
	    String[] ignoredList = {"id", "ID"};
    	
        CSVReader reader = new CSVReader(new FileReader(sourcePath));
        String [] titles = reader.readNext(); // gets the column titles
        for(int i=0; i<startOffset; i++) // skips start offset
        	reader.readNext();
        String [] nextLine;
        int count = 0;
        while ((nextLine = reader.readNext()) != null) {
            Resource r = new Resource(nextLine[0]);
            for(int i=0; i<nextLine.length; i++)
                if(!isIgnored(titles[i].toLowerCase(), ignoredList)) {
                    if(nextLine[i] != null)
                        r.setPropertyValue(titles[i], nextLine[i]);
                    else
                        r.setPropertyValue(titles[i], "");
                }
            sources.add(r);
            if(++count >= endOffset)
            	break;
        }
        
        reader = new CSVReader(new FileReader(targetPath));
        titles = reader.readNext(); // gets the column titles
        for(int i=0; i<startOffset; i++) // skips offset
        	reader.readNext();
        count = 0;
        while ((nextLine = reader.readNext()) != null) {
            Resource r = new Resource(nextLine[0]);
            for(int i=0; i<nextLine.length; i++)
                if(!isIgnored(titles[i].toLowerCase(), ignoredList)) {
                    if(nextLine[i] != null)
                        r.setPropertyValue(titles[i], nextLine[i]);
                    else
                        r.setPropertyValue(titles[i], "");
                }
            targets.add(r);
            if(++count >= endOffset)
            	break;
        }
        
        reader.close();
    }
    
    private static boolean isIgnored(String title, String[] ignoredList) {
        for(String ign : ignoredList) {
            if(title.equals(ign))
                return true;
        }
        return false;
    }

    @SuppressWarnings("unused")
    private static void testPassJoinThresholds(String dataset, Property p) throws IOException {
		
		System.out.println(sources.size());

		ArrayList<Couple> passjResults = null;
	    
	    PassJoin pj = new PassJoin(p);
	    
		for(int theta=0; theta<=5; theta++) {
	
			long start = System.currentTimeMillis();
			
			passjResults = pj.passJoin(new ArrayList<Resource>(sources), new ArrayList<Resource>(targets),
					p.getName(), theta);
			
			double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
			System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+passjResults.size());
		}
		
	}

    @SuppressWarnings("unused")
	private static void testEDJoinThresholds(String dataset, String propertyName) throws IOException {

        TreeSet<Entry> sTree = new TreeSet<Entry>();
        for(Resource s : sources)
            sTree.add(new Entry(s.getID(), s.getPropertyValue(propertyName)));
        TreeSet<Entry> tTree = new TreeSet<Entry>();
        for(Resource s : targets)
            tTree.add(new Entry(s.getID(), s.getPropertyValue(propertyName)));

		System.out.println(sources.size());
		
		TreeSet<String> edjResults = null;
		
		for(int theta=0; theta<=5; theta++) {
			
			long start = System.currentTimeMillis();

			SystemOutHandler.shutDown();
            edjResults = EdJoinPlus.runOnEntries(0, theta, sTree, tTree);
            SystemOutHandler.turnOn();

    		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
    		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+edjResults.size());
		}
	}

    @SuppressWarnings("unused")
	private static void crossValidation(String dataset, Property p) throws IOException {
		System.out.println("PassJoin");
		ArrayList<Couple> pj = testPassJoinOnce(p, 1);
		
		System.out.println("EDJoin");
		ArrayList<Couple> ej = testEdJoinOnce(p, 1);

		// Cross-validation.
		int i = 0;
		for(Couple c : pj) {
			i++;
			if(!ej.contains( c.getSource().getID()+"#"+c.getTarget().getID() ))
				System.out.println(i+". "+c.getSource().getID()
						+ "\t" + c.getTarget().getID()
						+ "\t" + c.getSource().getPropertyValue(p.getName())
						+ "\t" + c.getTarget().getPropertyValue(p.getName())
						+ "\t" + c.getDistanceAt(0));
		}
		
		// (!) Cannot directly check the other way, because we would like to print the titles
		// and EDJoin returns only the IDs.
		
		for(Couple s : ej) {
			String[] ss = s.toString().split("#");
			Couple c = new Couple(new Resource(ss[0]), new Resource(ss[1]));
			if(!pj.contains(c))
				System.out.println(c.getSource().getID()+"#"+c.getTarget().getID());
		}
	}

	private static ArrayList<Couple> testEdJoinOnce(Property p, double theta) throws IOException {
		
//		System.out.println(sources.size());
		
		ArrayList<Couple> edjResults = null;
		
		EdJoinFilter ed = new EdJoinFilter(p);
		ed.setVerbose(false);
		
		long start = System.currentTimeMillis();
		edjResults = ed.filter(sources, targets, p.getName(), theta);
        long now = System.currentTimeMillis();
		double compTime = (double)(now-start)/1000.0;
		
		System.out.println(theta+"\t"+compTime+"\t"+edjResults.size());
		sys_out += theta+"\t"+compTime+"\t"+edjResults.size()+"\n";
		
		return edjResults;
	}

	private static ArrayList<Couple> testPassJoinOnce(Property p, int theta) throws IOException {

		System.out.println(sources.size());

		ArrayList<Couple> passjResults = null;
	    
		long start = System.currentTimeMillis();
		
		PassJoin pj = new PassJoin(p);
		passjResults = pj.passJoin(new ArrayList<Resource>(sources), new ArrayList<Resource>(targets),
				p.getName(), theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+passjResults.size());
		
		return passjResults;
	}

	private static ArrayList<Couple> testReededFilter(Property p, double theta) throws IOException {

//		System.out.println(sources.size());

		ArrayList<Couple> oafResults = null;
	    
		ReededFilter rf = new ReededFilter(p);
	    rf.setVerbose(false);
	    
	    long start = System.currentTimeMillis();
	    
		oafResults = rf.filter(sources, targets, p.getName(), theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
//		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+oafResults.size());
		System.out.println(compTime+"\t"+oafResults.size());
		sys_out += theta+"\t"+compTime+"\t"+oafResults.size()+"\n";
		
		return oafResults;
	}

	private static ArrayList<Couple> testPassJoin(Property p, double theta) throws IOException {

//		System.out.println(sources.size());

		ArrayList<Couple> passjResults = null;
	    
	    PassJoin pj = new PassJoin(p);
	    pj.setVerbose(false);
	    
	    long start = System.currentTimeMillis();

	    passjResults = pj.passJoin(new ArrayList<Resource>(sources), new ArrayList<Resource>(targets),
				p.getName(), theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
//		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+passjResults.size());
		System.out.print(theta+"\t"+compTime+"\t"+passjResults.size()+"\t");
		sys_out += theta+"\t"+compTime+"\t"+passjResults.size()+"\n";
		
		return passjResults;
	}

    private static double testNewNgFilter(Property p, double theta) {
    	ArrayList<Couple> results = null;
	    
	    CrowFilter ngf = new CrowFilter(p);
	    ngf.setVerbose(false);
	    
	    long start = System.currentTimeMillis();

	    results = ngf.filter(sources, targets, p.getName(), theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
//		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+passjResults.size());
		System.out.println(theta+"\t"+compTime+"\t"+results.size()+"\t");
		sys_out += theta+"\t"+compTime+"\t"+results.size()+"\n";
		
		return compTime;
	}
    
    private static ArrayList<Couple> testIndexNgFilter(Property p, double theta) {
    	ArrayList<Couple> results = null;
	    
	    IndexNgFilter ngf = new IndexNgFilter(p);
	    ngf.setVerbose(false);
	    
	    long start = System.currentTimeMillis();

	    results = ngf.filter(sources, targets, p.getName(), theta);
		
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
//		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+passjResults.size());
		System.out.print(theta+"\t"+compTime+"\t"+results.size()+"\t");
		sys_out += theta+"\t"+compTime+"\t"+results.size()+"\n";
		
		return results;
	}

    @SuppressWarnings("unused")
	private static TreeSet<Couple> testQuadraticJoin(String propertyName, double theta) throws IOException {
		
		System.out.println(sources.size());

	    TreeSet<Couple> quadrResults = new TreeSet<Couple>();
	    
	    long start = System.currentTimeMillis();
		
	    WeightedEditDistanceExtended wed = new WeightedEditDistanceExtended() {
			@Override
			public double transposeWeight(char cFirst, char cSecond) {
				return Double.POSITIVE_INFINITY;
			}
			@Override
			public double substituteWeight(char cDeleted, char cInserted) {
				if((cDeleted >= 'A' && cDeleted <= 'Z' && cDeleted+32 == cInserted) || 
						(cDeleted >= 'a' && cDeleted <= 'z' && cDeleted-32 == cInserted))
					return 0.5;
				else return 1.0;
			}
			@Override
			public double matchWeight(char cMatched) {
				return 0.0;
			}
			@Override
			public double insertWeight(char cInserted) {
				return 1.0;
			}
			@Override
			public double deleteWeight(char cDeleted) {
				switch(cDeleted) {
				case 'i': case 'r': case 's': case 't': return 0.5;
				}
				return 1.0;
			}
		};  
	    
		for(Resource s : sources)
			for(Resource t : targets) {
				double d = wed.proximity(s.getPropertyValue(propertyName), t.getPropertyValue(propertyName));
				if(d <= theta)
					quadrResults.add(new Couple(s,t));
			}
		System.out.println(sources.size()*targets.size());
				
		double compTime = (double)(System.currentTimeMillis()-start)/1000.0;
		System.out.println("theta = "+theta+"\t\tΔt = "+compTime+"\t\t|R| = "+quadrResults.size());
		
		return quadrResults;
	}
	
    private static void notify(String s) {
    	
  		String sysout = "";
		try {
			sysout = URLEncoder.encode(s, "ISO-8859-1");
		} catch (UnsupportedEncodingException e1) {
			e1.printStackTrace();
		}
    	
        try {
            // Create a URL for the desired page
            URL url = new URL("http://mommi84.altervista.org/notifier/index.php?"
                    + "sysout="+sysout);

            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            // Read all the text returned by the server
            BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            in.close();
        } catch (IOException e) {
                e.printStackTrace();
        }
    }

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		
//		String s = "Guest editorial", t = "Guest Editorial";
//		Vector<Character> cs = new Vector<Character>();
//		Vector<Character> ct = new Vector<Character>();
//		for(int i=0; i<s.length(); i++)
//			cs.add(s.charAt(i));
//		for(int i=0; i<t.length(); i++)
//			ct.add(t.charAt(i));
//		Vector<Character> ct2 = new Vector<Character>(ct);
//		for(Character c1 : ct2)
//			if(cs.remove(c1))
//				ct.remove(c1);
//		System.out.println(cs);
//		System.out.println(ct);
//		
//		System.exit(0);
		
		String[] dataset = { 
//				"data/1-dblp-acm/sources.csv",
//				"data/1-dblp-acm/targets.csv",
//				"data/2-dblp-scholar/targets.csv",
//				"data/3-amazon-googleproducts/targets.csv",
//				"data/4-abt-buy/sources.csv",
				"data/5-person1/sources.csv",
//				"data/6-restaurant/sources.csv",
//			"data/8-scalability/persons.csv",
//			"data/8-scalability/places.csv",
//			"data/8-scalability/works.csv",
		};
		String[] pname = {
//				"title",
//				"authors",
//				"title",
//				"name",
//				"name",
				"surname",
//				"name",
//			"name",
//			"name",
//			"name",
		};
		
		final int TOT = 5;
		double avg = 0;
		for(int i=0; i<TOT+1; i++) {
			System.out.println("=== TEST #"+(i+1)+" ===");
			double time = launchTests(dataset, pname);
			if(i > 0)
				avg += time;
		}
		System.out.println("avg = "+(avg/TOT));
//		scalabilityTests(dataset, pname);
		
	}

	private static void scalabilityTests(String[] dataset, String[] pname) throws IOException {
		
		int delta = 50000;
		THETA_MIN = 8;
		
		for(int i=0; i<dataset.length; i++) {
			
			clearKnowledgeBases();
			
			Property p = new Property(pname[i], Property.TYPE_STRING, i);
			
			for(int j=0; j*delta <= sources.size(); j++) {
				
				loadKnowledgeBases(dataset[i], dataset[i], j*delta, (j+1)*delta);
				
				System.out.println(dataset[i]+" ("+sources.size()+")");
				sys_out += dataset[i]+" ("+sources.size()+")\n";
			
//				for(double theta=1; theta<=THETA_MAX; theta*=2)
//					testPassJoin(pname[i], theta);
			
				for(double theta=1; theta<=THETA_MIN; theta*=2)
					testReededFilter(p, theta);
	
				notify(sys_out);
				sys_out = "\n";
				
			}
		}
	}

	private static double launchTests(String[] dataset, String[] pname) throws IOException {
		
		THETA_MIN = 0.9;
		double ngf = Double.NaN;
		
		for(int i=0; i<dataset.length; i++) {
			clearKnowledgeBases();
			loadKnowledgeBases(dataset[i], dataset[i]);
			
			Property p = new Property(pname[i], Property.TYPE_STRING, i);
			
			System.out.println(dataset[i]+"\t"+pname[i]);
			sys_out += dataset[i]+"\n";
		
//				TreeSet<String> pjs = null, oafs = null;
			for(double theta=0.9; theta>=THETA_MIN; theta-=0.1) {
//				TreeSet<Couple> inf = testIndexNgFilter(p, theta);
				ngf = testNewNgFilter(p, theta);
				
//				TreeSet<Couple> ed = testEdJoinOnce(p, theta);
//			    	TreeSet<Couple> pj = 
//					testPassJoin(pname[i], theta);
//					pjs = new TreeSet<String>();
//					for(Couple c : pj) {
//		//				System.out.println(c.getSource().getID()
//		//						+ "\t" + c.getTarget().getID()
//		//						+ "\t" + c.getSource().getOriginalPropertyValue(pname)
//		//						+ "\t" + c.getTarget().getOriginalPropertyValue(pname)
//		//						+ "\td = " + c.getDistances().get(0));
//						pjs.add(c.getSource().getOriginalPropertyValue(pname[i])+"#"+
//								c.getTarget().getOriginalPropertyValue(pname[i]));
//					}
//			}
//		
//			for(double theta=1; theta<=THETA_MAX; theta++) {
//				System.out.println("theta = "+theta);
//			    	TreeSet<Couple> oaf = 
//					TreeSet<Couple> rd = testReededFilter(p, theta);
//					TreeSet<String> rd_id = new TreeSet<String>();
//					oafs = new TreeSet<String>();
//					for(Couple c : rd) {
//						rd_id.add(c.toString());
//						System.out.println(c.getSource().getID()
//								+ "\t" + c.getTarget().getID()
//								+ "\t" + c.getSource().getOriginalPropertyValue(pname)
//								+ "\t" + c.getTarget().getOriginalPropertyValue(pname)
//								+ "\td = " + c.getDistances().get(0));
//						oafs.add(c.getSource().getOriginalPropertyValue(pname[i])+"#"+
//								c.getTarget().getOriginalPropertyValue(pname[i]));
//					}
//					for(Couple edc : ed) {
//						boolean f = false;
//						for(Couple rdc : rd) {
//							if(rdc.toString().equals(edc.toString())) {
//								f = true;
//								break;
//							}
//						}
//						if(!f) {
//							System.out.println(edc.getSource().getPropertyValue(pname[i])
//							+ "\t" + edc.getTarget().getPropertyValue(pname[i])
//							+ "\td = " + edc.getDistances().get(0));
//							ReededFilter rf = new ReededFilter();
//							System.out.println(rf.getDistance(edc.getSource().getPropertyValue(pname[i]), 
//									edc.getTarget().getPropertyValue(pname[i])));
//						}
//					}
			}

//			crossValidate(pjs, oafs);



//				for(double theta=1; theta<=THETA_MAX; theta++)
//						testEdJoinOnce(pname[i], theta);
			
//			notify(sys_out);
			sys_out = "\n";
		}
		return ngf;
	}

	@SuppressWarnings("unused")
	private static void crossValidate(TreeSet<String> pjs, TreeSet<String> oafs) {
			System.out.println("\nPJ but not OAF");
			for(String s : pjs)
				if(!oafs.contains(s))
					System.out.println(s);

			System.out.println("\nOAF but not PJ");
			for(String s : oafs)
				if(!pjs.contains(s))
					System.out.println(s);
	}

}
