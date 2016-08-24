package filters.reeding;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Scanner;
import java.util.TreeSet;

import utility.StringUtilities;

import filters.WeightedNGramFilter;

import acids2.Couple;
import acids2.Resource;

// TODO extends...
public class WeightedPPJoinPlus {

	/**
	 * The "n" of n-grams.
	 */
	private static final int n = 3;
	
	public static ArrayList<Couple> run(ArrayList<Resource> sources,
			ArrayList<Resource> targets, String propertyName, double theta) {
		ArrayList<Couple> results = new ArrayList<Couple>(); // S
		
		double t = theta / (2.0 - theta);
		
		ArrayList<Resource> srcArray = new ArrayList<Resource>(sources);
		ArrayList<Resource> tgtArray = new ArrayList<Resource>(targets);
		
		ArrayList<Record> src = new ArrayList<Record>(); // R_s
		ArrayList<Record> tgt = new ArrayList<Record>(); // R_t
		
		for(Resource r : srcArray)
			src.add(new Record(WeightedNGramFilter.getNgrams(r.getPropertyValue(propertyName), n), r));
		for(Resource r : tgtArray)
			tgt.add(new Record(WeightedNGramFilter.getNgrams(r.getPropertyValue(propertyName), n), r));
		
		Collections.sort(src, new RecordComparator());
		Collections.sort(tgt, new RecordComparator());
		
		// xs = record = <resource, set of tokens>
		HashMap<String, ArrayList<Record>> index = new HashMap<String, ArrayList<Record>>(); // I_i
		
		for(int xi=0; xi<src.size(); xi++) {
			ArrayList<String> x = src.get(xi).x;
//			System.out.println("xi = "+xi+"\txs = "+xs);
//			for(int j=0; j<tgt.size(); j++) {
//				ArrayList<String> xt = tgt.get(j);
			HashMap<Record, Integer> setA = new HashMap<Record, Integer>(); // A
			int p = x.size() - (int) Math.ceil(t * x.size()) + 1;
//			System.out.println("p = "+p);
			int alpha = 0;
			for(int i=1; i<=p; i++) {
				String w = x.get(i-1);
//				System.out.println("\tw = "+w);
				ArrayList<Record> Iw = index.get(w);
//				System.out.println("\tI_w = "+Iw);
				if(Iw != null) {
					for(Record y : Iw) {
						if(y.x.size() < t * x.size())
							continue;
						alpha = (int) Math.ceil((t / (1 + t)) * (x.size() + y.x.size()));
						int ubound = 1 + Math.min(x.size() - i, y.x.size() - y.j);
						if(setA.get(y) == null)
							setA.put(y, 0);
						if(setA.get(y) + ubound >= alpha)
							setA.put(y, setA.get(y) + 1);
						else
							setA.put(y, 0);
					}
				} else {
					Iw = new ArrayList<Record>();
					index.put(w, Iw);
				}
				src.get(xi).j = i;
				Iw.add(src.get(xi));
			}
//			}
			// verify
			for(Record y : setA.keySet()) {
				if(setA.get(y) > 0) {
					String wx = x.get(p-1);
					int py = y.x.size() - (int) Math.ceil(t * y.x.size()) + 1;
					String wy = y.x.get(py-1);
//					System.out.println("\t# "+wx+"; "+wy);
					int over = setA.get(y); // O
					alpha = (int) Math.ceil((t / (1 + t)) * (x.size() + y.x.size()));
					if(wx.compareTo(wy) < 0) {
						int ubound = setA.get(y) + x.size() - p;
						if(ubound >= alpha)
							over = over + joinAndCount(x, y.x, p+1, setA.get(y)+1);
					} else {
						int ubound = setA.get(y) + y.x.size() - py;
						if(ubound >= alpha)
							over = over + joinAndCount(x, y.x, setA.get(y)+1, py+1);
					}
//					System.out.println("t="+t+"\tpy="+py+"\tover="+over+"\talpha="+alpha);
					if(over >= alpha) {
						Couple c = new Couple(src.get(xi).r, y.r);
						// TODO replace null with: new Property
						ReedingFilter rf = new ReedingFilter(null);
						// TODO replace "0" with: this.property.getIndex();
						c.setDistance(rf.getDistance(src.get(xi).r.getPropertyValue(propertyName), y.r.getPropertyValue(propertyName)), 0); 
						results.add(c);
					} else {
						if(src.get(xi).r != y.r)
							System.out.println(over+"\t"+alpha+"\t"+src.get(xi).r.getPropertyValue(propertyName)+" # "+ y.r.getPropertyValue(propertyName));
					}
				}
			}
		}
		
		return results;
	}
	
	private static int joinAndCount(ArrayList<String> x, ArrayList<String> y,
			int a, int b) {
		int count = 0;
		for(int i=a-1; i<x.size(); i++) {
			String s1 = x.get(i);
			for(int j=b-1; j<y.size(); j++) {
				String s2 = y.get(j);
				if(s1.equals(s2)) {
					count++;
					y.remove(j);
					break;
				}
			}
		}
		return count;
	}

	public static void main(String[] args) {
	
		System.out.println("Filter started...");
		
		String[] strings = load();
		ArrayList<Resource> sources = new ArrayList<Resource>();
		ArrayList<Resource> targets = new ArrayList<Resource>();
		String propertyName = "name";
		for(String s : strings) {
			Resource r = new Resource(s);
			r.setPropertyValue(propertyName, s);
			sources.add(r);
			targets.add(r);
		}
		double theta = 0.5;
		
		ArrayList<Couple> res = run(sources, targets, propertyName, theta);
		for(Couple c : res)
			System.out.println(c+"\t"+c.getDistanceAt(0));
	}

	private static String[] load() {
		Scanner in = null;
		try {
			in = new Scanner(new File("dblp.txt"));
		} catch (FileNotFoundException e) {}
		ArrayList<String> ar = new ArrayList<String>();
		while(in.hasNext()) {
			ar.add(StringUtilities.normalize(in.nextLine()));
		}
		in.close();
		return ar.toArray(new String[0]);
	}

}

class Record {
	
	ArrayList<String> x;
	Resource r;
	Integer j;
	
	protected Record(ArrayList<String> x, Resource r) {
		this.x = x;
		this.r = r;
	}

	protected Record(ArrayList<String> x, Resource r, Integer j) {
		this.x = x;
		this.r = r;
		this.j = j;
	}

}

class RecordComparator implements Comparator<Record> {

	@Override
	public int compare(Record o1, Record o2) {
		return o1.x.size() - o2.x.size();
	}

}

