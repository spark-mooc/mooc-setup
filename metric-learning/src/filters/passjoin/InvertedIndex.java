package filters.passjoin;

import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;

import acids2.Resource;


/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class InvertedIndex {
	
	/*
	 * The tuple is formed by two integers:
	 * 		l = length of the original string
	 * 		i = position of the segment
	 * 
	 * Each tuple gives back a function L(w) where:
	 * 		w = segment (object of the search)
	 * 
	 */
	
	private HashMap<Tuple<Integer, Integer>, HashMap<String, TreeSet<Resource>>> Lmatrix = 
			new HashMap<Tuple<Integer, Integer>, HashMap<String, TreeSet<Resource>>>();
	
	public TreeSet<Resource> getStringsBySegment(String segm, int l, int i) {
		Tuple<Integer, Integer> tkey = getKey(l, i);
		HashMap<String, TreeSet<Resource>> Lli = Lmatrix.get(tkey);
		if(Lli != null)
			return Lli.get(segm);
		else
			return new TreeSet<Resource>();
	}
	
	public HashMap<String, TreeSet<Resource>> getLli(int l, int i) {		
		Tuple<Integer, Integer> tkey = getKey(l, i);
		HashMap<String, TreeSet<Resource>> Lli = Lmatrix.get(tkey);
		if(Lli != null)
			return Lli;
		else
			return new HashMap<String, TreeSet<Resource>>();
	}
	
	public void addToIndex(String segm, Resource res, int l, int i) {
		Tuple<Integer, Integer> tkey = getKey(l, i);
		HashMap<String, TreeSet<Resource>> Lli = Lmatrix.get(tkey);
		if(Lli != null) {
			if(Lli.containsKey(segm))
				Lli.get(segm).add(res);
			else {
				TreeSet<Resource> ts = new TreeSet<Resource>();
				ts.add(res);
				Lli.put(segm, ts);
			}
		} else {
			HashMap<String, TreeSet<Resource>> hm = new HashMap<String, TreeSet<Resource>>();
			TreeSet<Resource> ts = new TreeSet<Resource>();
			ts.add(res);
			hm.put(segm, ts);
			Lmatrix.put(new Tuple<Integer, Integer>(l, i), hm);
		}		
	}
	
	public String toString() {
		String str = "";
//		TODO
//		for(Tuple<Integer, Integer> key : Lmatrix.keySet()) {
//			str += (key + " -> ");
//			HashMap<String, TreeSet<Resource>> hm = Lmatrix.get(key);
//			for(String key2 : hm.keySet()) {
//				str += ""+key2+" -> ";
//				for(Resource value : hm.get(key2))
//					str += (value + ", ");
//			}
//			str += "\n";
//		}
		return str;
	}

	public int getSegmentLength(int l, int i) {
		Tuple<Integer, Integer> tkey = getKey(l, i);
		if(tkey != null) {
			HashMap<String, TreeSet<Resource>> Lli = Lmatrix.get(tkey);
			Set<String> keys = Lli.keySet();
			if(keys.isEmpty())
				return 0;
			else
				return ((String)keys.toArray()[0]).length();
		}
		return 0;
	}

	public Tuple<Integer, Integer> getKey(int l, int i) {
		for(Tuple<Integer, Integer> tkey : Lmatrix.keySet())
			if(tkey.getX() == l && tkey.getY() == i) {
				return tkey;
			}
		return null;
	}

}