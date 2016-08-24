package filters.passjoin;

import java.util.LinkedList;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class PartitionStrategy implements Comparable<PartitionStrategy> {

	private LinkedList<String> partitions = new LinkedList<String>();
	private int n_min;
	private int n_max;
	private int marker = 0;
	private String name;
	
	public int getMarker() {
		return marker;
	}

	public int getN_min() {
		return n_min;
	}

	public void decreaseN_min() {
		this.n_min--;
	}

	public int getN_max() {
		return n_max;
	}

	public void decreaseN_max() {
		this.n_max--;
	}

	public void addPartition(String p) {
		partitions.add(p);
		marker += p.length();
	}
	
	public String getPartitionAt(int i) {
		return partitions.get(i);
	}
	
	public int getPartitionSize() {
		return partitions.size();
	}

	public LinkedList<String> getPartitions() {
		return partitions;
	}

	public PartitionStrategy(String name, int n_min, int n_max) {
		super();
		this.name = name;
		this.n_min = n_min;
		this.n_max = n_max;
	}
	
	public String getName() {
		return name;
	}

	public PartitionStrategy clone() {
		PartitionStrategy ps = new PartitionStrategy(name, n_min, n_max);
		for(String p : partitions)
			ps.addPartition(p);
		return ps;
	}

	@Override
	public int compareTo(PartitionStrategy ps) {
		if(partitions.size() != ps.getPartitionSize()) return -1;
		for(int i=0; i<partitions.size(); i++) {
			int c = this.getPartitionAt(i).compareTo(ps.getPartitionAt(i));
			if(c != 0)
				return c;
		}
		return 0;
	}
	
	public String toString() {
		String t = "";
		for(String p : partitions)
			if(t.equals(""))
				t = t + p;
			else
				t = t + "-" + p;
		return t;
	}
	
}
