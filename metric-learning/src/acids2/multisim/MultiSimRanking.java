package acids2.multisim;

public class MultiSimRanking implements Comparable<MultiSimRanking> {

	private MultiSimSimilarity sim;
	private Double weight;
	
	
	public MultiSimRanking(MultiSimSimilarity sim, Double weight) {
		super();
		this.sim = sim;
		this.weight = weight;
	}

	@Override
	public int compareTo(MultiSimRanking o) {
		// inverse order!
		return Double.compare(o.getWeight(), this.getWeight());
	}


	public MultiSimSimilarity getSim() {
		return sim;
	}


	public Double getWeight() {
		return weight;
	}
	
	public String toString() {
		return sim.getName()+" | "+sim.getProperty().getName()+" | "+weight;
	}

}
