package acids2.multisim;

import java.util.ArrayList;

public class MultiSimProperty {

	private String name;
	private int datatype;
	private MultiSimMeasures measures;

	private ArrayList<MultiSimSimilarity> similarities = new ArrayList<MultiSimSimilarity>();

	public ArrayList<MultiSimSimilarity> getSimilarities() {
		return similarities;
	}

	public MultiSimProperty(String name, int datatype, int index, MultiSimMeasures measures) {
		super();
		this.name = name;
		this.datatype = datatype;
		this.measures = measures;
		
		switch(datatype) {
		case MultiSimDatatype.TYPE_STRING:
			similarities.add(new MultiSimWeightedNgramSimilarity(this, index));
			similarities.add(new MultiSimCosineSimilarity(this, index + 1));
			similarities.add(new MultiSimWeightedEditSimilarity(this, index + 2));
			break;
		case MultiSimDatatype.TYPE_NUMERIC:
			similarities.add(new MultiSimNumericSimilarity(this, index));
			break;
		case MultiSimDatatype.TYPE_DATETIME: // TODO datetime similarity and filtering?
			similarities.add(new MultiSimWeightedNgramSimilarity(this, index));
			similarities.add(new MultiSimCosineSimilarity(this, index + 1));
			similarities.add(new MultiSimWeightedEditSimilarity(this, index + 2));
			break;
		default:
			System.err.println("Error: Invalid datatype for property " + name + ".");
			break;
		}
		
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public MultiSimMeasures getMeasures() {
		return measures;
	}

	public int getDatatype() {
		return datatype;
	}
	public void setDatatype(int datatype) {
		this.datatype = datatype;
	}
	
	public int getSize() {
		return similarities.size();
	}

}
