package acids2.multisim;

public abstract class MultiSimStringSimilarity extends MultiSimSimilarity {
	
	public MultiSimStringSimilarity(MultiSimProperty property, int index) {
		super(property, index);
	}

	public int getDatatype() {
		return MultiSimDatatype.TYPE_STRING;
	}
	
}
