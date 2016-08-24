package acids2.multisim;

public class MultiSimDatatype {

	public static final int TYPE_STRING = 0;
	public static final int TYPE_NUMERIC = 1;
	public static final int TYPE_DATETIME = 2;

	public static String asString(int type) {
		switch(type) {
		case TYPE_STRING:
			return "string";
		case TYPE_NUMERIC:
			return "numeric";
		case TYPE_DATETIME:
			return "datetime";
		default:
			return "unknown";
		}
	}
	
}
