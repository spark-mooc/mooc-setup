package acids2;

import filters.StandardFilter;
import filters.mahalanobis.MahalaFilter;
import filters.reeding.CrowFilter;

public class Property {
	
	public static final int TYPE_STRING = 0;
	public static final int TYPE_NUMERIC = 1;
	public static final int TYPE_DATETIME = 2;

	private String name;
	private int datatype;
	private StandardFilter filter;

	private boolean noisy = false;
	
	private boolean filtered = false;

	public boolean isFiltered() {
		return filtered;
	}

	public void setFiltered(boolean filtered) {
		this.filtered = filtered;
	}

	private int index;

	public Property(String name, int datatype, int index) {
		super();
		this.name = name;
		this.datatype = datatype;
		this.index = index;
		switch(datatype) {
		case TYPE_STRING:
			this.filter = new CrowFilter(this); // HybridFilter(this);
			break;
		case TYPE_NUMERIC:
			this.filter = new MahalaFilter(this);
			break;
		case TYPE_DATETIME: // TODO datetime similarity and filtering?
			this.filter = new MahalaFilter(this);
			break;
		default: // string comparison always works.
			this.filter = new CrowFilter(this); // HybridFilter(this);
			break;
		}
	}
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public int getDatatype() {
		return datatype;
	}
	public void setDatatype(int datatype) {
		this.datatype = datatype;
	}
	public StandardFilter getFilter() {
		return filter;
	}
	public void setFilter(StandardFilter filter) {
		this.filter = filter;
	}

	public String getDatatypeAsString() {
		switch(datatype) {
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
	
	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public boolean isNoisy() {
		return noisy;
	}

	public void setNoisy(boolean noisy) {
		this.noisy = noisy;
	}
	
}
