package acids2;

import java.util.ArrayList;
import java.util.TreeMap;

import utility.StringUtilities;

/**
 * @author Tommaso Soru <tsoru@informatik.uni-leipzig.de>
 *
 */
public class Resource implements Comparable<Resource> {
    
    private String ID;
    
    /**
     * Properties including symbols.
     */
    private TreeMap<String, String> originalProperties = new TreeMap<String, String>();
    
    /**
     * Properties without symbols. Generally similarities are computed on these.
     */
    private TreeMap<String, String> properties = new TreeMap<String, String>();
    
    private ArrayList<String> propertyOrder = new ArrayList<String>();

    public Resource(String ID) {
        this.ID = ID;
    }

    public String getID() {
        return ID;
    }
    
    public String getPropertyValue(String p) {
        return properties.get(p);
    }
    
    public String getOriginalPropertyValue(String p) {
        return originalProperties.get(p);
    }

    public ArrayList<String> getPropertyNames() {
        return propertyOrder;
    }
    
    public void setPropertyValue(String p, String v) {
    	String vn = StringUtilities.normalize(v);
        originalProperties.put(p, v);
        properties.put(p, vn);
        propertyOrder.add(p);
    }
    
    // TODO Change to MultiSimDatatype!
    public int checkDatatype(String prop) {
    	if(this.getPropertyValue(prop).equals(""))
    		return Property.TYPE_NUMERIC;
    	try {
			Double.parseDouble(this.getPropertyValue(prop));
		} catch (NumberFormatException e) {
			return Property.TYPE_STRING;
		}
    	return Property.TYPE_NUMERIC;
    }

	@Override
	public int compareTo(Resource o) {
		return this.getID().compareTo(o.getID());
	}
    
    
}
