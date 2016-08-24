/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.HashMap;
import java.util.Set;

/**
 *
 * @author tom
 */
public class Resource1 implements Comparable<Resource1> {
    
    private String ID;
    /**
     * originalProperties include symbols.
     * properties are for weighted edit distance calculation.
     */
    private HashMap<String, String> originalProperties = new HashMap<String, String>();
    private HashMap<String, String> properties = new HashMap<String, String>();

    public Resource1(String ID) {
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

    public Set<String> getPropertyNames() {
        return properties.keySet();
    }
    
    public void setPropertyValue(String p, String v) {
        originalProperties.put(p, v);
        properties.put(p, normalize(v));
    }

	@Override
	public int compareTo(Resource1 o) {
		return this.getID().compareTo(o.getID());
	}
    
	/**
	 * This method filters out all the characters that are different from:
	 * digits (ASCII code 48-57), upper case (65-90), lower-case letters (97-122) and space (32).
	 * @param in
	 * @return
	 */
    private static String normalize(String in) {
        in = Normalizer.normalize(in, Form.NFD).trim();
        String out = "";
        for(int i=0; i<in.length(); i++) {
            char c = in.charAt(i);
            if((48 <= c && c <= 57) || (65 <= c && c <= 90) || (97 <= c && c <= 122) || c == 32)
                out += c;
        }
        return out;
    }
    
}
