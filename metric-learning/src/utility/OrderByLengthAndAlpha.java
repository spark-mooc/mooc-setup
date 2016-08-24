package utility;

import java.util.Comparator;

import acids2.Resource;


public class OrderByLengthAndAlpha implements Comparator<Resource> {
	
	private String propertyName = "";
	
	public OrderByLengthAndAlpha(String propertyName) {
		this.propertyName = propertyName;
	}
	
	@Override
	public int compare(Resource r1, Resource r2) {
		
		String o1 = r1.getPropertyValue(propertyName);
		String o2 = r2.getPropertyValue(propertyName);
				
		if (o1.length() > o2.length()) {
			return 1;
		} else if (o1.length() < o2.length()) {
			return -1;
		} else {
			return o1.compareTo(o2);
		}
	}
}
