package utility;

public class Transform {

	private static final double DEG = 1.0;
	
	public static double toSimilarity(double d) {
		if(d < 0)
			d = 0;
		return 1.0 / Math.pow(1.0 + d, 1.0 / DEG);
	}
	
	public static double toDistance(double s) {
		if(s < 0)
			s = 0;
		if(s > 1)
			s = 1;
		return 1.0 / Math.pow(s, DEG) - 1.0;
	}
	
}
