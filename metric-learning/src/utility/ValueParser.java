package utility;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ValueParser {

	public static double parse(String s) {
		if(s.equals(""))
			return Double.NaN;
//			return 0.0;
        Matcher m = Pattern.compile("\\d+").matcher(s);
        if(m.find()) {
            double i = Double.parseDouble(m.group());
            return i;
        }
		return Double.parseDouble(s);
	}
	
}
