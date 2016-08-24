package utility;

import java.text.Normalizer;
import java.text.Normalizer.Form;

public class StringUtilities {

	/**
	 * This method filters out all the characters that are different from:
	 * digits (ASCII code 48-57), upper case (65-90), lower-case letters (97-122) and space (32).
	 * @param in
	 * @return
	 */
    public static String normalize(String in) {
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
