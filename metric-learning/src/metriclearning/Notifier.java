/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 *
 * @author tom
 * Sends an email to notify about the computation results.
 */
public class Notifier {
    
    public static void notify(double f, double pre, double rec, double tp,
            double fp, double tn, double fn, int perc) {
            try {
                // Create a URL for the desired page
                URL url = new URL("http://mommi84.altervista.org/notifier/index.php?"
                        + "f="+f+"&pre="+pre+"&rec="+rec+"&tp="+tp+"&fp="+fp
                        +"&tn="+tn+"&fn="+fn+"&perc="+perc);

                HttpURLConnection conn = (HttpURLConnection) url.openConnection();

                // Read all the text returned by the server
                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                in.close();
            } catch (IOException e) {
                    e.printStackTrace();
            }
    }
    
}
