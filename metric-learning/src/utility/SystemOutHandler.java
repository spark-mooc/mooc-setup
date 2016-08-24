package utility;

import java.io.OutputStream;
import java.io.PrintStream;

public class SystemOutHandler {

	static boolean statusOn = true;
	static PrintStream originalStream;
	
	public static void shutDown() {
		if(statusOn) {
	        // shutting up System.out
	        originalStream = System.out;
	        PrintStream dummyStream = new PrintStream(new OutputStream(){
	            public void write(int b) {
	                //NO-OP
	            }
	        });
	        System.setOut(dummyStream);
	        statusOn = false;
		}
	}
	
	public static void turnOn() {
		if(!statusOn) {
            // turning on System.out
            System.setOut(originalStream);
            statusOn = true;
		}
	}
}
