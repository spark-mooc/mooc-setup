package acids2.output;

import java.io.IOException;
import java.util.ArrayList;

import acids2.Property;
import acids2.Resource;

public interface ScriptCreator {
	
	public void create(ArrayList<Resource> sources, ArrayList<Resource> targets, 
			ArrayList<Property> props, double[] w_linear, double theta) throws IOException;

}
