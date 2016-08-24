package acids2.multisim;

import java.util.ArrayList;

import acids2.Couple;

public class MultiSimOracle {

	private ArrayList<String> oraclesAnswers;
	
	public MultiSimOracle(ArrayList<String> oraclesAnswers) {
		this.oraclesAnswers = oraclesAnswers;
	}
    
	public boolean ask(String ids) {
		return oraclesAnswers.contains(ids);
	}

	public boolean ask(Couple c) {
		boolean feedback = ask(c.getID());
		c.setPositive(feedback);
		return feedback;
	}

	public ArrayList<String> getOraclesAnswers() {
		return oraclesAnswers;
	}	
	
}
