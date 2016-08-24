package acids2.multisim;

import java.util.ArrayList;

import acids2.Couple;

public class MultiSimTrainer {

	MultiSimSetting setting;
	
	public MultiSimTrainer(MultiSimSetting setting) {
		this.setting = setting;
	}

	/**
	 * Selects and trains <i>size</i> couples from the primary set ordered by their gamma values.
	 * Ascending or descending order depends on <i>nearest</i>.
	 * @param primary Set of couples, usually the set of positive or negative examples.
	 * @param secondary Set of couples, usually the complementary of primary.
	 * @param inferred Set of logically inferred (or labeled) couples so far.
	 * @param labelled Set of manually labeled couples so far.
	 * @param poslbl Set of positive-labeled couples.
	 * @param neglbl Set of negative-labeled couples.
	 * @param size Query size.
	 * @param nearest Select the nearest couples (points) to the classifier.
	 */
	public void train(ArrayList<Couple> primary, ArrayList<Couple> secondary, 
			ArrayList<Couple> inferred, ArrayList<Couple> labelled, ArrayList<Couple> poslbl, ArrayList<Couple> neglbl, int size, boolean nearest) {
		
		ArrayList<Couple> temp = new ArrayList<Couple>();
		for(int i=0; temp.size() < size && i < primary.size(); i++) {
			Couple c = nearest ? primary.get(i) : primary.get(primary.size()-i-1);
			if(!inferred.contains(c)) {
				temp.add(c);
				if(setting.getOracle().ask(c))
					poslbl.add(c);
				else
					neglbl.add(c);
			}
		}
		labelled.addAll(temp);
		// needed to prevent looping if called again.
		primary.removeAll(temp);
		secondary.removeAll(temp);
	}

}
