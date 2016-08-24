package acids2.output;

public class Fscore {

	private String description;
	private double f1, pre, rec, tp, fp, tn, fn;
	
	public String getDescription() {
		return description;
	}

	public void setDescription(String description) {
		this.description = description;
	}

	public double getTp() {
		return tp;
	}

	public void setTp(double tp) {
		this.tp = tp;
	}

	public double getFp() {
		return fp;
	}

	public void setFp(double fp) {
		this.fp = fp;
	}

	public double getTn() {
		return tn;
	}

	public void setTn(double tn) {
		this.tn = tn;
	}

	public double getFn() {
		return fn;
	}

	public void setFn(double fn) {
		this.fn = fn;
	}

	public double getF1() {
		return f1;
	}

	public double getPre() {
		return pre;
	}

	public double getRec() {
		return rec;
	}

	// TODO Create object before evaluation and add methods: .countTP() .countFP() etc.
	public Fscore(String description, double tp, double fp, double tn, double fn) {
		super();
		this.description = description;
		this.tp = tp;
		this.fp = fp;
		this.tn = tn;
		this.fn = fn;
        pre = tp+fp != 0 ? tp / (tp + fp) : 0;
        rec = tp+fn != 0 ? tp / (tp + fn) : 0;
        f1 = pre+rec != 0 ? 2 * pre * rec / (pre + rec) : 0;
	}

	public void print() {
        System.out.println("pre = "+pre+", rec = "+rec);
        System.out.println("f1 = "+f1+" (tp="+tp+", fp="+fp+", tn="+tn+", fn="+fn+")");
	}
	
	public String toString() {
		return "pre = "+pre+", rec = "+rec+"\n"+"f1 = "+f1+" (tp="+tp+", fp="+fp+", tn="+tn+", fn="+fn+")";
	}
	
}
