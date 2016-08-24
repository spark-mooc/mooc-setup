package utility;

import java.util.Arrays;

public class Statistics {

	private double[] data;
	private double size;

	public Statistics(double[] data) {
		this.data = data;
		size = data.length;
	}

	public double getMean() {
		double sum = 0.0;
		for (double a : data)
			sum += a;
		return sum / size;
	}

	public double getVariance() {
		double mean = getMean();
		double temp = 0;
		for (double a : data)
			temp += (mean - a) * (mean - a);
		return temp / size;
	}

	public double getStdDev() {
		return Math.sqrt(getVariance());
	}

	public double getPercentile(double p) {
		if(p < 0 || p > 1)
			return Double.NaN;
		int n = (int) (data.length * p);

		double[] sortedArray = Arrays.copyOf(data, data.length); 
		Arrays.sort(sortedArray);
		if(n == 0)
			return sortedArray[0] / 2;
		return (sortedArray[n - 1] + sortedArray[n]) / 2;
	}
	
	public static void main(String[] args) {
		double[] t = {0.32, 0.231312, 0.1, 1.0, 0.3231, 0.6, 0.2};
		Statistics s = new Statistics(t);
		System.out.println(s.getPercentile(0.2));
		System.out.println(s.getPercentile(0.5));
		System.out.println(s.getPercentile(0));
		System.out.println(s.getPercentile(1));
	}

}
