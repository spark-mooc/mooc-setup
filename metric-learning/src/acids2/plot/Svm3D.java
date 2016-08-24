package acids2.plot;

import java.util.TreeSet;

import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import org.jzy3d.chart.Chart;
import org.jzy3d.chart.ChartLauncher;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

public class Svm3D {

	private static final double BOUND = 1;
	private static final float POINT_SIZE = 3;

	private static Chart chart;
	
	private static svm_model model;
	private static double[][] sv;
	private static double[][] alpha;
	private static double theta;
	private static double theta0;
	
	private static TreeSet<Double> unique_x_0 = new TreeSet<Double>();
	private static TreeSet<Double> unique_y_0 = new TreeSet<Double>();
	private static TreeSet<Double> unique_x_1 = new TreeSet<Double>();
	private static TreeSet<Double> unique_y_1 = new TreeSet<Double>();
	
	private static int i0 = 0, i1 = 1, i2 = 2; // plot orientation

	public static void draw(svm_model _model, svm_problem problem,
			double _theta, double[][] _sv_d, double _theta0) {
		
		model = _model;
		theta = _theta;
		alpha = model.sv_coef;
		sv = _sv_d;
		theta0 = _theta0;
				
		if(!orientate()) {
			System.err.println("Can't plot classifier.");
			return;
		}
		
		// Define a function to plot
		Mapper mapper1 = new Mapper() {
		    public double f(double x0, double x1) {
		    	return Svm3D.f(x0, x1, +1);
		    }
		};
		Mapper mapper2 = new Mapper() {
		    public double f(double x0, double x1) {
		    	return Svm3D.f(x0, x1, -1);
		    }
		};

		svm_node[][] x = problem.x;
		double[] y = problem.y;
		
		// Define range and precision for the function to plot
		Range range = new Range(0, BOUND);
		int steps = 200;

		// Create a surface drawing that function
		Shape surface1 = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper1);
		surface1.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface1.getBounds().getZmin(), surface1.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
		surface1.setFaceDisplayed(true);
		surface1.setWireframeDisplayed(false);
		surface1.setWireframeColor(Color.BLACK);
		Shape surface2 = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper2);
		surface2.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface2.getBounds().getZmin(), surface2.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
		surface2.setFaceDisplayed(true);
		surface2.setWireframeDisplayed(false);
		surface2.setWireframeColor(Color.BLACK);
		
//		surface2.getBounds().setZmax((float) BOUND);

		Coord3d[] points = new Coord3d[problem.l];
		Color[] colors = new Color[problem.l];

		// Create scatter points
		for(int i=0; i<points.length; i++){
		    float x0 = (float) x[i][i0].value;
		    float y0 = (float) x[i][i1].value;
		    float z0 = (float) x[i][i2].value;
		    points[i] = new Coord3d(x0, y0, z0);
		    if(y[i] == 1)
		    	colors[i] = Color.BLACK;
		    else
		    	colors[i] = Color.RED;
		}       

		// Create a drawable scatter with a colormap
		Scatter scatter = new Scatter( points, colors, POINT_SIZE );
		
		// Create a chart and add the surface
		chart = new Chart(Quality.Advanced);
		if(unique_x_0.size() > 1 && unique_y_0.size() > 1) {
			chart.getScene().getGraph().add(surface1);
			System.out.println("PLOT: surface1 plotted.");
		}
		if(unique_x_1.size() > 1 && unique_y_1.size() > 1) {
			chart.getScene().getGraph().add(surface2);
			System.out.println("PLOT: surface2 plotted.");
		}
		chart.getScene().getGraph().add(scatter);
//        Svm3D.drawInitialClassifier();
        chart.render();
		ChartLauncher.openChart(chart);
		
	}
	
	private static boolean orientate() {
		double den;
		while(true) {
			den = 0.0;
			for(int j=0; j<alpha[0].length; j++)
				// TODO handle ArrayIndexOutOfBoundsException in order to plot in 2D.
				den += alpha[0][j] * sv[j][i2];
			if(Math.abs(den) < 1E-3) {
				System.out.println("PLOT: orientation changed.");
				i0 = (++i0) % 3;
				i1 = (++i1) % 3;
				i2 = (++i2) % 3;
				if(i0 == 0)
					return false;
			} else return true;
		}
	}
	
	protected static double f(double x0, double x1, double sign) {
		if(model.param.kernel_type == svm_parameter.POLY) {
	    	double a = 0.0, b = 0.0, c = 0.0;
	    	for(int j=0; j<alpha[0].length; j++)
		    	a += alpha[0][j] * Math.pow(sv[j][i2], 2);
	    	for(int j=0; j<alpha[0].length; j++)
		    	b += alpha[0][j] * 2*sv[j][i2]*(x0*sv[j][i0]+x1*sv[j][i1]);
	    	for(int j=0; j<alpha[0].length; j++)
		    	c += alpha[0][j] * (Math.pow(x0*sv[j][i0], 2) + 
		    			Math.pow(x1*sv[j][i1], 2) +
		    			2*x0*x1*sv[j][i0]*sv[j][i1]);
	    	c += theta;
	    	double r = (-b + sign * Math.sqrt(Math.pow(b, 2) - 4 * a * c))/(2 * a);
	    	if(r > BOUND || r < 0.0 || Double.isNaN(r)) return Double.NaN; else {
	    		if(sign == 1.0) {
	    			unique_x_0.add(x0);
	    			unique_y_0.add(x1);
	    		} else {
	    			unique_x_1.add(x0);
	    			unique_y_1.add(x1);
	    		}
	    		return r;
	    	}
		} else {
			double num = -theta;
			for(int j=0; j<alpha[0].length; j++)
				num -= alpha[0][j] * (x0 * sv[j][i0] + x1 * sv[j][i1]);
			double den = 0.0;
			for(int j=0; j<alpha[0].length; j++)
				den += alpha[0][j] * sv[j][i2];
			double r = num / den;
	    	if(r > BOUND || r < 0.0 || Double.isNaN(r)) return Double.NaN; else {
    			unique_x_0.add(x0);
    			unique_y_0.add(x1);
	    		return r;
	    	}
		}
	}
	
	public static void drawInitialClassifier() {
		Mapper mapper = new Mapper() {
		    public double f(double x0, double x1) {
		    	double r = - x0 - x1 + theta0;
		    	if(r > BOUND || r < 0.0 || Double.isNaN(r)) return Double.NaN; else return r;
		    }
		};
		Range range = new Range(0, BOUND);
		int steps = 200;
		Shape surface = Builder.buildOrthonormal(new OrthonormalGrid(range, steps, range, steps), mapper);
		surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new Color(1, 1, 1, .5f)));
		surface.setFaceDisplayed(true);
		surface.setWireframeDisplayed(false);
		surface.setWireframeColor(Color.BLACK);
		chart.getScene().getGraph().add(surface);

	}
	
}


