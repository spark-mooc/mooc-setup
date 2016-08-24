/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package metriclearning;

import java.text.Normalizer;
import java.text.Normalizer.Form;

import distances.WeightedEditDistanceExtended;



/**
 *
 * @author tom
 */
public class EditSimilarities {

    /**
     * Costs matrix, dimension: 64x64
     * 64 = 26 characters * 2 cases + 10 digits + space + empty string
     */
    private static double[][][] M;
    private static int[][][] count;
    private static final int MDIM = 64;
    private static final int EMPTY_STRING = MDIM - 1;
    private static String source, target;
    private static int N;
    private static int K;

    

    private static String filter(String sIn) {
        sIn = Normalizer.normalize(sIn, Form.NFD);
        String sOut = "";
        for (int i = 0; i < sIn.length(); i++) {
            char c = sIn.charAt(i);
            // digits 48-57, uppercase 65-90, lowercase 97-122
            if ((48 <= c && c <= 57) || (65 <= c && c <= 90) || (97 <= c && c <= 122) || c == 32) {
                sOut += c;
            }
        }
        return sOut;
    }

    private static void countSub(int i, int j, int k, Couple1 c) {
        count[charToPosition(source.charAt(i))][charToPosition(target.charAt(j))][k]++;
        c.count(charToPosition(source.charAt(i)), charToPosition(target.charAt(j)), k);
    }
    
    private static void countDel(int i, int k, Couple1 c) {
        count[charToPosition(source.charAt(i))][EMPTY_STRING][k]++;
        c.count(charToPosition(source.charAt(i)), EMPTY_STRING, k);
    }
    
    private static void countIns(int j, int k, Couple1 c) {
        count[EMPTY_STRING][charToPosition(target.charAt(j))][k]++;
        c.count(EMPTY_STRING, charToPosition(target.charAt(j)), k);
    }
        
    private static int charToPosition(char c) {
        // input:  digits 48-57, uppercase 65-90, lowercase 97-122
        // output: digits 0-9,   uppercase 10-35, lowercase 36-61
        if (48 <= c && c <= 57) {
            return c - 48;
        } else if (65 <= c && c <= 90) {
            return c - 55;
        } else if (97 <= c && c <= 122) {
            return c - 61;
        } else {
            return 62; // space
        }
    }
    
    private static char positionToChar(int p) {
        // input: digits 0-9,   uppercase 10-35, lowercase 36-61
        // output:  digits 48-57, uppercase 65-90, lowercase 97-122
        if (0 <= p && p <= 9) {
            return (char)(p + 48);
        } else if (10 <= p && p <= 35) {
            return (char)(p + 55);
        } else if (36 <= p && p <= 61) {
            return (char)(p + 61);
        } else {
            return ' '; // space
        }
    }

    /* ----- public methods ----- */
    
    public static void initialize(int n) {
        N = n;
        M = new double[MDIM][MDIM][N];
        count = new int[MDIM][MDIM][N];
        // initialize the costs matrix
        for (int i = 0; i < MDIM; i++) {
            for (int j = 0; j < MDIM; j++) {
                for (int k = 0; k < N; k++) {
                    if (i == j) {
                        M[i][j][k] = 0.0;
                    } else if((i >= 10 && i == j-26) || (j >= 10 && j == i-26)) {
                        M[i][j][k] = 0.1;
                    } else {
                        M[i][j][k] = 1.0;
                    }
                    count[i][j][k] = 0;
                }
            }
        }
    }

    public static void resetCount() {
        for (int i = 0; i < MDIM; i++) {
            for (int j = 0; j < MDIM; j++) {
                for (int k = 0; k < N; k++) {
                    count[i][j][k] = 0;
                }
            }
        }

    }

    
    public static double[] getCostsMatrixAsArray(int k) {
        double[] Marr = new double[4096];
        int h = 0;
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[i].length; j++) {
//                if (i != j) {
                    Marr[h] = M[i][j][k];
                    h++;
//                }
            }
        }
        return Marr;
    }

    public static int[] getCountMatrixAsArray(int k) {
        int[] Carr = new int[4096];
        int h = 0;
        for (int i = 0; i < count.length; i++) {
            for (int j = 0; j < count[i].length; j++) {
//                if (i != j) {
                    Carr[h] = count[i][j][k];
                    h++;
//                }
            }
        }
        return Carr;
    }

    public static void setWeight(int i, int j, int k, double d) {
        M[i][j][k] = d;
    }

    public static double getWeight(int i, int j, int k) {
        return M[i][j][k];
    }

    public static void main(String[] args) {
        
        initialize(1);
        
        for (int i = 0; i < M.length; i++) {
            for (int j = 0; j < M[i].length; j++) {
                if (i == j) {
                    M[i][j][0] = 0.0;
                } else {
                    M[i][j][0] = 1.0;
                }
            }
        }

//        String as = "The WASA2 object-oriented workflow management system";
//        String bs = "World Wide Database-integrating the Web, CORBA and databases";
//        String b = "Semantic Integration of Environmental Models for Application to Global Information Systems and Decision-Making";
        String a = "Democrat";
        String b = "Republican";
        Couple1 c = new Couple1(new Resource1(a), new Resource1(b));
        
//        double d = getDijkstraSimilarity(a, b, 0, c);
        double d = getEditSimilarity(a, b, 0, c);
//        System.out.println("a = "+a);
//        System.out.println("b = "+b);
        System.out.println("sim(a,b) = "+d);

    }

    public static double getMatrixCheckSum(int k) {
        double sum = 0.0;
        for (int i = 0; i < MDIM; i++) {
            for (int j = 0; j < MDIM; j++) {
                sum += M[i][j][k];
            }
        }
        return sum;
    }

    public static void showCostsMatrix(int k, boolean showMaxMinValues) {
        System.out.println("");
        for(int i=0; i<M.length-1; i++)
            System.out.print("\t"+positionToChar(i));
        System.out.println("\tε");
        double lowest = 1.0;
        for (int i = 0; i < M.length; i++) {
            printChar(i,true);
            for (int j = 0; j < M[i].length; j++) {
                System.out.print(((double) (int) (M[i][j][k] * 1000)) / 1000 + "\t");
                if(M[i][j][k] < lowest && M[i][j][k] > 0)
                    lowest = M[i][j][k];
            }
            System.out.println("");
        }
        
        if(showMaxMinValues) {
            System.out.println("Highest");
            for(int i=0; i<M.length; i++) {
                for(int j=0; j<M[i].length; j++)
                    if(M[i][j][k] > 0.9) {
                        System.out.print("(");
                        printChar(i,false);
                        System.out.print(", ");
                        printChar(j,false);
                        System.out.println(") = "+M[i][j][k]);
                    }
            }
            System.out.println("Lowest");
            for(int i=0; i<M.length; i++) {
                for(int j=0; j<M[i].length; j++)
                    if(M[i][j][k] == lowest) {
                        System.out.print("(");
                        printChar(i,false);
                        System.out.print(", ");
                        printChar(j,false);
                        System.out.println(") = "+M[i][j][k]);
                    }
            }
        }
    }
    
    public static double getEditSimilarity(String s, String t, int k, Couple1 c) {
        
        K = k;
        source = filter(s);
        target = filter(t);
        
        WeightedEditDistanceExtended wed = new WeightedEditDistanceExtended() {
            private double f(double d) {
                return Math.log( - 0.99 * d + 1.00 ) / 20.0;
            }
            @Override
            public double matchWeight(char c) {
                return 0.0;
            }
            @Override
            public double deleteWeight(char c) {
                return f( M[ charToPosition(c) ][63][K] );
            }
            @Override
            public double insertWeight(char c) {
                return f( M[63][ charToPosition(c) ][K] );
            }
            @Override
            public double substituteWeight(char c1, char c2) {
                return f( M[ charToPosition(c1) ][ charToPosition(c2) ][K] );
            }
            @Override
            public double transposeWeight(char c1, char c2) {
                return Double.NEGATIVE_INFINITY;
            }
        };
        
        double dist = wed.distance(source, target);
        double[][] lev = wed.getMatrix();
        
        int i = lev.length-1;
        int j = lev[i].length-1;
        while(i != 0 || j != 0) {
            double left = j>0 ? -lev[i][j-1] : Double.POSITIVE_INFINITY;
            double up = i>0 ? -lev[i-1][j] : Double.POSITIVE_INFINITY;
            double upleft = (i>0 && j>0) ? -lev[i-1][j-1] : Double.POSITIVE_INFINITY;
            
//            char I = source.charAt(i-1);
//            char J = target.charAt(j-1);
            
            if(upleft <= left && upleft <= up) {
                // substitution
//                System.out.println("sub("+I+","+J+")");
                countSub(i-1, j-1, k, c);
                i--;
                j--;
            } else {
                if(left < upleft) {
                    // insertion
//                    System.out.println("ins("+J+")");
                    countIns(j-1, k, c);
                    j--;
                } else {
                    // deletion
//                    System.out.println("del("+I+")");
                    countDel(i-1, k, c);
                    i--;
                }
            }
        }
        
        
        // distance-similarity conversion
        return 1.0 / (1.0 + dist);
        
        
    }

    private static void printChar(int i, boolean tab) {
        String t;
        if(tab) t = "\t"; else t = "";
        if(i < 63)
            System.out.print(positionToChar(i)+t);
        else
            System.out.print("ε"+t);
    }

}
