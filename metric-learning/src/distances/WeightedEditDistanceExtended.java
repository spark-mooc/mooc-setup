/*
 * LingPipe v. 4.1.0
 * Copyright (C) 2003-2011 Alias-i
 *
 * This program is licensed under the Alias-i Royalty Free License
 * Version 1 WITHOUT ANY WARRANTY, without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the Alias-i
 * Royalty Free License Version 1 for more details.
 * 
 * You should have received a copy of the Alias-i Royalty Free License
 * Version 1 along with this program; if not, visit
 * http://alias-i.com/lingpipe/licenses/lingpipe-license-1.txt or contact
 * Alias-i, Inc. at 181 North 11th Street, Suite 401, Brooklyn, NY 11211,
 * +1 (718) 290-9170.
 */

package distances;

import com.aliasi.util.Distance;
import com.aliasi.util.Proximity;

/**
 * The <code>WeightedEditDistanceExtended</code> class implements both the
 * proximity and distance interfaces based on the negative proximity
 * weights assigned to independent atomic edit operations.
 *
 * <h4>Weights Scaled as Log Probability</h4>
 *
 * <p>Weights on edit operations are scaled as log probabilities.  
 * Practically speaking, this means that the larger the weight, the
 * more likely the edit operation; keep in mind that -1 is larger than
 * -3, representing <code>2<sup>-1</sup> = 1/2</code> and
 * <code>2<sup>-3</sup> = 1/8</code> respectively on a linear
 * probability scale.
 *
 * <h4>Proximity and Edit Sequences</h4>
 
 * <p>The log probability of a sequence of independent edits is the
 * sum of the log probabilities of the individual edits.  Proximity
 * between strings <code>s1</code> and <code>s2</code> is defined as
 * the maximum sum of edit weights over sequences of edits that
 * convert <code>s1</code> to <code>s2</code>.
 *
 * <p>Like the individual edit weights, proximity is scaled as
 * a log probability of the complete edit.  The larger the proximity,
 * the closer the strings; again, keep in mind that -10 is larger than
 * -20, representing roughly 1/1000 and 1/1,000,000 on the linear
 * probability scale.
 *
 * <h4>Distance is Negative Proximity</h4>
 *
 * <p>Distance is just negative proximity.  This scales edit distances
 * in the usual way, with distance of 3 between strings indicating they
 * are further away from each other than strings at distance 1.25.
 *
 * <h4>Relation to Simple Edit Distance</h4>
 * 
 * <P>This class generalizes the behavior of the class
 * <code>spell.EditDistance</code> without extending it in the inheritance
 * sense.  Weighted edit distance agrees with edit distance (up to
 * arithmetic precision) as a distance assuming the following weights:
 * match weight is 0, substitute, insert and delete weights are
 * <code>-1</code>, and the transposition weight is <code>-1</code> if
 * transpositions are allowed in the edit distance and
 * <code>Double.NEGATIVE_INFINITY</code> otherwise.
 *
 * <h4>Symmetry</h4>
 * 
 * <P>If the substitution and transposition weights are symmetric and
 * the insert and delete costs of a character are equal, then weighted
 * edit distance will be symmetric.  
 * 
 * <h4>Metricity</h4>
 *
 * <p>If the match weight of all
 * characters is zero, then the distance between a character sequence
 * and itself will be zero.  

 * <p>If transpose weights are negative infinity so that transposition is
 * not allowed, and if the assignment of substitution weights forms a
 * metric (see {@link Distance} for a definition), and if delete and
 * insert weights are non-negative and equal for all characters, and
 * if match weights are all zero, then weighted edit distance will
 * form a proper metric.  Other values may also form metrics, such as
 * a weight of -1 for all edits other than transpose.
 *
 *
 * <h4>Probabilistic Channel</h4>
 *
 * <p>A probabilistic relational model between strings is defined if
 * the weights are properly scaled as log probabilities.  Because
 * probabilities are between 0 and 1, log probabilities will be
 * between negative infinity and zero.  Proximity between two strings
 * <code>in</code> and <code>out</code> is defined by:
 *
 * <blockquote><pre>
 * proximity(in,out)
 * = Max<sub><sub>edit(in)=out</sub></sub> log2 P(edit)
 * </pre></blockquote>
 *
 * where the cost of the edit is defined to be:
 *
 * <blockquote><code>
 * log2 P(edit) 
 * <br> = log2 P(edit<sub>0</sub>,...,edit<sub>n-1</sub>)
 * <br> ~ log2 P(edit<sub>0</sub>) + ... + log P(edit<sub>n-1</sub>)
 * </code></blockquote>
 *
 * The last line is an approximation assuming edits are
 * independent.
 * 
 * <p>In order to create a proper probabilistic channel, exponentiated
 * edit weights must sum to 1.0.  This is not technically possible
 * with a local model if transposition is allowed, because of boundary
 * conditions and independence assumptions.
 * 
 * It is possible to define a proper channel if transposition is off,
 * and if all edit weights for a position (including all sequences of
 * arbitrarily long insertions) sum to 1.0.  In particular, if any
 * edits at all are allowed (have finite weights), then there must be
 * a non-zero weight assigned to matching, otherwise exponentiated
 * edit weight sum would exceed 1.0.  It is always possible to add an
 * offset to normalize the values to a probability model (the offset
 * will be negative if the sum exceeds 1.0 and positive if it falls
 * below 1.0 and zero otherwise).

 * <p>A fully probabilistic model would have to take the sum over all
 * edits rather than the maximum.  This class makes the so-called
 * Viterbi approximation, assuming the full probability is close to
 * that of the best probability, or at least proportional to it.
 * 
 * 
 * @author  Bob Carpenter
 * @version 3.0
 * @since   LingPipe2.0
 */
public abstract class WeightedEditDistanceExtended 
    implements Distance<CharSequence>,
               Proximity<CharSequence> {

    double[][] matrix;
    
    /**
     * Construct a weighted edit distance.
     */
    public WeightedEditDistanceExtended() {
        /* do nothing */
    }

    /**
     * Returns the weighted edit distance between the specified
     * character sequences.  If the edit distances are interpreted as
     * entropies, this distance may be interpreted as the entropy of
     * the best edit path converting the input character sequence to
     * the output sequence.  The first argument is taken to be the
     * input and the second argument the output.
     *
     * <p>This method is thread
     * safe and may be accessed concurrently if the abstract weighting
     * methods are thread safe.
     *
     * @param csIn First character sequence.
     * @param csOut Second character sequence.
     * @return The edit distance between the sequences.
     */
    public double distance(CharSequence csIn, CharSequence csOut) {
        return -proximity(csIn,csOut);
    }

    /**
     * Returns the weighted proximity between the specified character
     * sequences. The first argument is taken to be the input and the
     * second argument the output.
     *
     * <p>This method is thread safe and may be accessed concurrently
     * if the abstract weighting methods are thread safe.
     *
     * @param csIn First character sequence.
     * @param csOut Second character sequence.
     * @return The edit distance between the sequences.
     */
    public double proximity(CharSequence csIn, CharSequence csOut) {
        return distance(csIn,csOut,false);
    }

    public double similarity(CharSequence csIn, CharSequence csOut) {
    	if(csIn.length() + csOut.length() == 0)
    		return 1.0;
        return 1.0 - distance(csIn,csOut,false) / (double)(Math.max(csIn.length(), csOut.length()));
    }
    
    /**
     * Returns the weighted edit distance between the specified
     * character sequences ordering according to the specified
     * similarity ordering.  The first argument is taken to
     * be the input and the second argument the output. 
     * If the boolean flag for similarity is set to <code>true</code>,
     * the distance is treated as a similarity measure, where
     * larger values are closer; if it is <code>false</code>, 
     * smaller values are closer.
     *
     * <p>This method is thread safe and may be accessed concurrently
     * if the abstract weighting methods are thread safe.
     *
     * @param csIn First character sequence.
     * @param csOut Second character sequence.
     * @param isSimilarity Set to <code>true</code> if distances are
     * similarities, false if they are dissimilarities.
     */
    double distance(CharSequence csIn, CharSequence csOut,
                    boolean isSimilarity) {

        int xsLength = csIn.length() + 1;  
        int ysLength = csOut.length() + 1; 
        matrix = new double[xsLength][ysLength];
        
        // can't reverse to make csOut always smallest, because weights
        // may be asymmetric

        if (ysLength == 1) {  // all deletes
            double sum = 0.0;
            matrix[0][0] = 0.0;
            for (int i = 0; i < csIn.length(); ++i) {
                sum += deleteWeight(csIn.charAt(i));
                matrix[i+1][0] = sum;
            }
            return sum;
        }
        if (xsLength == 1) { // all inserts
            double sum = 0.0;
            matrix[0][0] = 0.0;
            for (int j = 0; j < csOut.length(); ++j) {
                sum += insertWeight(csOut.charAt(j));
                matrix[0][j+1] = sum;
            }
            return sum;
        }
    

        // x=0: first slice, all inserts
        double lastSlice[] = new double[ysLength];
        lastSlice[0] = 0.0;  // upper left corner of lattice
        for (int y = 1; y < ysLength; ++y)
            lastSlice[y] = lastSlice[y-1] + insertWeight(csOut.charAt(y-1));

        // x=1: second slice, no transpose
        double[] currentSlice = new double[ysLength];
        currentSlice[0] = insertWeight(csOut.charAt(0));
        char cX = csIn.charAt(0);
        for (int y = 1; y < ysLength; ++y) {
            int yMinus1 = y-1;
            char cY = csOut.charAt(yMinus1);
            double matchSubstWeight 
                = lastSlice[yMinus1]
                +  ((cX == cY) ? matchWeight(cX) : substituteWeight(cX,cY));
            double deleteWeight = lastSlice[y] + deleteWeight(cX);
            double insertWeight = currentSlice[yMinus1] + insertWeight(cY);
            currentSlice[y] = best(isSimilarity,
                                   matchSubstWeight,
                                   deleteWeight,
                                   insertWeight);
        }
    
        // avoid third array allocation if possible
//        if (xsLength == 2) return currentSlice[currentSlice.length-1];

        char cYZero = csOut.charAt(0);
        double[] twoLastSlice = new double[ysLength];

        System.arraycopy(lastSlice, 0, matrix[0], 0, ysLength);
        System.arraycopy(currentSlice, 0, matrix[1], 0, ysLength);
        
        // x>1:transpose after first element
        for (int x = 2; x < xsLength; ++x) {
            char cXMinus1 = cX;
            cX = csIn.charAt(x-1);

            // rotate slices
            double[] tmpSlice = twoLastSlice;
            twoLastSlice = lastSlice;
            lastSlice = currentSlice;
            currentSlice = tmpSlice;

            currentSlice[0] = lastSlice[0] + deleteWeight(cX); 

            // y=1: no transpose here
            currentSlice[1] = best(isSimilarity,
                                   (cX == cYZero)
                                   ? (lastSlice[0] + matchWeight(cX))
                                   : (lastSlice[0] + substituteWeight(cX,cYZero)),
                                   lastSlice[1] + deleteWeight(cX),
                                   currentSlice[0] + insertWeight(cYZero));
        
            // y > 1: transpose
            char cY = cYZero;
            for (int y = 2; y < ysLength; ++y) {
                int yMinus1 = y-1;
                char cYMinus1 = cY;
                cY = csOut.charAt(yMinus1);
                currentSlice[y] = best(isSimilarity,
                                       (cX == cY)
                                       ? (lastSlice[yMinus1] + matchWeight(cX))
                                       : (lastSlice[yMinus1] + substituteWeight(cX,cY)),
                                       lastSlice[y] + deleteWeight(cX),
                                       currentSlice[yMinus1] + insertWeight(cY));
//                if(cX == cY) {
//                    if(currentSlice[y] == lastSlice[yMinus1] + matchWeight(cX))
//                        System.out.print("Mtc("+cX+") = ");
//                    else if(currentSlice[y] == lastSlice[y] + deleteWeight(cX))
//                        System.out.print("Del("+cX+") = ");
//                    else System.out.print("Ins("+cY+") = ");
//                } else {
//                    if(currentSlice[y] == lastSlice[yMinus1] + substituteWeight(cX,cY))
//                        System.out.print("Sub("+cX+","+cY+") = ");
//                    else if(currentSlice[y] == lastSlice[y] + deleteWeight(cX))
//                        System.out.print("Del("+cX+") = ");
//                    else System.out.print("Ins("+cY+") = ");
//                }
//                System.out.println(currentSlice[y]);
                if (cX == cYMinus1 && cY == cXMinus1)
                    currentSlice[y] = best(isSimilarity,
                                           currentSlice[y],
                                           twoLastSlice[y-2] + transposeWeight(cXMinus1,cX));
            }
            System.arraycopy(currentSlice, 0, matrix[x], 0, ysLength);
        }
        
//        for(int i=0; i<matrix.length; i++) {
//            for(int j=0; j<matrix[i].length; j++)
//                System.out.print(matrix[i][j]+"\t");
//            System.out.println("");
//        }
        return currentSlice[currentSlice.length-1];
    }

    private double best(boolean isSimilarity, double x, double y, double z) {
        return best(isSimilarity,x,best(isSimilarity,y,z));
    }

    private double best(boolean isSimilarity, double x, double y) {
        return isSimilarity
            ? Math.max(x,y)
            : Math.min(x,y);
    }

    /**
     * Returns the weight of matching the specified character.  For
     * most weighted edit distances, the match weight is zero so that
     * identical strings are total distance zero apart.
     *
     * <P>All weights should be less than or equal to zero, with
     * heavier weights being larger absolute valued negatives.
     * Basically, the weights may be treated as unscaled log
     * probabilities.  Thus valid values will range between 0.0
     * (probablity 1) and {@link Double#NEGATIVE_INFINITY}
     * (probability 0).  See the class documentation above for more
     * information.
     *
     * @param cMatched Character matched.
     * @return Weight of matching character.
     */
    public abstract double matchWeight(char cMatched);

    /**
     * Returns the weight of deleting the specified character.
     *
     * <P>All weights should be less than or equal to zero, with
     * heavier weights being larger absolute valued negatives.
     * Basically, the weights may be treated as unscaled log
     * probabilities.  Thus valid values will range between 0.0
     * (probablity 1) and {@link Double#NEGATIVE_INFINITY}
     * (probability 0).  See the class documentation above for more
     * information.
     *
     * @param cDeleted Character deleted.
     * @return Weight of deleting character.
     */
    public abstract double deleteWeight(char cDeleted);

    /**
     * Returns the weight of inserting the specified character.
     *
     * <P>All weights should be less than or equal to zero, with
     * heavier weights being larger absolute valued negatives.
     * Basically, the weights may be treated as unscaled log
     * probabilities.  Thus valid values will range between 0.0
     * (probablity 1) and {@link Double#NEGATIVE_INFINITY}
     * (probability 0).  See the class documentation above for more
     * information.
     *
     * @param cInserted Character inserted.
     * @return Weight of inserting character.
     */
    public abstract double insertWeight(char cInserted);

    /**
     * Returns the weight of substituting the inserted character for
     * the deleted character.
     *
     * <P>All weights should be less than or equal to zero, with
     * heavier weights being larger absolute valued negatives.
     * Basically, the weights may be treated as unscaled log
     * probabilities.  Thus valid values will range between 0.0
     * (probablity 1) and {@link Double#NEGATIVE_INFINITY}
     * (probability 0).  See the class documentation above for more
     * information.
     * 
     * @param cDeleted Deleted character.
     * @param cInserted Inserted character.
     * @return The weight of substituting the inserted character for
     * the deleted character.
     */
    public abstract double substituteWeight(char cDeleted, char cInserted);

    /**
     * Returns the weight of transposing the specified characters.  Note
     * that the order of arguments follows that of the input.
     *
     * <P>All weights should be less than or equal to zero, with
     * heavier weights being larger absolute valued negatives.
     * Basically, the weights may be treated as unscaled log
     * probabilities.  Thus valid values will range between 0.0
     * (probablity 1) and {@link Double#NEGATIVE_INFINITY}
     * (probability 0).  See the class documentation above for more
     * information.
     * 
     * @param cFirst First character in input.
     * @param cSecond Second character in input.
     * @return The weight of transposing the specified characters.
     */
    public abstract double transposeWeight(char cFirst, char cSecond);

    public double[][] getMatrix() {
        return matrix;
    }


}
