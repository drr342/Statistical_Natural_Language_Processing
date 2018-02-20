package nlp.assignments;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import nlp.classify.BasicFeatureVector;
import nlp.classify.BasicLabeledFeatureVector;
import nlp.classify.FeatureExtractor;
import nlp.classify.FeatureVector;
import nlp.classify.LabeledFeatureVector;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.math.DoubleArrays;
import nlp.util.Counter;
import nlp.util.Indexer;

/**
 * Maximum entropy classifier for assignment 2. You will have to fill in the
 * code gaps marked by TODO flags. To test whether your classifier is
 * functioning correctly, you can invoke the main method of this class using
 * <p/>
 * java nlp.assignments.MaximumEntropyClassifier
 * <p/>
 * This will run a toy test classification.
 */
public class _drr342PerceptronClassifier<I, F, L> implements
		ProbabilisticClassifier<I, L> {

	/**
	 * Factory for training MaximumEntropyClassifiers.
	 */
	public static class Factory<I, F, L> implements
			ProbabilisticClassifierFactory<I, L> {
		
		boolean average;
		FeatureExtractor<I, F> featureExtractor; 

		public ProbabilisticClassifier<I, L> trainClassifier(
				List<LabeledInstance<I, L>> trainingData) {
			// build data encodings so the inner loops can be efficient
			Encoding<F, L> encoding = buildEncoding(trainingData);
			IndexLinearizer indexLinearizer = buildIndexLinearizer(encoding);
			EncodedDatum[] data = encodeData(trainingData, encoding);
			// build the objective function for this data
			ObjectiveFunction<F, L> objective = new ObjectiveFunction<F, L>(
					encoding, data, indexLinearizer, average);
			// learn our voting weights
			double[] weights = objective.perceptron();
			// build a classifier using these weights (and the data encodings)
			return new _drr342PerceptronClassifier<I, F, L>(weights, encoding,
					indexLinearizer, featureExtractor);
		}

		private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
			return new IndexLinearizer(encoding.getNumFeatures(),
					encoding.getNumLabels());
		}

		private Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) {
			Indexer<F> featureIndexer = new Indexer<F>();
			Indexer<L> labelIndexer = new Indexer<L>();
			for (LabeledInstance<I, L> labeledInstance : data) {
				L label = labeledInstance.getLabel();
				Counter<F> features = featureExtractor
						.extractFeatures(labeledInstance.getInput());
				LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
						label, features);
				labelIndexer.add(labeledDatum.getLabel());
				for (F feature : labeledDatum.getFeatures().keySet()) {
					featureIndexer.add(feature);
				}
			}
			return new Encoding<F, L>(featureIndexer, labelIndexer);
		}

		private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
				Encoding<F, L> encoding) {
			EncodedDatum[] encodedData = new EncodedDatum[data.size()];
			for (int i = 0; i < data.size(); i++) {
				LabeledInstance<I, L> labeledInstance = data.get(i);
				L label = labeledInstance.getLabel();			
				Counter<F> features = featureExtractor
						.extractFeatures(labeledInstance.getInput());				
				LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
						label, features);
				encodedData[i] = EncodedDatum.encodeLabeledDatum(
						labeledFeatureVector, encoding);
			}
			return encodedData;
		}

		public Factory(boolean average,
				FeatureExtractor<I, F> featureExtractor) {
			this.average = average;
			this.featureExtractor = featureExtractor;
		}
	}



	/**
	 * This is the MaximumEntropy objective function: the (negative) log
	 * conditional likelihood of the training data, possibly with a penalty for
	 * large weights. Note that this objective get MINIMIZED so it's the
	 * negative of the objective we normally think of.
	 */
	public static class ObjectiveFunction<F, L> {
		IndexLinearizer indexLinearizer;
		Encoding<F, L> encoding;
		EncodedDatum[] data;
		boolean average;

		public int dimension() {
			return indexLinearizer.getNumLinearIndexes();
		}

		/**
		 * The most important part of the classifier learning process! This
		 * method determines, for the given weight vector x, what the (negative)
		 * log conditional likelihood of the data is, as well as the derivatives
		 * of that likelihood wrt each weight parameter.
		 * @param average 
		 */
		
		// Perceptron Model
		private double[] perceptron() {
			double w[] = DoubleArrays.constantArray(0.0, dimension());
			double[] avgW = DoubleArrays.constantArray(0.0, dimension());
			double score, maxScore;
			int argMax, wIndex, count = 0;
			
			// RANDOMIZE DATA
			Collections.shuffle(Arrays.asList(data));
			// CALCULATE WEIGHTS
			for (EncodedDatum datum : data) {
				maxScore = Double.NEGATIVE_INFINITY;
				argMax = 0;
				for (int i = 0; i < encoding.getNumLabels(); i++) {
					score = 0.0;
					for (int j = 0; j < datum.getNumActiveFeatures(); j++) {
						wIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(j), i);
						score += w[wIndex] * datum.getFeatureCount(j);
					}
					if (score > maxScore) {
						maxScore = score;
						argMax = i;
					}
				}
				if (argMax != datum.getLabelIndex()) {
					for (int i = 0; i < datum.getNumActiveFeatures(); i++) {
						wIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(i), argMax);
						w[wIndex] -= datum.getFeatureIndex(i);
						wIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(i), datum.getLabelIndex());
						w[wIndex] += datum.getFeatureIndex(i);
					}
				}
				for (int i = 0; i < avgW.length; i++) {
					avgW[i] = (avgW[i] * count) + w[i];
					avgW[i] /= (count + 1);
				}
				count++;
			}
			return (average) ? avgW : w;
		}	

		public ObjectiveFunction(Encoding<F, L> encoding, EncodedDatum[] data,
				IndexLinearizer indexLinearizer, boolean average) {
			this.indexLinearizer = indexLinearizer;
			this.encoding = encoding;
			this.data = data;
			this.average = average;
		}
	}

	/**
	 * EncodedDatums are sparse representations of (labeled) feature count
	 * vectors for a given data point. Use getNumActiveFeatures() to see how
	 * many features have non-zero count in a datum. Then, use getFeatureIndex()
	 * and getFeatureCount() to retreive the number and count of each non-zero
	 * feature. Use getLabelIndex() to get the label's number.
	 */
	public static class EncodedDatum {

		public static <F, L> EncodedDatum encodeDatum(
				FeatureVector<F> featureVector, Encoding<F, L> encoding) {
			Counter<F> features = featureVector.getFeatures();
			Counter<F> knownFeatures = new Counter<F>();
			for (F feature : features.keySet()) {
				if (encoding.getFeatureIndex(feature) < 0)
					continue;
				knownFeatures.incrementCount(feature,
						features.getCount(feature));
			}
			int numActiveFeatures = knownFeatures.keySet().size();
			int[] featureIndexes = new int[numActiveFeatures];
			double[] featureCounts = new double[knownFeatures.keySet().size()];
			int i = 0;
			for (F feature : knownFeatures.keySet()) {
				int index = encoding.getFeatureIndex(feature);
				double count = knownFeatures.getCount(feature);
				featureIndexes[i] = index;
				featureCounts[i] = count;
				i++;
			}
			EncodedDatum encodedDatum = new EncodedDatum(-1, featureIndexes,
					featureCounts);
			return encodedDatum;
		}

		public static <F, L> EncodedDatum encodeLabeledDatum(
				LabeledFeatureVector<F, L> labeledDatum, Encoding<F, L> encoding) {
			EncodedDatum encodedDatum = encodeDatum(labeledDatum, encoding);
			encodedDatum.labelIndex = encoding.getLabelIndex(labeledDatum
					.getLabel());
			return encodedDatum;
		}

		int labelIndex;
		int[] featureIndexes;
		double[] featureCounts;

		public int getLabelIndex() {
			return labelIndex;
		}

		public int getNumActiveFeatures() {
			return featureCounts.length;
		}

		public int getFeatureIndex(int num) {
			return featureIndexes[num];
		}

		public double getFeatureCount(int num) {
			return featureCounts[num];
		}

		public EncodedDatum(int labelIndex, int[] featureIndexes,
				double[] featureCounts) {
			this.labelIndex = labelIndex;
			this.featureIndexes = featureIndexes;
			this.featureCounts = featureCounts;
		}
	}

	/**
	 * The Encoding maintains correspondences between the various representions
	 * of the data, labels, and features. The external representations of labels
	 * and features are object-based. The functions getLabelIndex() and
	 * getFeatureIndex() can be used to translate those objects to integer
	 * representatiosn: numbers between 0 and getNumLabels() or getNumFeatures()
	 * (exclusive). The inverses of this map are the getLabel() and getFeature()
	 * functions.
	 */
	public static class Encoding<F, L> {
		Indexer<F> featureIndexer;
		Indexer<L> labelIndexer;

		public int getNumFeatures() {
			return featureIndexer.size();
		}

		public int getFeatureIndex(F feature) {
			return featureIndexer.indexOf(feature);
		}

		public F getFeature(int featureIndex) {
			return featureIndexer.get(featureIndex);
		}

		public int getNumLabels() {
			return labelIndexer.size();
		}

		public int getLabelIndex(L label) {
			return labelIndexer.indexOf(label);
		}

		public L getLabel(int labelIndex) {
			return labelIndexer.get(labelIndex);
		}

		public Encoding(Indexer<F> featureIndexer, Indexer<L> labelIndexer) {
			this.featureIndexer = featureIndexer;
			this.labelIndexer = labelIndexer;
		}
	}

	/**
	 * The IndexLinearizer maintains the linearization of the two-dimensional
	 * features-by-labels pair space. This is because, while we might think
	 * about lambdas and derivatives as being indexed by a feature-label pair,
	 * the optimization code expects one long vector for lambdas and
	 * derivatives. To go from a pair featureIndex, labelIndex to a single
	 * pairIndex, use getLinearIndex().
	 */
	public static class IndexLinearizer {
		int numFeatures;
		int numLabels;

		public int getNumLinearIndexes() {
			return numFeatures * numLabels;
		}

		public int getLinearIndex(int featureIndex, int labelIndex) {
			return labelIndex + featureIndex * numLabels;
		}

		public int getFeatureIndex(int linearIndex) {
			return linearIndex / numLabels;
		}

		public int getLabelIndex(int linearIndex) {
			return linearIndex % numLabels;
		}

		public IndexLinearizer(int numFeatures, int numLabels) {
			this.numFeatures = numFeatures;
			this.numLabels = numLabels;
		}
	}
	
	private double[] weights;
	private Encoding<F, L> encoding;
	private IndexLinearizer indexLinearizer;
	private FeatureExtractor<I, F> featureExtractor;

	/**
	 * Calculate the log probabilities of each class, for the given datum
	 * (feature bundle). Note that the weighted votes (refered to as
	 * activations) are *almost* log probabilities, but need to be normalized.
	 */
	
	private static <F, L> double[] getScores(EncodedDatum datum, double[] weights, Encoding<F, L> encoding,
			IndexLinearizer indexLinearizer) {
		// TODO: apply the classifier to this feature vector
		int wIndex;
		double sum, max = 1;
		double[] scores = DoubleArrays.constantArray(Double.NEGATIVE_INFINITY, encoding.getNumLabels());
		for (int i = 0; i < scores.length; i++) {
			sum = 0.0;
			for (int j = 0; j < datum.getNumActiveFeatures(); j++) {
				wIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(j), i);
				sum += weights[wIndex] * datum.getFeatureCount(j);
			}
			scores[i] = sum;
			if (scores[i] > max) max = scores[i];
		}
		
		// scale to (0,1)
		for (int i = 0; i < scores.length; i++) {
			scores[i] /= max;
		}
		return scores;
	}	

	public Counter<L> getProbabilities(I input) {
		FeatureVector<F> featureVector = new BasicFeatureVector<F>(
				featureExtractor.extractFeatures(input));
		return getProbabilities(featureVector);
	}

	private Counter<L> getProbabilities(FeatureVector<F> featureVector) {
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		double[] scores = getScores(encodedDatum, weights,
				encoding, indexLinearizer);
		return scoresToProbabiltyCounter(scores);
	}

	private Counter<L> scoresToProbabiltyCounter(
			double[] scores) {
		Counter<L> probabiltyCounter = new Counter<L>();
		for (int labelIndex = 0; labelIndex < scores.length; labelIndex++) {
			double probability = scores[labelIndex];
			L label = encoding.getLabel(labelIndex);
			probabiltyCounter.setCount(label, probability);
		}
		return probabiltyCounter;
	}

	public L getLabel(I input) {
		return getProbabilities(input).argMax();
	}

	public _drr342PerceptronClassifier(double[] weights, Encoding<F, L> encoding,
			IndexLinearizer indexLinearizer,
			FeatureExtractor<I, F> featureExtractor) {
		this.weights = weights;
		this.encoding = encoding;
		this.indexLinearizer = indexLinearizer;
		this.featureExtractor = featureExtractor;
	}

	public static void main(String[] args) {
		// create datums
		LabeledInstance<String[], String> datum1 = new LabeledInstance<String[], String>(
				"cat", new String[] { "fuzzy", "claws", "small" });
		LabeledInstance<String[], String> datum2 = new LabeledInstance<String[], String>(
				"bear", new String[] { "fuzzy", "claws", "big" });
		LabeledInstance<String[], String> datum3 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "medium" });
		LabeledInstance<String[], String> datum4 = new LabeledInstance<String[], String>(
				"cat", new String[] { "claws", "small" });

		// create training set
		List<LabeledInstance<String[], String>> trainingData = new ArrayList<LabeledInstance<String[], String>>();
		trainingData.add(datum1);
		trainingData.add(datum2);
		trainingData.add(datum3);

		// create test set
		List<LabeledInstance<String[], String>> testData = new ArrayList<LabeledInstance<String[], String>>();
		testData.add(datum4);

		// build classifier
		FeatureExtractor<String[], String> featureExtractor = new FeatureExtractor<String[], String>() {
			public Counter<String> extractFeatures(String[] featureArray) {
				return new Counter<String>(Arrays.asList(featureArray));
			}
		};
		_drr342PerceptronClassifier.Factory<String[], String, String> perceptronClassifierFactory = new _drr342PerceptronClassifier.Factory<String[], String, String>(
				true, featureExtractor);
		ProbabilisticClassifier<String[], String> perceptronClassifier = perceptronClassifierFactory
				.trainClassifier(trainingData);
		System.out.println("Scores on test instance: "
				+ perceptronClassifier.getProbabilities(datum4.getInput()));
	}
}

