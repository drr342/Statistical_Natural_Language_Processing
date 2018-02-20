package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.BoundedList;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class _drr342ProperNameTester {

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			
//			System.out.println(name);
			
			char[] characters = name.toCharArray();
			
//			System.out.println(Arrays.toString(characters));
			
			Counter<String> features = new Counter<String>();
			// add character unigram features
//			for (int i = 0; i < characters.length; i++) {
//				char character = characters[i];
//				features.incrementCount("UNI-" + character, 1.0);
//			}
			// TODO : extract better features!
			
			List<String> cList = new ArrayList<>();
			for (int i = 0; i < characters.length; i++) {
				cList.add(String.valueOf(characters[i]));
			}
			BoundedList<String> bString = new BoundedList<>(cList, "<S>", "<E>");
			
			// 2-gram
			for (int i = 0; i < characters.length + 1; i++) {
				features.incrementCount("BI-" + bString.get(i - 1)
						+ bString.get(i), 1.0);
			}
			// 3-gram
			for (int i = 0; i < characters.length + 2; i++) {
				features.incrementCount("TRI-" + bString.get(i - 2)
						+ bString.get(i - 1) + bString.get(i), 1.0);
			}
			// 4-gram
			for (int i = 0; i < characters.length + 3; i++) {
				features.incrementCount("QUA-" + bString.get(i - 3)
						+ bString.get(i - 2) + bString.get(i - 1)
						+ bString.get(i), 1.0);
			}
			
			// Words
			for (String word : name.toLowerCase().split("[\\s\\-]")) {
				features.incrementCount("WORD-" + word, 1.0);
			}
			// Number of words
			features.incrementCount("WORDS-" + name.split("[\\s\\-]").length, 1.0);		
			// One word
			if (name.split("[\\s\\-]").length == 1) features.incrementCount("1WORD", 1.0);
									
			Pattern p;
			Matcher m;
	
			// Corporations
			p = Pattern.compile("Inc\\.|Co\\.|Co\\.?$|Corp\\.|Corp\\.?$|Corporation");
			m = p.matcher(name);
			if (m.find()) features.incrementCount("CORP", 10.0);
			// Starts with number
			p = Pattern.compile("^[^a-zA-Z]");
			m = p.matcher(name);
			if (m.find()) features.incrementCount("START_NUMBER", 1.0);
			// Starts with Article
			p = Pattern.compile("^(The|A|El|Le|La|Les|Los|Las|Un|Una|Unos|Unas|Des)\\s");
			m = p.matcher(name);
			if (m.find()) features.incrementCount("START_ARTICLE", 1.0);
			// Drugs endings
			p = Pattern.compile("(ane|x|yl|ate|id|il|in|ol|ine)(\\b|$|-)|\\d$");
			m = p.matcher(name);
			if (m.find()) features.incrementCount("MEDS", 1.0);
			
			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		reader.close();
		return labeledInstances;
	}

	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose, boolean useValidation) throws IOException {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		FileWriter fw;
		String str = (useValidation) ? "output_dev.txt" : "output_test.txt";
		fw = new FileWriter(str);
		PrintWriter pw = new PrintWriter(fw);
		for (LabeledInstance<String, String> testDatum : testData) {
			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			if (label.equals(testDatum.getLabel())) {
				numCorrect += 1.0;
			}
//			System.out.println(label + " : " + testDatum.getLabel());
			if (verbose) {
				// display an error
				System.err.println("Example: " + name + " guess=" + label + " gold=" + testDatum.getLabel()
						+ " confidence=" + confidence);
			}
			pw.printf("Example: %s guess=%s gold=%s confidence=%f%n", name, label, testDatum.getLabel(), confidence);
			numTotal += 1.0;
		}
		pw.close();
		fw.close();
		double accuracy = numCorrect / numTotal;
		// comment when using verbose
		System.out.println("Accuracy: " + accuracy);
	}
	
	private enum Models {
		MOST_FREQUENT("baseline"),
		MAXIMUM_ENTROPY("maxent"),
		PERCEPTRON("perceptron");
		
		private final String model;
		private Models (String model) {
			this.model = model;
		}
		public String toString() {
			return this.model;
		}
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
//		C:\\Users\\drr34\\eclipse-workspace\\java\\StatisticalNLP_Fall17_Assignment2\\data2
		String basePath = "data2";
		Models model = Models.MAXIMUM_ENTROPY;
		String testString = "";
		boolean verbose = false;
		boolean useValidation = true;
		boolean useAverage = true;
		int iterations = 100;
		double sigma = 1.0;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = Models.valueOf(argMap.get("-model"));
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		ProbabilisticClassifierFactory<String, String> factory = null;
		switch (model) {
		case MAXIMUM_ENTROPY:
			 factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					sigma, iterations, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
			break;
		case MOST_FREQUENT:
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
				.trainClassifier(trainingData);
			break;
		case PERCEPTRON:
			factory = new _drr342PerceptronClassifier.Factory<String, String, String>(
					useAverage, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
			break;
		default:
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test classifier
		testClassifier(classifier, (useValidation ? validationData : testData),
				verbose, useValidation);
	}
}
