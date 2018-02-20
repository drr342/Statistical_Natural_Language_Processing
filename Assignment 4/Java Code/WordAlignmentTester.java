package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import nlp.io.IOUtils;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Pair;

/**
 * Harness for testing word-level alignments. The code is hard-wired for the
 * alignment source to be English and the alignment target to be French (recall
 * that's the direction for translating INTO English in the noisy channel
 * model).
 *
 * Your projects will implement several methods of word-to-word alignment.
 */
public class WordAlignmentTester {

	static final String ENGLISH_EXTENSION = "e";
	static final String FRENCH_EXTENSION = "f";
	static final String NULL = "<NULL>"; // NULL = English Null Word, to be appended to all English sentences at
											// position zero.
	static boolean useEuroparl = false; // specify if the europarl corpus is being used or not.

	/**
	 * A holder for a pair of sentences, each a list of strings. Sentences in the
	 * test sets have integer IDs, as well, which are used to retreive the gold
	 * standard alignments for those sentences.
	 */
	public static class SentencePair {
		int sentenceID;
		String sourceFile;
		List<String> englishWords;
		List<String> frenchWords;

		public int getSentenceID() {
			return sentenceID;
		}

		public String getSourceFile() {
			return sourceFile;
		}

		public List<String> getEnglishWords() {
			return englishWords;
		}

		public List<String> getFrenchWords() {
			return frenchWords;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
				String englishWord = englishWords.get(englishPosition);
				sb.append(englishPosition);
				sb.append(":");
				sb.append(englishWord);
				sb.append(" ");
			}
			sb.append("\n");
			for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
				String frenchWord = frenchWords.get(frenchPosition);
				sb.append(frenchPosition);
				sb.append(":");
				sb.append(frenchWord);
				sb.append(" ");
			}
			sb.append("\n");
			return sb.toString();
		}

		public SentencePair(int sentenceID, String sourceFile, List<String> englishWords, List<String> frenchWords) {
			this.sentenceID = sentenceID;
			this.sourceFile = sourceFile;
			this.englishWords = englishWords;
			this.frenchWords = frenchWords;
		}
	}

	/**
	 * Alignments serve two purposes, both to indicate your system's guessed
	 * alignment, and to hold the gold standard alignments. Alignments map index
	 * pairs to one of three values, unaligned, possibly aligned, and surely
	 * aligned. Your alignment guesses should only contain sure and unaligned pairs,
	 * but the gold alignments contain possible pairs as well.
	 *
	 * To build an alignment, start with an empty one and use
	 * addAlignment(i,j,true). To display one, use the render method.
	 */
	public static class Alignment {
		Set<Pair<Integer, Integer>> sureAlignments;
		Set<Pair<Integer, Integer>> possibleAlignments;

		public boolean containsSureAlignment(int englishPosition, int frenchPosition) {
			return sureAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
		}

		public boolean containsPossibleAlignment(int englishPosition, int frenchPosition) {
			return possibleAlignments.contains(new Pair<Integer, Integer>(englishPosition, frenchPosition));
		}

		public void addAlignment(int englishPosition, int frenchPosition, boolean sure) {
			Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(englishPosition, frenchPosition);
			if (sure)
				sureAlignments.add(alignment);
			possibleAlignments.add(alignment);
		}

		public Alignment() {
			sureAlignments = new HashSet<Pair<Integer, Integer>>();
			possibleAlignments = new HashSet<Pair<Integer, Integer>>();
		}

		public static String render(Alignment alignment, SentencePair sentencePair) {
			return render(alignment, alignment, sentencePair);
		}

		public static String render(Alignment reference, Alignment proposed, SentencePair sentencePair) {
			StringBuilder sb = new StringBuilder();
			for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords()
						.size(); englishPosition++) {
					boolean sure = reference.containsSureAlignment(englishPosition, frenchPosition);
					boolean possible = reference.containsPossibleAlignment(englishPosition, frenchPosition);
					char proposedChar = ' ';
					if (proposed.containsSureAlignment(englishPosition, frenchPosition))
						proposedChar = '#';
					if (sure) {
						sb.append('[');
						sb.append(proposedChar);
						sb.append(']');
					} else {
						if (possible) {
							sb.append('(');
							sb.append(proposedChar);
							sb.append(')');
						} else {
							sb.append(' ');
							sb.append(proposedChar);
							sb.append(' ');
						}
					}
				}
				sb.append("| ");
				sb.append(sentencePair.getFrenchWords().get(frenchPosition));
				sb.append('\n');
			}
			for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords().size(); englishPosition++) {
				sb.append("---");
			}
			sb.append("'\n");
			boolean printed = true;
			int index = 0;
			while (printed) {
				printed = false;
				StringBuilder lineSB = new StringBuilder();
				for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords()
						.size(); englishPosition++) {
					String englishWord = sentencePair.getEnglishWords().get(englishPosition);
					if (englishWord.length() > index) {
						printed = true;
						lineSB.append(' ');
						lineSB.append(englishWord.charAt(index));
						lineSB.append(' ');
					} else {
						lineSB.append("   ");
					}
				}
				index += 1;
				if (printed) {
					sb.append(lineSB);
					sb.append('\n');
				}
			}
			return sb.toString();
		}
	}

	/**
	 * WordAligners have one method: alignSentencePair, which takes a sentence pair
	 * and produces an alignment which specifies an english source for each french
	 * word which is not aligned to "null". Explicit alignment to position -1 is
	 * equivalent to alignment to "null".
	 */
	static interface WordAligner {
		Alignment alignSentencePair(SentencePair sentencePair);
	}

	/**
	 * Simple alignment baseline which maps french positions to english positions.
	 * If the french sentence is longer, all final word map to null.
	 */
	static class BaselineWordAligner implements WordAligner {
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			for (int frenchPosition = 0; frenchPosition < numFrenchWords; frenchPosition++) {
				int englishPosition = frenchPosition;
				if (englishPosition >= numEnglishWords)
					englishPosition = -1;
				alignment.addAlignment(englishPosition, frenchPosition, true);
			}
			return alignment;
		}
	}

	// PART 1 OF ASSIGNMENT DEFINITION
	/**
	 * 
	 * Heuristic aligner based on absolute counts in the training corpus. Uses
	 * c(e,f) / (c(e) * c(f)) as a score for each english-french pair. For pairs not
	 * observed during training, aligns the french word to the english null word.
	 *
	 */
	static class HeuristicWordAligner implements WordAligner {
		private List<SentencePair> sentencePairs;
		private CounterMap<String, String> counterMapFrenchEnglish = new CounterMap<>();
		private Counter<String> counterFrench = new Counter<>();
		private Counter<String> counterEnglish = new Counter<>();

		public HeuristicWordAligner(List<SentencePair> sentencePairs) {
			this.sentencePairs = sentencePairs;
			getCounters();
		}

		// Calculates counters c(e, f), c(e) and c(f) from training data encoded in
		// sentencePairs.
		public void getCounters() {
			for (SentencePair sentencePair : sentencePairs) {
				sentencePair.getEnglishWords().add(0, NULL);
				for (String frenchWord : sentencePair.getFrenchWords()) {
					counterFrench.incrementCount(frenchWord, 1.0);
				}
				for (String englishWord : sentencePair.getEnglishWords()) {
					counterEnglish.incrementCount(englishWord, 1.0);
				}
				int n = Math.min(sentencePair.getFrenchWords().size(), sentencePair.getEnglishWords().size() - 1);
				for (int i = 0; i < n; i++) {
					counterMapFrenchEnglish.incrementCount(sentencePair.getFrenchWords().get(i),
							sentencePair.getEnglishWords().get(i + 1), 1.0);
				}
			}
		}

		// Returns the alignment for the given sentencePair. French words unseen during
		// training are aligned to the English null word.
		public Alignment alignSentencePair(SentencePair sentencePair) {
			if (sentencePair.getEnglishWords().get(0).equals(NULL))
				sentencePair.getEnglishWords().remove(0);
			Alignment alignment = new Alignment();
			CounterMap<Integer, Integer> score = new CounterMap<>();
			int m = sentencePair.getFrenchWords().size();
			int l = sentencePair.getEnglishWords().size();
			String frenchWord, englishWord;
			for (int i = 0; i < m; i++) {
				frenchWord = sentencePair.getFrenchWords().get(i);
				for (int j = 0; j < l; j++) {
					englishWord = sentencePair.getEnglishWords().get(j);
					score.setCount(i, j, counterMapFrenchEnglish.getCount(frenchWord, englishWord)
							/ (counterFrench.getCount(frenchWord) * counterEnglish.getCount(englishWord)));
				}
			}
			for (int frenchPosition = 0; frenchPosition < m; frenchPosition++) {
				int englishPosition;
				if (!Double.isNaN(score.getCount(frenchPosition, score.getCounter(frenchPosition).argMax())))
					englishPosition = score.getCounter(frenchPosition).argMax();
				else
					englishPosition = -1;
				alignment.addAlignment(englishPosition, frenchPosition, true);
			}
			return alignment;
		}
	}

	// PART 2 OF ASSIGNMENT DEFINITION
	/**
	 * Implementation of the IBM-Model 1 aligner.
	 */
	static class IBM1WordAligner implements WordAligner {
		private final int MAX_ITERATIONS; // number of iterations for the EM algorithm.
		private List<SentencePair> sentencePairs;
		private CounterMap<String, String> t = new CounterMap<>();
		private CounterMap<String, String> counterMapEnglishFrench = new CounterMap<>();
		private Counter<String> counterEnglish = new Counter<>();

		// Constructor to be used in the initialization of the IBM-Model 2.
		public IBM1WordAligner(List<SentencePair> sentencePairs, int maxIterations) {
			this.MAX_ITERATIONS = maxIterations;
			this.sentencePairs = sentencePairs;
			initialize();
			EM();
		}

		// Constructor to be used for the IBM-Model 1 in stand-alone mode.
		public IBM1WordAligner(List<SentencePair> sentencePairs) {
			this(sentencePairs, 20);
		}

		// Method to pass the initialization t parameters to IBM-Model 2.
		public CounterMap<String, String> getT() {
			return this.t;
		}

		// Initialize t parameters to random values.
		public void initialize() {
			for (SentencePair sentencePair : sentencePairs) {
				sentencePair.getEnglishWords().add(0, NULL);
				for (String frenchWord : sentencePair.getFrenchWords()) {
					for (String englishWord : sentencePair.getEnglishWords()) {
						t.setCount(frenchWord, englishWord, Math.random());
					}
				}
			}
		}

		// EM algorithm.
		public void EM() {
			// Loop for parameters convergence.
			for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
				// Set counts to zero.
				for (String keyE : counterMapEnglishFrench.keySet()) {
					for (String keyF : counterMapEnglishFrench.getCounter(keyE).keySet()) {
						counterMapEnglishFrench.setCount(keyE, keyF, 0.0);
					}
				}
				for (String keyE : counterEnglish.keySet()) {
					counterEnglish.setCount(keyE, 0.0);
				}
				// counterMapEnglishFrench = new CounterMap<>();
				// counterEnglish = new Counter<>();
				// Loop all training sentences (k).
				for (SentencePair sentencePair : sentencePairs) {
					// Loop french words f_1 ... f_mk
					for (String frenchWord : sentencePair.getFrenchWords()) {
						// Loop english words e_0 ... e_lk
						for (String englishWord : sentencePair.getEnglishWords()) {
							// Update delta and expected counts c(e, f) and c(e). (E step).
							double delta = t.getCount(frenchWord, englishWord) / t.getCounter(frenchWord).totalCount();
							counterMapEnglishFrench.incrementCount(englishWord, frenchWord, delta);
							counterEnglish.incrementCount(englishWord, delta);
						}
					}
				}
				// Update t parameters (M step).
				double newT;
				for (String englishWord : counterEnglish.keySet()) {
					for (String frenchWord : counterMapEnglishFrench.getCounter(englishWord).keySet()) {
						newT = counterMapEnglishFrench.getCount(englishWord, frenchWord)
								/ counterEnglish.getCount(englishWord);
						// change += Math.abs(t.getCount(frenchWord, englishWord) - newT);
						t.setCount(frenchWord, englishWord, newT);
					}
				}
			}
		}

		/*
		 * Returns the alignment for the given sentencePair. As with the heuristic
		 * algorithm, all french words unseen during training are mapped to the english
		 * null word. If the sentence was used during training, the NULL is removed so
		 * it wont show in the verbose output.
		 */
		public Alignment alignSentencePair(SentencePair sentencePair) {
			if (sentencePair.getEnglishWords().get(0).equals(NULL))
				sentencePair.getEnglishWords().remove(0);
			Alignment alignment = new Alignment();
			int m = sentencePair.getFrenchWords().size();
			int l = sentencePair.getEnglishWords().size();
			for (int i = 0; i < m; i++) {
				String frenchWord = sentencePair.getFrenchWords().get(i);
				double max = t.getCount(frenchWord, NULL);
				int argmax = -1;
				for (int j = 0; j < l; j++) {
					String englishWord = sentencePair.getEnglishWords().get(j);
					if (t.containsKey(frenchWord)) {
						if (t.getCount(frenchWord, englishWord) > max) {
							max = t.getCount(frenchWord, englishWord);
							argmax = j;
						}
					}
				}
				alignment.addAlignment(argmax, i, true);
			}
			return alignment;
		}
	}

	// PART 2 AND 3 OF ASSIGNMENT DEFINITION
	/**
	 * Implementation of the IBM-Model 2 aligner.
	 */
	static class IBM2WordAligner implements WordAligner {
		private final int MAX_ITERATIONS = 20; // number of iterations for the EM algorithm.
		private List<SentencePair> sentencePairs;
		private CounterMap<String, String> t = new CounterMap<>();
		private CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> q = new CounterMap<>();
		private CounterMap<String, String> counterMapEnglishFrench = new CounterMap<>();
		private Counter<String> counterEnglish = new Counter<>();
		private CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> counterMapJI = new CounterMap<>();
		private Counter<Pair<Pair<Integer, Integer>, Integer>> counterI = new Counter<>();

		// Constructor
		public IBM2WordAligner(List<SentencePair> sentencePairs) {
			this.sentencePairs = sentencePairs;
			initialize();
			EM();
		}

		/*
		 * Initiailization routine: 1) Runs IBM-Model 1 with 20 iterations and uses the
		 * resulting t parameters as a first approximation. 2) Initializes the q
		 * parameters randomly.
		 */
		public void initialize() {
			IBM1WordAligner ibm1 = new IBM1WordAligner(sentencePairs, 20);
			t = ibm1.getT();
			for (SentencePair sentencePair : sentencePairs) {
				int m = sentencePair.getFrenchWords().size();
				int l = sentencePair.getEnglishWords().size();
				Pair<Integer, Integer> lm = new Pair<>(l, m);
				for (int i = 0; i < m; i++) {
					Pair<Pair<Integer, Integer>, Integer> lmi = new Pair<>(lm, i);
					for (int j = 0; j < l; j++) {
						q.setCount(lmi, j - 1, Math.random());
					}
				}
			}
		}

		// EM algorithm.
		public void EM() {
			// Loop for parameters convergence.
			for (int iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
				// Set counts to zero.
				for (String keyE : counterMapEnglishFrench.keySet()) {
					for (String keyF : counterMapEnglishFrench.getCounter(keyE).keySet()) {
						counterMapEnglishFrench.setCount(keyE, keyF, 0.0);
					}
				}
				for (String keyE : counterEnglish.keySet()) {
					counterEnglish.setCount(keyE, 0.0);
				}
				for (Pair<Pair<Integer, Integer>, Integer> keyE : counterMapJI.keySet()) {
					for (Integer keyF : counterMapJI.getCounter(keyE).keySet()) {
						counterMapJI.setCount(keyE, keyF, 0.0);
					}
				}
				for (Pair<Pair<Integer, Integer>, Integer> keyI : counterI.keySet()) {
					counterI.setCount(keyI, 0.0);
				}
				// counterMapEnglishFrench = new CounterMap<>();
				// counterEnglish = new Counter<>();
				// counterMapJI = new CounterMap<>();
				// counterI = new Counter<>();
				// Loop all training sentences (k).
				for (SentencePair sentencePair : sentencePairs) {
					int m = sentencePair.getFrenchWords().size();
					int l = sentencePair.getEnglishWords().size();
					Pair<Integer, Integer> lm = new Pair<>(l, m);
					// Loop french words f_1 ... f_mk
					for (int i = 0; i < m; i++) {
						Pair<Pair<Integer, Integer>, Integer> lmi = new Pair<>(lm, i);
						String frenchWord = sentencePair.getFrenchWords().get(i);
						double sum = 0.0;
						// Calculate the denominator (sum) for the delta parameter.
						for (int j = 0; j < l; j++) {
							String englishWord = sentencePair.getEnglishWords().get(j);
							sum += q.getCount(lmi, j - 1) * t.getCount(frenchWord, englishWord);
						}
						// Loop english words e_0 ... e_lk
						for (int j = 0; j < l; j++) {
							String englishWord = sentencePair.getEnglishWords().get(j);
							// Update delta and expected counts c(e, f), c(e), c(j|i, l, m) and c(i, l, m).
							// (E step).
							double delta = q.getCount(lmi, j - 1) * t.getCount(frenchWord, englishWord) / sum;
							counterMapEnglishFrench.incrementCount(englishWord, frenchWord, delta);
							counterEnglish.incrementCount(englishWord, delta);
							counterMapJI.incrementCount(lmi, j - 1, delta);
							counterI.incrementCount(lmi, delta);
						}
					}
				}

				// Update t and q parameters (M step).
				double newT, newQ;
				for (String englishWord : counterEnglish.keySet()) {
					for (String frenchWord : counterMapEnglishFrench.getCounter(englishWord).keySet()) {
						newT = counterMapEnglishFrench.getCount(englishWord, frenchWord)
								/ counterEnglish.getCount(englishWord);
						t.setCount(frenchWord, englishWord, newT);
					}
				}
				for (Pair<Pair<Integer, Integer>, Integer> lmi : counterMapJI.keySet()) {
					for (Integer j : counterMapJI.getCounter(lmi).keySet()) {
						newQ = counterMapJI.getCount(lmi, j) / counterI.getCount(lmi);
						q.setCount(lmi, j, newQ);
					}
				}
			}
		}

		/*
		 * Returns the alignment for the given sentencePair. As with the previous
		 * algorithms, all french words unseen during training are mapped to the english
		 * null word. If the sentence was used during training, the NULL is removed so
		 * it wont show in the verbose output.
		 */
		public Alignment alignSentencePair(SentencePair sentencePair) {
			if (sentencePair.getEnglishWords().get(0).equals(NULL))
				sentencePair.getEnglishWords().remove(0);
			Alignment alignment = new Alignment();
			int m = sentencePair.getFrenchWords().size();
			int l = sentencePair.getEnglishWords().size();
			Pair<Integer, Integer> lm = new Pair<>(l + 1, m);
			for (int i = 0; i < m; i++) {
				String frenchWord = sentencePair.getFrenchWords().get(i);
				Pair<Pair<Integer, Integer>, Integer> lmi = new Pair<>(lm, i);
				double max = t.getCount(frenchWord, NULL);
				int argmax = -1;
				if (t.containsKey(frenchWord) && q.containsKey(lmi)) {
					for (int j = 0; j < l; j++) {
						String englishWord = sentencePair.getEnglishWords().get(j);
						if (t.getCount(frenchWord, englishWord) * q.getCount(lmi, j) > max) {
							max = t.getCount(frenchWord, englishWord) * q.getCount(lmi, j);
							argmax = j;
						}
					}
				}
				// for (String key : t.keySet()) {
				// for (Entry<String, Double> entry : t.getCounter(key).getEntrySet()) {
				// System.out.println(key + ": " + entry.getKey() + "-" + entry.getValue());
				// }
				// }
				alignment.addAlignment(argmax, i, true);
			}
			return alignment;
		}
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = "data4";
		int maxTrainingSentences = 0;
		boolean verbose = false;
		String dataset = "minitest";
		String model = "model2"; // ibm_model1 //model2 //baseline //heuristic

		// Update defaults using command line specifications
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
			System.out.println("Using base path: " + basePath);
		}
		if (argMap.containsKey("-sentences")) {
			maxTrainingSentences = Integer.parseInt(argMap.get("-sentences"));
			System.out.println("Using an additional " + maxTrainingSentences + " training sentences.");
		}
		if (argMap.containsKey("-data")) {
			dataset = argMap.get("-data");
			System.out.println("Running with data: " + dataset);
		} else {
			System.out.println("No data set specified.  Use -data [miniTest, validate, test].");
		}
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
			System.out.println("Running with model: " + model);
		} else {
			System.out.println("No model specified.  Use -model modelname.");
		}
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		// use the option -europarl at running time to include sentences from the
		// europarl
		// corpus in the training data set.
		if (argMap.containsKey("-europarl")) {
			useEuroparl = true;
		}

		// Read appropriate training and testing sets.
		List<SentencePair> trainingSentencePairs = new ArrayList<SentencePair>();
		if (!dataset.equals("miniTest") && maxTrainingSentences > 0)
			trainingSentencePairs = readSentencePairs(basePath + "/training", maxTrainingSentences);
		List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
		Map<Integer, Alignment> testAlignments = new HashMap<Integer, Alignment>();
		if (dataset.equalsIgnoreCase("validate")) {
			testSentencePairs = readSentencePairs(basePath + "/trial", Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/trial/trial.wa");
		} else if (dataset.equalsIgnoreCase("miniTest")) {
			testSentencePairs = readSentencePairs(basePath + "/mini", Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/mini/mini.wa");
		} else {
			throw new RuntimeException("Bad data set mode: " + dataset + ", use validate or miniTest.");
		}
		trainingSentencePairs.addAll(testSentencePairs);

		// Build model
		WordAligner wordAligner = null;
		switch (model.toLowerCase()) {
		case "baseline":
			wordAligner = new BaselineWordAligner();
			break;
		case "heuristic":
			wordAligner = new HeuristicWordAligner(trainingSentencePairs);
			break;
		case "ibm_model1":
			wordAligner = new IBM1WordAligner(trainingSentencePairs);
			break;
		case "model2":
			wordAligner = new IBM2WordAligner(trainingSentencePairs);
			break;
		default:
			break;
		}

		// Test model
		test(wordAligner, testSentencePairs, testAlignments, verbose);
		// Generate file for submission
		testSentencePairs = readSentencePairs(basePath + "/test", Integer.MAX_VALUE);
		predict(wordAligner, testSentencePairs, basePath + "/" + model + maxTrainingSentences + ".out");
	}

	private static void test(WordAligner wordAligner, List<SentencePair> testSentencePairs,
			Map<Integer, Alignment> testAlignments, boolean verbose) {
		int proposedSureCount = 0;
		int proposedPossibleCount = 0;
		int sureCount = 0;
		int proposedCount = 0;
		for (SentencePair sentencePair : testSentencePairs) {
			Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
			Alignment referenceAlignment = testAlignments.get(sentencePair.getSentenceID());
			if (referenceAlignment == null)
				throw new RuntimeException(
						"No reference alignment found for sentenceID " + sentencePair.getSentenceID());
			if (verbose)
				System.out.println(
						"Alignment:\n" + Alignment.render(referenceAlignment, proposedAlignment, sentencePair));
			for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords()
						.size(); englishPosition++) {
					boolean proposed = proposedAlignment.containsSureAlignment(englishPosition, frenchPosition);
					boolean sure = referenceAlignment.containsSureAlignment(englishPosition, frenchPosition);
					boolean possible = referenceAlignment.containsPossibleAlignment(englishPosition, frenchPosition);
					if (proposed && sure)
						proposedSureCount += 1;
					if (proposed && possible)
						proposedPossibleCount += 1;
					if (proposed)
						proposedCount += 1;
					if (sure)
						sureCount += 1;
				}
			}
		}
		System.out.println("Precision: " + proposedPossibleCount / (double) proposedCount);
		System.out.println("Recall: " + proposedSureCount / (double) sureCount);
		System.out.println(
				"AER: " + (1.0 - (proposedSureCount + proposedPossibleCount) / (double) (sureCount + proposedCount)));
	}

	private static void predict(WordAligner wordAligner, List<SentencePair> testSentencePairs, String path)
			throws IOException {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		for (SentencePair sentencePair : testSentencePairs) {
			// sentencePair.getEnglishWords().add(0, NULL);
			Alignment proposedAlignment = wordAligner.alignSentencePair(sentencePair);
			for (int frenchPosition = 0; frenchPosition < sentencePair.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair.getEnglishWords()
						.size(); englishPosition++) {
					if (proposedAlignment.containsSureAlignment(englishPosition, frenchPosition)) {
						writer.write(frenchPosition + "-" + englishPosition + " ");
					}
				}
			}
			writer.write("\n");
		}
		writer.close();
	}

	// BELOW HERE IS IO CODE

	private static Map<Integer, Alignment> readAlignments(String fileName) {
		Map<Integer, Alignment> alignments = new HashMap<Integer, Alignment>();
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(fileName));
			while (in.ready()) {
				String line = in.readLine();
				String[] words = line.split("\\s+");
				if (words.length != 4)
					throw new RuntimeException("Bad alignment file " + fileName + ", bad line was " + line);
				Integer sentenceID = Integer.parseInt(words[0]);
				Integer englishPosition = Integer.parseInt(words[1]) - 1;
				Integer frenchPosition = Integer.parseInt(words[2]) - 1;
				String type = words[3];
				Alignment alignment = alignments.get(sentenceID);
				if (alignment == null) {
					alignment = new Alignment();
					alignments.put(sentenceID, alignment);
				}
				alignment.addAlignment(englishPosition, frenchPosition, type.equals("S"));
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		} finally {
			try {
				in.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return alignments;
	}

	private static List<SentencePair> readSentencePairs(String path, int maxSentencePairs) {
		// if europarl corpus is being used and we are reading training sentences,
		// duplicate the size of the sentencePairs list.
		boolean duplicate = path.endsWith("training") && useEuroparl;
		int sentencePairsSize = duplicate ? 2 * maxSentencePairs : maxSentencePairs;
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		List<String> baseFileNames = getBaseFileNames(path);
		for (String baseFileName : baseFileNames) {
			// if the current file corresponds to the europarl corpus but it is not being
			// used,
			// continue to the next file without reading anything.
			if (!useEuroparl && baseFileName.startsWith("europarl"))
				continue;
			if (sentencePairs.size() >= sentencePairsSize)
				continue;
			// the maxSentencePairs parameter has to be passed to this method, in order to
			// stop reading sentences in the long europarl files in case they are being
			// used.
			sentencePairs.addAll(readSentencePairs2(baseFileName, maxSentencePairs));
		}
		return sentencePairs;
	}

	private static List<SentencePair> readSentencePairs2(String baseFileName, int maxSentencePairs) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		String englishFileName = baseFileName + "." + ENGLISH_EXTENSION;
		String frenchFileName = baseFileName + "." + FRENCH_EXTENSION;
		BufferedReader englishIn = null;
		BufferedReader frenchIn = null;
		try {
			englishIn = new BufferedReader(new FileReader(englishFileName));
			frenchIn = new BufferedReader(new FileReader(frenchFileName));
			while (englishIn.ready() && frenchIn.ready()) {
				String englishLine = englishIn.readLine();
				String frenchLine = frenchIn.readLine();
				Pair<Integer, List<String>> englishSentenceAndID = readSentence(englishLine);
				Pair<Integer, List<String>> frenchSentenceAndID = readSentence(frenchLine);
				if (!englishSentenceAndID.getFirst().equals(frenchSentenceAndID.getFirst()))
					throw new RuntimeException("Sentence ID confusion in file " + baseFileName + ", lines were:\n\t"
							+ englishLine + "\n\t" + frenchLine);
				sentencePairs.add(new SentencePair(englishSentenceAndID.getFirst(), baseFileName,
						englishSentenceAndID.getSecond(), frenchSentenceAndID.getSecond()));
				// if the size of the list sentencePairs list has reached the desired limit,
				// exit the while loop and return to the readSentencePairs method.
				if (sentencePairs.size() == maxSentencePairs)
					break;
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		} finally {
			try {
				englishIn.close();
				frenchIn.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}
		return sentencePairs;
	}

	private static Pair<Integer, List<String>> readSentence(String line) {
		int id = -1;
		List<String> words = new ArrayList<String>();
		String[] tokens = line.split("\\s+");
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("<s"))
				continue;
			if (token.equals("</s>"))
				continue;
			if (token.startsWith("snum=")) {
				String idString = token.substring(5, token.length() - 1);
				id = Integer.parseInt(idString);
				continue;
			}
			words.add(token.intern());
		}
		return new Pair<Integer, List<String>>(id, words);
	}

	private static List<String> getBaseFileNames(String path) {
		List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
			public boolean accept(File pathname) {
				if (pathname.isDirectory())
					return true;
				String name = pathname.getName();
				return name.endsWith(ENGLISH_EXTENSION);
			}
		});
		List<String> baseFileNames = new ArrayList<String>();
		for (File englishFile : englishFiles) {
			String baseFileName = chop(englishFile.getAbsolutePath(), "." + ENGLISH_EXTENSION);
			baseFileNames.add(baseFileName);
		}
		return baseFileNames;
	}

	private static String chop(String name, String extension) {
		if (!name.endsWith(extension))
			return name;
		return name.substring(0, name.length() - extension.length());
	}

}
