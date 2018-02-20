package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import nlp.classify.FeatureExtractor;
import nlp.classify.LabeledInstance;
import nlp.classify.ProbabilisticClassifier;
import nlp.classify.ProbabilisticClassifierFactory;
import nlp.util.BoundedList;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.CounterMap;
import nlp.util.Counters;
import nlp.util.Interner;
import nlp.util.Pair;

/**
 * Harness for POS Tagger project.
 */
public class _drr342POSTaggerTester {

	static final String START_WORD = "<S>";
	static final String STOP_WORD = "</S>";
	static final String START_TAG = "<S>";
	static final String STOP_TAG = "<S>";
	static final double UNKNOWN_THRESHOLD = 5.0;

	/**
	 * Tagged sentences are a bundling of a list of words and a list of their tags.
	 */
	static class TaggedSentence {
		List<String> words;
		List<String> tags;

		public int size() {
			return words.size();
		}

		public List<String> getWords() {
			return words;
		}

		public List<String> getTags() {
			return tags;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int position = 0; position < words.size(); position++) {
				String word = words.get(position);
				String tag = tags.get(position);
				sb.append(word);
				sb.append("_");
				sb.append(tag);
			}
			return sb.toString();
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof TaggedSentence))
				return false;

			final TaggedSentence taggedSentence = (TaggedSentence) o;

			if (tags != null ? !tags.equals(taggedSentence.tags) : taggedSentence.tags != null)
				return false;
			if (words != null ? !words.equals(taggedSentence.words) : taggedSentence.words != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (words != null ? words.hashCode() : 0);
			result = 29 * result + (tags != null ? tags.hashCode() : 0);
			return result;
		}

		public TaggedSentence(List<String> words, List<String> tags) {
			this.words = words;
			this.tags = tags;
		}
	}

	/**
	 * States are pairs of tags along with a position index, representing the two
	 * tags preceding that position. So, the START state, which can be gotten by
	 * State.getStartState() is [START, START, 0]. To build an arbitrary state, for
	 * example [DT, NN, 2], use the static factory method State.buildState("DT",
	 * "NN", 2). There isn't a single final state, since sentences lengths vary, so
	 * State.getEndState(i) takes a parameter for the length of the sentence.
	 */
	static class State {

		private static transient Interner<State> stateInterner = new Interner<State>(
				new Interner.CanonicalFactory<State>() {
					public State build(State state) {
						return new State(state);
					}
				});

		private static transient State tempState = new State();

		public static State getStartState() {
			return buildState(START_TAG, START_TAG, 0);
		}

		public static State getStopState(int position) {
			return buildState(STOP_TAG, STOP_TAG, position);
		}

		public static State buildState(String previousPreviousTag, String previousTag, int position) {
			tempState.setState(previousPreviousTag, previousTag, position);
			return stateInterner.intern(tempState);
		}

		public static List<String> toTagList(List<State> states) {
			List<String> tags = new ArrayList<String>();
			if (states.size() > 0) {
				tags.add(states.get(0).getPreviousPreviousTag());
				for (State state : states) {
					tags.add(state.getPreviousTag());
				}
			}
			return tags;
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public State getNextState(String tag) {
			return State.buildState(getPreviousTag(), tag, getPosition() + 1);
		}

		public State getPreviousState(String tag) {
			return State.buildState(tag, getPreviousPreviousTag(), getPosition() - 1);
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof State))
				return false;

			final State state = (State) o;

			if (position != state.position)
				return false;
			if (previousPreviousTag != null ? !previousPreviousTag.equals(state.previousPreviousTag)
					: state.previousPreviousTag != null)
				return false;
			if (previousTag != null ? !previousTag.equals(state.previousTag) : state.previousTag != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = position;
			result = 29 * result + (previousTag != null ? previousTag.hashCode() : 0);
			result = 29 * result + (previousPreviousTag != null ? previousPreviousTag.hashCode() : 0);
			return result;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getPosition() + "]";
		}

		int position;
		String previousTag;
		String previousPreviousTag;

		private void setState(String previousPreviousTag, String previousTag, int position) {
			this.previousPreviousTag = previousPreviousTag;
			this.previousTag = previousTag;
			this.position = position;
		}

		private State() {
		}

		private State(State state) {
			setState(state.getPreviousPreviousTag(), state.getPreviousTag(), state.getPosition());
		}
	}

	/**
	 * A Trellis is a graph with a start state an an end state, along with successor
	 * and predecessor functions.
	 */
	static class Trellis<S> {
		S startState;
		S endState;
		CounterMap<S, S> forwardTransitions;
		CounterMap<S, S> backwardTransitions;

		/**
		 * Get the unique start state for this trellis.
		 */
		public S getStartState() {
			return startState;
		}

		public void setStartState(S startState) {
			this.startState = startState;
		}

		/**
		 * Get the unique end state for this trellis.
		 */
		public S getEndState() {
			return endState;
		}

		public void setStopState(S endState) {
			this.endState = endState;
		}

		/**
		 * For a given state, returns a counter over what states can be next in the
		 * markov process, along with the cost of that transition. Caution: a state not
		 * in the counter is illegal, and should be considered to have cost
		 * Double.NEGATIVE_INFINITY, but Counters score items they don't contain as 0.
		 */
		public Counter<S> getForwardTransitions(S state) {
			return forwardTransitions.getCounter(state);

		}

		/**
		 * For a given state, returns a counter over what states can precede it in the
		 * markov process, along with the cost of that transition.
		 */
		public Counter<S> getBackwardTransitions(S state) {
			return backwardTransitions.getCounter(state);
		}

		public void setTransitionCount(S start, S end, double count) {
			forwardTransitions.setCount(start, end, count);
			backwardTransitions.setCount(end, start, count);
		}

		public Trellis() {
			forwardTransitions = new CounterMap<S, S>();
			backwardTransitions = new CounterMap<S, S>();
		}
	}

	/**
	 * A TrellisDecoder takes a Trellis and returns a path through that trellis in
	 * which the first item is trellis.getStartState(), the last is
	 * trellis.getEndState(), and each pair of states is conntected in the trellis.
	 */
	// static interface TrellisDecoder {
	// List<State> getBestPath(Trellis<State> trellis);
	// }
	static interface TrellisDecoder<S> {
		List<S> getBestPath(Trellis<S> trellis);
	}

	static class GreedyDecoder<S> implements TrellisDecoder<S> {
		public List<S> getBestPath(Trellis<S> trellis) {
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getStartState();
			states.add(currentState);
			while (!currentState.equals(trellis.getEndState())) {
				Counter<S> transitions = trellis.getForwardTransitions(currentState);
				S nextState = transitions.argMax();
				states.add(nextState);
				currentState = nextState;
			}
			return states;
		}
	}

	static class ViterbiDecoder<S> implements TrellisDecoder<S> {

		private Pair<Counter<S>, Map<S, S>> updateValues(List<Pair<Pair<S, S>, Double>> temp) {
			Counter<S> pi = new Counter<>();
			Map<S, S> bp = new HashMap<>();
			for (Pair<Pair<S, S>, Double> pair : temp) {
				if (!pi.containsKey(pair.getFirst().getSecond())
						|| (pair.getSecond() > pi.getCount(pair.getFirst().getSecond()))) {
					pi.setCount(pair.getFirst().getSecond(), pair.getSecond());
					bp.put(pair.getFirst().getSecond(), pair.getFirst().getFirst());
				}
			}
			return new Pair<Counter<S>, Map<S, S>>(pi, bp);
		}

		public List<S> getBestPath(Trellis<S> trellis) {
			Counter<S> pi = new Counter<>();
			List<Map<S, S>> bp = new ArrayList<>();
			pi.setCount(trellis.getStartState(), 0.0);

			while (true) {
				Counter<S> wCounter = pi;
				List<Pair<Pair<S, S>, Double>> temp = new ArrayList<>();
				for (S u : wCounter.keySet()) {
					Counter<S> uCounter = trellis.getForwardTransitions(u);
					for (S v : uCounter.keySet()) {
						temp.add(new Pair<Pair<S, S>, Double>(new Pair<S, S>(u, v),
								wCounter.getCount(u) + uCounter.getCount(v)));
					}
				}
				Pair<Counter<S>, Map<S, S>> update = updateValues(temp);
				if (update.getFirst().isEmpty())
					break;
				pi = update.getFirst();
				bp.add(update.getSecond());
			}

			List<S> bestPath = new ArrayList<>();
			bestPath.add(pi.keySet().iterator().next());
			for (int i = bp.size() - 1, j = 0; i >= 0; i--, j++) {
				bestPath.add(bp.get(i).get(bestPath.get(j)));
			}

			Collections.reverse(bestPath);
			return bestPath;
		}
	}

	static class POSTagger {

		LocalTrigramScorer localTrigramScorer;
		TrellisDecoder<State> trellisDecoder;

		// chop up the training instances into local contexts and pass them on
		// to the local scorer.
		public void train(List<TaggedSentence> taggedSentences) {
			localTrigramScorer.train(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		// chop up the validation instances into local contexts and pass them on
		// to the local scorer.
		public void validate(List<TaggedSentence> taggedSentences) {
			localTrigramScorer.validate(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				List<TaggedSentence> taggedSentences) {
			List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			for (TaggedSentence taggedSentence : taggedSentences) {
				localTrigramContexts.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
			}
			return localTrigramContexts;
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(TaggedSentence taggedSentence) {
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			List<String> words = new BoundedList<String>(taggedSentence.getWords(), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(taggedSentence.getTags(), START_TAG, STOP_TAG);
			for (int position = 0; position <= taggedSentence.size() + 1; position++) {
				labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(words, position, tags.get(position - 2),
						tags.get(position - 1), tags.get(position)));
			}
			return labeledLocalTrigramContexts;
		}

		/**
		 * Builds a Trellis over a sentence, by starting at the state State, and
		 * advancing through all legal extensions of each state already in the trellis.
		 * You should not have to modify this code (or even read it, really).
		 */
		private Trellis<State> buildTrellis(List<String> sentence) {
			Trellis<State> trellis = new Trellis<State>();
			trellis.setStartState(State.getStartState());
			State stopState = State.getStopState(sentence.size() + 2);
			trellis.setStopState(stopState);
			Set<State> states = Collections.singleton(State.getStartState());
			for (int position = 0; position <= sentence.size() + 1; position++) {
				Set<State> nextStates = new HashSet<State>();
				for (State state : states) {
					if (state.equals(stopState))
						continue;
					LocalTrigramContext localTrigramContext = new LocalTrigramContext(sentence, position,
							state.getPreviousPreviousTag(), state.getPreviousTag());
					Counter<String> tagScores = localTrigramScorer.getLogScoreCounter(localTrigramContext);
					for (String tag : tagScores.keySet()) {
						double score = tagScores.getCount(tag);
						State nextState = state.getNextState(tag);
						trellis.setTransitionCount(state, nextState, score);
						nextStates.add(nextState);
					}
				}
				// System.out.println("States: " + nextStates);
				states = nextStates;
			}
			return trellis;
		}

		// to tag a sentence: build its trellis and find a path through that
		// trellis
		public List<String> tag(List<String> sentence) {
			Trellis<State> trellis = buildTrellis(sentence);
			List<State> states = trellisDecoder.getBestPath(trellis);
			List<String> tags = State.toTagList(states);
			tags = stripBoundaryTags(tags);
			return tags;
		}

		/**
		 * Scores a tagging for a sentence. Note that a tag sequence not accepted by the
		 * markov process should receive a log score of Double.NEGATIVE_INFINITY.
		 */
		public double scoreTagging(TaggedSentence taggedSentence) {
			double logScore = 0.0;
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(
					taggedSentence);
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				Counter<String> logScoreCounter = localTrigramScorer.getLogScoreCounter(labeledLocalTrigramContext);
				String currentTag = labeledLocalTrigramContext.getCurrentTag();
				if (logScoreCounter.containsKey(currentTag)) {
					logScore += logScoreCounter.getCount(currentTag);
				} else {
					logScore += Double.NEGATIVE_INFINITY;
				}
			}
			return logScore;
		}

		private List<String> stripBoundaryTags(List<String> tags) {
			return tags.subList(2, tags.size() - 2);
		}

		public POSTagger(LocalTrigramScorer localTrigramScorer, TrellisDecoder<State> trellisDecoder) {
			this.localTrigramScorer = localTrigramScorer;
			this.trellisDecoder = trellisDecoder;
		}
	}

	/**
	 * A LocalTrigramContext is a position in a sentence, along with the previous
	 * two tags -- basically a FeatureVector.
	 */
	static class LocalTrigramContext {
		List<String> words;
		int position;
		String previousTag;
		String previousPreviousTag;

		public List<String> getWords() {
			return words;
		}

		public String getCurrentWord() {
			return words.get(position);
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "]";
		}

		public LocalTrigramContext(List<String> words, int position, String previousPreviousTag, String previousTag) {
			this.words = words;
			this.position = position;
			this.previousTag = previousTag;
			this.previousPreviousTag = previousPreviousTag;
		}
	}

	/**
	 * A LabeledLocalTrigramContext is a context plus the correct tag for that
	 * position -- basically a LabeledFeatureVector
	 */
	static class LabeledLocalTrigramContext extends LocalTrigramContext {
		String currentTag;

		public String getCurrentTag() {
			return currentTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag() + ", " + getCurrentWord() + "_"
					+ getCurrentTag() + "]";
		}

		public LabeledLocalTrigramContext(List<String> words, int position, String previousPreviousTag,
				String previousTag, String currentTag) {
			super(words, position, previousPreviousTag, previousTag);
			this.currentTag = currentTag;
		}
	}

	/**
	 * LocalTrigramScorers assign scores to tags occuring in specific
	 * LocalTrigramContexts.
	 */
	static interface LocalTrigramScorer {
		/**
		 * The Counter returned should contain log probabilities, meaning if all values
		 * are exponentiated and summed, they should sum to one. For efficiency, the
		 * Counter can contain only the tags which occur in the given context with
		 * non-zero model probability.
		 */
		Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext);

		void train(List<LabeledLocalTrigramContext> localTrigramContexts);

		void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
	}

	private static class TagFeatureExtractor implements FeatureExtractor<String, String> {

		public Counter<String> extractFeatures(String word) {

			char[] characters = word.toCharArray();
			Counter<String> features = new Counter<String>();

			List<String> cList = new ArrayList<>();
			for (int i = 0; i < characters.length; i++) {
				cList.add(String.valueOf(characters[i]));
			}
			BoundedList<String> bString = new BoundedList<>(cList, "<S>", "<E>");
			// 2-gram
			for (int i = 0; i < characters.length + 1; i++) {
				features.incrementCount("BI-" + bString.get(i - 1) + bString.get(i), 1.0);
			}
			// 3-gram
			for (int i = 0; i < characters.length + 2; i++) {
				features.incrementCount("TRI-" + bString.get(i - 2) + bString.get(i - 1) + bString.get(i), 1.0);
			}
			// 4-gram
			for (int i = 0; i < characters.length + 3; i++) {
				features.incrementCount(
						"QUA-" + bString.get(i - 3) + bString.get(i - 2) + bString.get(i - 1) + bString.get(i), 1.0);
			}
			// // 5-gram
			// for (int i = 0; i < characters.length + 4; i++) {
			// features.incrementCount(
			// "5-" + bString.get(i - 4) + bString.get(i - 3) + bString.get(i - 2) +
			// bString.get(i - 1) + bString.get(i), 1.0);
			// }

			return features;
		}
	}

	private static class UnknownWordsClassifier {

		static ProbabilisticClassifier<String, String> classifier;

		private static List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				List<TaggedSentence> taggedSentences) {
			List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			for (TaggedSentence taggedSentence : taggedSentences) {
				localTrigramContexts.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
			}
			return localTrigramContexts;
		}

		private static List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				TaggedSentence taggedSentence) {
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			List<String> words = new BoundedList<String>(taggedSentence.getWords(), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(taggedSentence.getTags(), START_TAG, STOP_TAG);
			for (int position = 0; position <= taggedSentence.size() + 1; position++) {
				labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(words, position, tags.get(position - 2),
						tags.get(position - 1), tags.get(position)));
			}
			return labeledLocalTrigramContexts;
		}

		private static CounterMap<String, String> train(List<TaggedSentence> taggedSentences, int iterations,
				double sigma) {
			System.out.println("Setting up Unkown Words Classifier...");
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(
					taggedSentences);
			String word, tag;
			CounterMap<String, String> wordsToTags = new CounterMap<>();
			// collect word-tag counts
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				word = labeledLocalTrigramContext.getCurrentWord();
				tag = labeledLocalTrigramContext.getCurrentTag();
				wordsToTags.incrementCount(word, tag, 1.0);
			}
			List<LabeledInstance<String, String>> labeledInstances = new ArrayList<>();
			for (String instance : wordsToTags.keySet()) {
				if (wordsToTags.getCounter(instance).totalCount() <= UNKNOWN_THRESHOLD) {
					String label = wordsToTags.getCounter(instance).argMax();
					LabeledInstance<String, String> labeledInstance = new LabeledInstance<>(label, instance);
					labeledInstances.add(labeledInstance);
				}
			}

			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					sigma, iterations, new TagFeatureExtractor());
			classifier = factory.trainClassifier(labeledInstances);
			return wordsToTags;
		}
	}

	static class TrigramHMMTagScorer implements LocalTrigramScorer {

		boolean restrictTrigrams; // if true, assign log score of
									// Double.NEGATIVE_INFINITY to illegal tag
									// trigrams.
		CounterMap<String, String> wordsToTags;
		CounterMap<String, String> emission = new CounterMap<>();
		CounterMap<String, String> transition = new CounterMap<>();
		Counter<String> unknownWordTags = new Counter<String>();
		Set<String> seenTagTrigrams = new HashSet<String>();
		// ProbabilisticClassifier<String, String> classifier;

		public int getHistorySize() {
			return 2;
		}

		public Counter<String> getLogScoreCounter(LocalTrigramContext localTrigramContext) {
			Counter<String> tagCounter = getTagCounter(localTrigramContext);
			Set<String> allowedFollowingTags = allowedFollowingTags(tagCounter.keySet(),
					localTrigramContext.getPreviousPreviousTag(), localTrigramContext.getPreviousTag());
			Counter<String> logScoreCounter = new Counter<String>();
			for (String tag : tagCounter.keySet()) {
				double logScore = Math.log(tagCounter.getCount(tag));
				if (!restrictTrigrams || allowedFollowingTags.isEmpty() || allowedFollowingTags.contains(tag))
					logScoreCounter.setCount(tag, logScore);
			}
			return logScoreCounter;
		}

		private Set<String> allowedFollowingTags(Set<String> tags, String previousPreviousTag, String previousTag) {
			Set<String> allowedTags = new HashSet<String>();
			for (String tag : tags) {
				String trigramString = makeTrigramString(previousPreviousTag, previousTag, tag);
				if (seenTagTrigrams.contains((trigramString))) {
					allowedTags.add(tag);
				}
			}
			return allowedTags;
		}

		private String makeTrigramString(String previousPreviousTag, String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}

		private Counter<String> getTagCounter(LocalTrigramContext localTrigramContext) {
			String word = localTrigramContext.getCurrentWord();
			String previousTag = localTrigramContext.getPreviousTag();
			String prePreviousTag = localTrigramContext.getPreviousPreviousTag();
			Counter<String> tagCounter = new Counter<>();
			double count;
			if (!wordsToTags.keySet().contains(word)) 
				word = UnknownWordsClassifier.classifier.getLabel(word);
			for (String tag : wordsToTags.getCounter(word).keySet()) {
				count = emission.getCount(tag, word) * transition.getCount(tag, prePreviousTag + "_" + previousTag);
				tagCounter.setCount(tag, count);
			}
			return tagCounter;
		}

		public void train(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			String word, tag, previousTag, prePreviousTag;
			Counter<String> unigramTags = new Counter<>();
			CounterMap<String, String> bigramTags = new CounterMap<>();
			CounterMap<String, String> trigramTags = new CounterMap<>();
			CounterMap<String, String> unknownEmissions = new CounterMap<>();

			// collect word-tag counts
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				word = labeledLocalTrigramContext.getCurrentWord();
				tag = labeledLocalTrigramContext.getCurrentTag();
				if (wordsToTags.getCounter(word).totalCount() <= UNKNOWN_THRESHOLD) {
					word = UnknownWordsClassifier.classifier.getLabel(word);
					unknownEmissions.incrementCount(tag, word, 1.0);
				}
			}

			double total = unknownEmissions.totalCount();
			double previous;
			for (String label : unknownEmissions.keySet()) {
				for (String key : unknownEmissions.getCounter(label).keySet()) {
					previous = unknownEmissions.getCounter(label).getCount(key);
					unknownEmissions.getCounter(label).setCount(key, previous / total);
				}
			}

			CounterMap<String, String> tempMap = wordsToTags;
			wordsToTags = new CounterMap<>();
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				word = labeledLocalTrigramContext.getCurrentWord();
				tag = labeledLocalTrigramContext.getCurrentTag();
				previousTag = labeledLocalTrigramContext.getPreviousTag();
				prePreviousTag = labeledLocalTrigramContext.getPreviousPreviousTag();
				if (tempMap.getCounter(word).totalCount() <= UNKNOWN_THRESHOLD) {
					word = UnknownWordsClassifier.classifier.getLabel(word);
					emission.incrementCount(tag, word, unknownEmissions.getCount(tag, word));
				} else {
					emission.incrementCount(tag, word, 1.0);
				}
				wordsToTags.incrementCount(word, tag, 1.0);
				unigramTags.incrementCount(tag, 1.0);
				bigramTags.incrementCount(tag, previousTag, 1.0);
				trigramTags.incrementCount(tag, prePreviousTag + "_" + previousTag, 1.0);
			}
			emission = Counters.conditionalNormalize(emission);

			// DELETED INTERPOLATION
			double[] lambda = new double[3];
			Arrays.fill(lambda, 0.0);
			double cTri, cBi, cUni, normCount;

			for (String key : trigramTags.keySet()) {
				cUni = (unigramTags.getCount(key) - 1.0) / (unigramTags.totalCount() - 1.0);
				for (Entry<String, Double> entry : trigramTags.getCounter(key).getEntrySet()) {
					String[] temp = entry.getKey().split("_");
					cTri = bigramTags.getCount(temp[1], temp[0]) - 1.0;
					cTri = (cTri == 0.0) ? 0.0 : (entry.getValue() - 1.0) / cTri;
					cBi = unigramTags.getCount(temp[1]) - 1.0;
					cBi = (cBi == 0.0) ? 0.0 : (bigramTags.getCount(key, temp[1]) - 1.0) / cBi;
					if (cTri > cBi && cTri > cUni) {
						lambda[2] += entry.getValue();
					} else if (cBi > cTri && cBi > cUni) {
						lambda[1] += entry.getValue();
					} else {
						lambda[0] += entry.getValue();
					}
					normCount = entry.getValue() / bigramTags.getCount(temp[1], temp[0]);
					trigramTags.setCount(key, entry.getKey(), normCount);
				}
			}
			for (String key : bigramTags.keySet()) {
				for (Map.Entry<String, Double> entry : bigramTags.getCounter(key).getEntrySet()) {
					normCount = entry.getValue() / unigramTags.getCount(entry.getKey());
					bigramTags.setCount(key, entry.getKey(), normCount);
				}
			}
			unigramTags = Counters.normalize(unigramTags);

			double sum = Arrays.stream(lambda).sum();
			for (int i = 0; i < lambda.length; i++) {
				lambda[i] /= sum;
			}

			for (String currentTag : emission.keySet()) {
				for (String preTag : emission.keySet()) {
					for (String prePreTag : emission.keySet()) {
						transition.setCount(currentTag, prePreTag + "_" + preTag,
								lambda[2] * trigramTags.getCount(currentTag, prePreTag + "_" + preTag)
										+ lambda[1] * bigramTags.getCount(currentTag, preTag)
										+ lambda[0] * unigramTags.getCount(currentTag));
					}
				}
			}
		}

		public void validate(List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// no tuning for this dummy model!
		}

		public TrigramHMMTagScorer(CounterMap<String, String> wordsToTags, boolean restrictTrigrams) {
			this.restrictTrigrams = restrictTrigrams;
			this.wordsToTags = wordsToTags;
		}
	}

	private static List<TaggedSentence> readTaggedSentences(String path, boolean hasTags) throws Exception {
		List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
		BufferedReader reader = new BufferedReader(new FileReader(path));
		String line = "";
		List<String> words = new LinkedList<String>();
		List<String> tags = new LinkedList<String>();
		while ((line = reader.readLine()) != null) {
			if (line.equals("")) {
				taggedSentences.add(new TaggedSentence(new BoundedList<String>(words, START_WORD, STOP_WORD),
						new BoundedList<String>(tags, START_WORD, STOP_WORD)));
				words = new LinkedList<String>();
				tags = new LinkedList<String>();
			} else {
				String[] fields = line.split("\\s+");
				words.add(fields[0]);
				tags.add(hasTags ? fields[1] : "");
			}
		}
		reader.close();
		System.out.println("Read " + taggedSentences.size() + " sentences.");
		return taggedSentences;
	}

	private static void labelTestSet(POSTagger posTagger, List<TaggedSentence> testSentences, String path)
			throws Exception {
		BufferedWriter writer = new BufferedWriter(new FileWriter(path));
		for (TaggedSentence sentence : testSentences) {
			List<String> words = sentence.getWords();
			List<String> guessedTags = posTagger.tag(words);
			for (int i = 0; i < words.size(); i++) {
				writer.write(words.get(i) + "\t" + guessedTags.get(i) + "\n");
			}
			writer.write("\n");
		}
		writer.close();
	}

	private static void evaluateTagger(POSTagger posTagger, List<TaggedSentence> taggedSentences,
			Set<String> trainingVocabulary, boolean verbose, String path) throws Exception {
		double numTags = 0.0;
		double numTagsCorrect = 0.0;
		double numUnknownWords = 0.0;
		double numUnknownWordsCorrect = 0.0;
		int numDecodingInversions = 0;
		BufferedWriter writer = new BufferedWriter(new FileWriter(path, true));
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			List<String> goldTags = taggedSentence.getTags();
			List<String> guessedTags = posTagger.tag(words);
			for (int position = 0; position < words.size() - 1; position++) {
				String word = words.get(position);
				String goldTag = goldTags.get(position);
				String guessedTag = guessedTags.get(position);
				if (guessedTag.equals(goldTag))
					numTagsCorrect += 1.0;
				numTags += 1.0;
				if (!trainingVocabulary.contains(word)) {
					if (guessedTag.equals(goldTag))
						numUnknownWordsCorrect += 1.0;
					numUnknownWords += 1.0;
				}
			}
			double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
			double scoreOfGuessedTagging = posTagger.scoreTagging(new TaggedSentence(words, guessedTags));
			if (scoreOfGoldTagging > scoreOfGuessedTagging) {
				numDecodingInversions++;
				if (verbose) {
					System.out.println(
							"WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
					writer.write(
							"WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.\n");
				}
			}
			if (verbose) {
				System.out.println(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
				writer.write(alignedTaggings(words, goldTags, guessedTags, true) + "\n");
			}
		}

		System.out.println("Tag Accuracy: " + (numTagsCorrect / numTags) + " (Unknown Accuracy: "
				+ (numUnknownWordsCorrect / numUnknownWords) + ")  Decoder Suboptimalities Detected: "
				+ numDecodingInversions);
		writer.write("Tag Accuracy: " + (numTagsCorrect / numTags) + " (Unknown Accuracy: "
				+ (numUnknownWordsCorrect / numUnknownWords) + ")  Decoder Suboptimalities Detected: "
				+ numDecodingInversions + "\n");
		writer.close();
	}

	// pretty-print a pair of taggings for a sentence, possibly suppressing the
	// tags which correctly match
	private static String alignedTaggings(List<String> words, List<String> goldTags, List<String> guessedTags,
			boolean suppressCorrectTags) {
		StringBuilder goldSB = new StringBuilder("Gold Tags: ");
		StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
		StringBuilder wordSB = new StringBuilder("Words: ");
		for (int position = 0; position < words.size(); position++) {
			equalizeLengths(wordSB, goldSB, guessedSB);
			String word = words.get(position);
			String gold = goldTags.get(position);
			String guessed = guessedTags.get(position);
			wordSB.append(word);
			if (position < words.size() - 1)
				wordSB.append(' ');
			boolean correct = (gold.equals(guessed));
			if (correct && suppressCorrectTags)
				continue;
			guessedSB.append(guessed);
			goldSB.append(gold);
		}
		return goldSB + "\n" + guessedSB + "\n" + wordSB;
	}

	private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2, StringBuilder sb3) {
		int maxLength = sb1.length();
		maxLength = Math.max(maxLength, sb2.length());
		maxLength = Math.max(maxLength, sb3.length());
		ensureLength(sb1, maxLength);
		ensureLength(sb2, maxLength);
		ensureLength(sb3, maxLength);
	}

	private static void ensureLength(StringBuilder sb, int length) {
		while (sb.length() < length) {
			sb.append(' ');
		}
	}

	private static Set<String> extractVocabulary(List<TaggedSentence> taggedSentences) {
		Set<String> vocabulary = new HashSet<String>();
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			vocabulary.addAll(words);
		}
		return vocabulary;
	}

	// private static boolean isCapitalized(String word) {
	// return word.substring(0, 1).matches("[A-Z]");
	// }

	public static void main(String[] args) throws Exception {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = "data3";
		boolean verbose = true;

		// Update defaults using command line specifications
		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// Whether or not to print the individual errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read in data
		System.out.print("Loading training sentences...");
		List<TaggedSentence> trainTaggedSentences = readTaggedSentences(basePath + "/en-wsj-train.pos", true);
		Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
		System.out.println("done.");
		System.out.print("Loading in-domain dev sentences...");
		List<TaggedSentence> devInTaggedSentences = readTaggedSentences(basePath + "/en-wsj-dev.pos", true);
		System.out.println("done.");
		System.out.print("Loading out-of-domain dev sentences...");
		List<TaggedSentence> devOutTaggedSentences = readTaggedSentences(basePath + "/en-web-weblogs-dev.pos", true);
		System.out.println("done.");
		System.out.print("Loading out-of-domain blind test sentences...");
		List<TaggedSentence> testSentences = readTaggedSentences(basePath + "/en-web-test.blind", false);
		System.out.println("done.");

		// Train unknown words classifier
		CounterMap<String, String> uwcViterbi = new CounterMap<>();
		CounterMap<String, String> uwcGreedy = new CounterMap<>();
		uwcViterbi = UnknownWordsClassifier.train(trainTaggedSentences, 100, 0.3);
		for (String key : uwcViterbi.keySet()) {
			for (Entry<String, Double> entry : uwcViterbi.getCounter(key).getEntrySet()) {
				uwcGreedy.setCount(key, entry.getKey(), entry.getValue());
			}
		}

		CounterMap<String, String> wordsToTagsViterbi = uwcViterbi;
		// Construct tagger components
		LocalTrigramScorer localTrigramScorerViterbi = new TrigramHMMTagScorer(wordsToTagsViterbi, false);
		TrellisDecoder<State> viterbiDecoder = new ViterbiDecoder<State>();
		// Train VITERBI tagger
		POSTagger posTaggerViterbi = new POSTagger(localTrigramScorerViterbi, viterbiDecoder);
		posTaggerViterbi.train(trainTaggedSentences);
		// Test taggers
		System.out.println("Evaluating VITERBI on in-domain data:.");
		evaluateTagger(posTaggerViterbi, devInTaggedSentences, trainingVocabulary, verbose,
				basePath + "/cline_output_viterbi_in.txt");
		System.out.println("Evaluating VITERBI on out-of-domain data:.");
		evaluateTagger(posTaggerViterbi, devOutTaggedSentences, trainingVocabulary, verbose,
				basePath + "/cline_output_viterbi_out.txt");

		CounterMap<String, String> wordsToTagsGreedy = uwcGreedy;
		LocalTrigramScorer localTrigramScorerGreedy = new TrigramHMMTagScorer(wordsToTagsGreedy, false);
		TrellisDecoder<State> greedyDecoder = new GreedyDecoder<State>();
		// Train GREEDY tagger
		POSTagger posTaggerGreedy = new POSTagger(localTrigramScorerGreedy, greedyDecoder);
		posTaggerGreedy.train(trainTaggedSentences);
		// Test taggers
		System.out.println("Evaluating GREEDY on in-domain data:.");
		evaluateTagger(posTaggerGreedy, devInTaggedSentences, trainingVocabulary, verbose,
				basePath + "/cline_output_greedy_in.txt");
		System.out.println("Evaluating GREEDY on out-of-domain data:.");
		evaluateTagger(posTaggerGreedy, devOutTaggedSentences, trainingVocabulary, verbose,
				basePath + "/cline_output_greedy_out.txt");

		// Optionally tune hyperparameters on dev data
		// posTagger.validate(devInTaggedSentences);

		labelTestSet(posTaggerViterbi, devInTaggedSentences, basePath + "/output_in_domain_dev.txt");
		labelTestSet(posTaggerViterbi, devOutTaggedSentences, basePath + "/output_out_of_domain_dev.txt");
		labelTestSet(posTaggerViterbi, testSentences, basePath + "/output_test.txt");
	}
}
