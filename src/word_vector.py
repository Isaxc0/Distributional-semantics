class word_vector:
    """
    Generates word vectors

    :attr targetwords: target words used to generate features
    :attr contextsize: window size
    :attr vectors: dictionary of vectors
    :attr wordtotals: total number of word apperances (used in PPMI)
    :attr featuretotals: total of feature apperances (used in PPMI)
    :attr distributional_sims: dictionary of all distributional similarity values

    """

    def __init__(self, target_words, context_size):
        """
        Constructor

        :param target_words: target words used to generate features
        :param context_size: window size

        """
        # intial values of all attributes
        self.targetwords = target_words
        self.contextsize = context_size
        self.vectors = {}
        self.wordtotals = {}
        self.featuretotals = {}
        self.distributional_sims = {}
        self.generate_features()
        self.generate_all_PPMI()

    def generate_features(self):
        """
        Generates all features for given window size (attribute contextsize)
        This method is the same as the method in Question 1 however it is slightly adapted
        """
        for x, sentence in enumerate(sentences):
            for i, token in enumerate(sentence):
                if token in self.targetwords:
                    current = self.vectors.get(token, {})
                    features = sentence[max(0, i - self.contextsize):i] + sentence[i + 1:i + self.contextsize + 1]
                    for feature in features:
                        current[feature] = current.get(feature, 0) + 1
                        self.featuretotals[feature] = self.featuretotals.get(feature, 0) + 1
                    self.vectors[token] = current
                    self.wordtotals[token] = len(features) + self.wordtotals.get(token, 0)
                    if len(features) == 0:  # if sentence has only one word, it has not features
                        self.vectors[token] = {}

    def generate_all_PPMI(self):
        """
        Generates all PPMI scores for each word with every feature.
        The scores are represented with a dictionary with the word as the key and the value as another dictionary.
        The other dictionaries key is the feature and the value is the PMI.
        This dictionary is then stored in the distributional_sims attribute
        """
        feature_sims = {}
        for vkey in self.vectors.keys():
            features = self.vectors.get(vkey)  # gets feature dictionary for word
            for fkey in features.keys():
                feature_sims[fkey] = self.PPMI(vkey, fkey)  # calculates PPMI
                self.distributional_sims[vkey] = feature_sims  # adds to the distributional_sims dictionary attribute
            feature_sims = {}

    def PPMI(self, word, feature):
        """
        Calculates the PPMI score between word and feature
        :param word
        :param feature
        :return: The PPMI score
        """
        # the PMi is calculated
        to_log = ((self.vectors[word].get(feature) * (sum(self.featuretotals.values()))) / (
                    self.featuretotals.get(feature) * self.wordtotals.get(word)))
        score = math.log(to_log, 2)
        # if a PMI score is negative 0 is returned
        if score < 0:
            return 0
        else:
            return score

    def similarity(self, word1, word2):
        """
        Uses the dot() method to calculate dot products needed for cosine similarity between word1 and word2

        :param word1: First word
        :param word2: Second word
        :return: The dot product
        """
        aa = self.dot(self.distributional_sims.get(word1, {}), self.distributional_sims.get(word1, {}))
        ab = self.dot(self.distributional_sims.get(word1, {}), self.distributional_sims.get(word2, {}))
        bb = self.dot(self.distributional_sims.get(word2, {}), self.distributional_sims.get(word2, {}))
        if (aa == 0 or ab == 0 or bb == 0):  # if word had no features (one word sentence) then similarity must be 0
            return 0
        cos_similarity = ab / (math.sqrt(aa * bb))  # cosine similarity
        return cos_similarity

    def dot(self, vecA, vecB):
        """
        Calculates dot product between two vectors

        :param vecA: First vector
        :param vecB: Second vector
        :return: The dot product
        """
        the_sum = 0
        for (key, value) in vecA.items():
            the_sum += value * vecB.get(key, 0)
        return the_sum