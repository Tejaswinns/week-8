from collections import defaultdict


class MarkovText(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None  # you'll need to build this

    def get_term_dict(self):

        # split corpus into tokens on whitespace
        tokens = self.corpus.split()

        # build a mapping from each token to the list of tokens that follow it
        term_map = defaultdict(list)

        # iterate through tokens and append the follower for each token
        for i in range(len(tokens) - 1):
            term_map[tokens[i]].append(tokens[i + 1])

        # store as a regular dict for nicer printing, but lists keep duplicates
        self.term_dict = dict(term_map)

        return self.term_dict


    def generate(self, seed_term=None, term_count=15):

        # Ensure the term dictionary exists
        if self.term_dict is None:
            self.get_term_dict()

        # If still empty, return empty string
        if not self.term_dict:
            return ''

        # lazy import to keep top-level light
        import numpy as _np

        # choose starting term
        if seed_term is None:
            current = _np.random.choice(list(self.term_dict.keys()))
        else:
            if seed_term not in self.term_dict:
                raise ValueError('seed_term not found in corpus')
            current = seed_term

        generated = [current]

        for _ in range(term_count - 1):
            followers = self.term_dict.get(current)
            # if no followers (e.g., last token in corpus), stop early
            if not followers:
                break
            # sample next token preserving empirical frequencies (followers list may contain duplicates)
            current = _np.random.choice(followers)
            generated.append(current)

        return ' '.join(generated)

    def sample_term_dict(self, n_keys=30, max_followers=8):
        """Return a compact sample of the term dictionary for display.

        Args:
            n_keys (int): number of keys to include (iteration order of dict keys).
            max_followers (int): maximum followers to show per key; if more exist, append '...'.

        Returns:
            dict: mapping of key -> list (possibly truncated) suitable for printing.
        """
        if self.term_dict is None:
            self.get_term_dict()

        sample = {}
        for key in list(self.term_dict.keys())[:n_keys]:
            followers = self.term_dict.get(key, [])
            if len(followers) > max_followers:
                sample[key] = followers[:max_followers] + ['...']
            else:
                sample[key] = list(followers)

        return sample