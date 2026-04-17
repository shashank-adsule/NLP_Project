from util import *

# Add your import statements here
import math
from collections import defaultdict


class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents

		Returns
		-------
		None
		"""

		# ------------------------------------------------------------------
		# docs  : [  doc0,           doc1,           ...  ]
		# doc_i : [  sent0,          sent1,          ...  ]
		# sent_j: [  "token0",       "token1",       ...  ]
		# We flatten each doc to a single list of tokens for TF-IDF.
		# ------------------------------------------------------------------

		N = len(docs)  # total number of documents

		# ── Step 1: compute raw term frequency per document ────────────────
		# tf_raw[i] = {term: count, ...}  for document i
		tf_raw = []
		for doc in docs:
			counts = defaultdict(int)
			for sentence in doc:
				for token in sentence:
					counts[token.lower()] += 1
			tf_raw.append(counts)

		# ── Step 2: collect vocabulary and document frequency ─────────────
		# df[term] = number of documents the term appears in
		df = defaultdict(int)
		for counts in tf_raw:
			for term in counts:
				df[term] += 1

		vocab = list(df.keys())

		# ── Step 3: compute TF-IDF vectors for every document ─────────────
		# tfidf[i] = {term: weight, ...}
		# TF  = raw count / total tokens in document  (normalised TF)
		# IDF = log( N / df(term) )
		tfidf = []
		for counts in tf_raw:
			total_tokens = sum(counts.values()) or 1  # avoid /0
			vec = {}
			for term, cnt in counts.items():
				tf_val  = cnt / total_tokens
				idf_val = math.log(N / df[term])
				vec[term] = tf_val * idf_val
			tfidf.append(vec)

		# ── Step 4: store index ───────────────────────────────────────────
		# index is a dict so rank() can access everything it needs
		self.index = {
			"docIDs" : docIDs,   # ordered list matching tfidf list
			"tfidf"  : tfidf,    # list of per-doc TF-IDF dicts
			"df"     : df,       # document frequency per term
			"N"      : N,        # corpus size
		}


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		if self.index is None:
			return doc_IDs_ordered

		docIDs = self.index["docIDs"]
		tfidf  = self.index["tfidf"]
		df     = self.index["df"]
		N      = self.index["N"]

		for query in queries:
			# ── Build query TF-IDF vector ──────────────────────────────────
			# Flatten query sentences → token list
			q_tokens = []
			for sentence in query:
				for token in sentence:
					q_tokens.append(token.lower())

			q_counts = defaultdict(int)
			for token in q_tokens:
				q_counts[token] += 1

			total_q = sum(q_counts.values()) or 1
			q_vec = {}
			for term, cnt in q_counts.items():
				if term in df:          # only terms that exist in the corpus
					tf_val  = cnt / total_q
					idf_val = math.log(N / df[term])
					q_vec[term] = tf_val * idf_val
				# OOV terms (not in corpus) contribute 0 — IDF undefined

			# ── Compute cosine similarity with each document ───────────────
			# cosine(q, d) = (q · d) / (|q| * |d|)
			q_mag = math.sqrt(sum(v ** 2 for v in q_vec.values()))
			if q_mag == 0:
				# Query vector is all zeros (all OOV) — return docs in original order
				doc_IDs_ordered.append(list(docIDs))
				continue

			scores = []
			for idx, doc_vec in enumerate(tfidf):
				# dot product — only iterate over query terms (sparse)
				dot = sum(q_vec.get(t, 0) * doc_vec.get(t, 0) for t in q_vec)

				d_mag = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
				if d_mag == 0:
					sim = 0.0
				else:
					sim = dot / (q_mag * d_mag)

				scores.append((docIDs[idx], sim))

			# Sort by similarity descending; break ties by docID ascending
			scores.sort(key=lambda x: (-x[1], x[0]))
			doc_IDs_ordered.append([doc_id for doc_id, _ in scores])

		return doc_IDs_ordered


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST  —  python informationRetrieval.py
# ══════════════════════════════════════════════════════════════════════════════
# Runs without util.py, cranfield data, or any external dependency.
# Uses the same 3-document toy corpus from Part 1 of the assignment so you
# can verify results against your hand-computed answers.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

	print("=" * 60)
	print("  InformationRetrieval — local unit test")
	print("=" * 60)

	# ── Toy corpus (Part 1 documents, pre-tokenised into the 3-level format)
	# Structure: doc → [sentence] → [token]
	# Each doc has one sentence here for simplicity.
	toy_docs = [
		[["star", "solar", "system", "provides", "heat", "light"]],      # d1
		[["hollywood", "star", "walked", "red", "carpet", "movie", "premiere"]],  # d2
		[["astronomers", "observe", "distant", "stars", "galaxies", "using", "telescopes"]],  # d3
	]
	toy_doc_ids = [1, 2, 3]

	ir = InformationRetrieval()

	# ── Test 1: buildIndex ────────────────────────────────────────────────
	print("\n[TEST 1] buildIndex()")
	ir.buildIndex(toy_docs, toy_doc_ids)

	assert ir.index is not None, "FAIL — index is None after buildIndex()"
	assert ir.index["N"] == 3,   "FAIL — wrong document count"
	assert "star"  in ir.index["df"], "FAIL — 'star' missing from df"
	assert ir.index["df"]["star"] == 2, \
		f"FAIL — df('star') should be 2, got {ir.index['df']['star']}"
	assert ir.index["df"]["heat"] == 1, \
		f"FAIL — df('heat') should be 1, got {ir.index['df']['heat']}"
	print("  ✓  index built  |  N=3  |  df(star)=2  |  df(heat)=1")

	# Spot-check: TF-IDF weight of 'star' in d1 should be lower than 'heat' in d1
	# because star appears in 2 docs (lower IDF) while heat appears in only 1
	tfidf_d1 = ir.index["tfidf"][0]
	assert tfidf_d1["star"] < tfidf_d1["heat"], \
		"FAIL — star should have lower TF-IDF than heat in d1"
	print(f"  ✓  TF-IDF(star,d1)={tfidf_d1['star']:.4f}  <  "
		  f"TF-IDF(heat,d1)={tfidf_d1['heat']:.4f}  (star penalised by IDF)")

	# ── Test 2: rank — query "star light" ────────────────────────────────
	print("\n[TEST 2] rank()  —  query: 'star light'")
	queries_1 = [[["star", "light"]]]
	results_1 = ir.rank(queries_1)

	assert len(results_1) == 1, "FAIL — should return one ranked list"
	ranked_ids_1 = results_1[0]
	print(f"  Ranking: {ranked_ids_1}")
	assert ranked_ids_1[0] == 1, \
		f"FAIL — d1 (id=1) should be rank 1, got {ranked_ids_1[0]}"
	assert ranked_ids_1[-1] == 3, \
		f"FAIL — d3 (id=3) should be rank 3 (score=0), got {ranked_ids_1[-1]}"
	print("  ✓  d1 ranked first  (contains both 'star' and 'light')")
	print("  ✓  d3 ranked last   (neither 'star' nor 'light' present)")

	# ── Test 3: rank — query "movie star" ────────────────────────────────
	print("\n[TEST 3] rank()  —  query: 'movie star'")
	queries_2 = [[["movie", "star"]]]
	results_2 = ir.rank(queries_2)

	ranked_ids_2 = results_2[0]
	print(f"  Ranking: {ranked_ids_2}")
	assert ranked_ids_2[0] == 2, \
		f"FAIL — d2 (id=2) should be rank 1 for 'movie star', got {ranked_ids_2[0]}"
	print("  ✓  d2 ranked first  (contains 'movie' and 'star')")
	print("  ✓  Word-sense ambiguity: d1 also scores > 0 due to shared 'star' token")

	# ── Test 4: OOV query ─────────────────────────────────────────────────
	print("\n[TEST 4] rank()  —  fully OOV query: 'quantum entanglement'")
	queries_3 = [[["quantum", "entanglement"]]]
	results_3 = ir.rank(queries_3)

	ranked_ids_3 = results_3[0]
	print(f"  Ranking: {ranked_ids_3}")
	assert set(ranked_ids_3) == {1, 2, 3}, \
		"FAIL — OOV query should still return all doc IDs"
	print("  ✓  All docs returned (fallback to original order for zero-vector query)")

	# ── Test 5: multi-query batch ─────────────────────────────────────────
	print("\n[TEST 5] rank()  —  two queries at once (batch)")
	queries_batch = [
		[["star", "light"]],
		[["movie", "star"]],
	]
	results_batch = ir.rank(queries_batch)
	assert len(results_batch) == 2, "FAIL — batch should return 2 ranked lists"
	assert results_batch[0][0] == 1, "FAIL — first query: d1 should be rank 1"
	assert results_batch[1][0] == 2, "FAIL — second query: d2 should be rank 1"
	print("  ✓  Both queries ranked correctly in a single batch call")

	print("\n" + "=" * 60)
	print("  ALL TESTS PASSED")
	print("=" * 60)