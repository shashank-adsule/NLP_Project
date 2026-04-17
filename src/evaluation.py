from util import *

# Add your import statements here
import math


class Evaluation():

	# ── Internal helper ────────────────────────────────────────────────────────
	def _get_true_doc_IDs(self, query_id, qrels):
		"""
		Extract the list of relevant document IDs for a given query from qrels.

		qrels format (cran_qrels.json):
		  [ {"query_num": "1", "id": 184, "position": 1}, ... ]
		  Any entry present is considered relevant (relevance score 1-4 per assignment).
		"""
		true_ids = []
		for entry in qrels:
			if str(entry["query_num"]) == str(query_id):
				true_ids.append(int(entry["id"]))
		return true_ids


	# ══════════════════════════════════════════════════════════════════════════
	# PRECISION
	# ══════════════════════════════════════════════════════════════════════════

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		# Top-k retrieved documents
		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)

		# Count how many of the top-k are relevant
		relevant_retrieved = sum(1 for doc_id in top_k if doc_id in true_set)

		precision = relevant_retrieved / k if k > 0 else 0.0

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if not true_doc_IDs:
				continue
			p = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(p)

		meanPrecision = sum(precisions) / len(precisions) if precisions else 0.0

		return meanPrecision


	# ══════════════════════════════════════════════════════════════════════════
	# RECALL
	# ══════════════════════════════════════════════════════════════════════════

	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query
		"""

		top_k = query_doc_IDs_ordered[:k]
		true_set = set(true_doc_IDs)

		relevant_retrieved = sum(1 for doc_id in top_k if doc_id in true_set)

		# Recall = relevant retrieved / total relevant
		recall = relevant_retrieved / len(true_set) if true_set else 0.0

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		recalls = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if not true_doc_IDs:
				continue
			r = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(r)

		meanRecall = sum(recalls) / len(recalls) if recalls else 0.0

		return meanRecall


	# ══════════════════════════════════════════════════════════════════════════
	# F0.5-SCORE
	# ══════════════════════════════════════════════════════════════════════════

	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of F0.5-score of the Information Retrieval System
		at a given value of k for a single query.

		F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
		beta = 0.5  →  weights precision twice as much as recall
		"""

		beta = 0.5
		p = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		r = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

		denom = (beta ** 2) * p + r
		fscore = (1 + beta ** 2) * p * r / denom if denom > 0 else 0.0

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of F0.5-score of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		fscores = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if not true_doc_IDs:
				continue
			f = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(f)

		meanFscore = sum(fscores) / len(fscores) if fscores else 0.0

		return meanFscore


	# ══════════════════════════════════════════════════════════════════════════
	# nDCG
	# ══════════════════════════════════════════════════════════════════════════

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query.

		Since qrels only tell us relevant (1) or not (0), relevance grades are binary.
		DCG@k  = sum_{i=1}^{k}  rel_i / log2(i + 1)
		IDCG@k = DCG of the ideal ranking (all relevant docs first)
		nDCG@k = DCG@k / IDCG@k
		"""

		true_set = set(true_doc_IDs)
		top_k    = query_doc_IDs_ordered[:k]

		# DCG of the system's ranking
		dcg = 0.0
		for rank, doc_id in enumerate(top_k, start=1):
			rel = 1 if doc_id in true_set else 0
			dcg += rel / math.log2(rank + 1)

		# Ideal DCG: place all relevant docs at the top
		ideal_rels = [1] * min(len(true_set), k)  # as many 1s as possible up to k
		idcg = sum(1 / math.log2(rank + 1) for rank, _ in enumerate(ideal_rels, start=1))

		nDCG = dcg / idcg if idcg > 0 else 0.0

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries
		"""

		ndcgs = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if not true_doc_IDs:
				continue
			n = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			ndcgs.append(n)

		meanNDCG = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

		return meanNDCG


	# ══════════════════════════════════════════════════════════════════════════
	# AVERAGE PRECISION  (AP@k)
	# ══════════════════════════════════════════════════════════════════════════

	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query.

		AP@k = (1 / R) * sum_{i=1}^{k}  P@i * rel_i
		where R = total number of relevant documents,
		rel_i = 1 if the document at rank i is relevant, else 0.
		"""

		true_set = set(true_doc_IDs)
		top_k    = query_doc_IDs_ordered[:k]

		R = len(true_set)
		if R == 0:
			return 0.0

		running_relevant = 0
		ap_sum = 0.0

		for rank, doc_id in enumerate(top_k, start=1):
			if doc_id in true_set:
				running_relevant += 1
				precision_at_i = running_relevant / rank
				ap_sum += precision_at_i

		avgPrecision = ap_sum / R

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries
		"""

		aps = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, q_rels)
			if not true_doc_IDs:
				continue
			ap = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			aps.append(ap)

		meanAveragePrecision = sum(aps) / len(aps) if aps else 0.0

		return meanAveragePrecision


	# ══════════════════════════════════════════════════════════════════════════
	# RECIPROCAL RANK  (RR / MRR)
	# ══════════════════════════════════════════════════════════════════════════

	def queryReciprocalRank(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of reciprocal rank for a single query.

		RR = 1 / rank_of_first_relevant_document
		Only considers the top-k documents.
		Returns 0 if no relevant document found in top-k.

		Parameters
		----------
		arg1 : list
			Ranked list of document IDs
		arg2 : int
			Query ID
		arg3 : list
			List of relevant document IDs
		arg4 : int
			The k value

		Returns
		-------
		float
			Reciprocal rank value
		"""

		true_set = set(true_doc_IDs)
		top_k    = query_doc_IDs_ordered[:k]

		reciprocalRank = 0.0
		for rank, doc_id in enumerate(top_k, start=1):
			if doc_id in true_set:
				reciprocalRank = 1.0 / rank
				break  # only the FIRST relevant document matters

		return reciprocalRank


	def meanReciprocalRank(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of Mean Reciprocal Rank (MRR)
		averaged over all queries

		Parameters
		----------
		arg1 : list
			List of ranked document lists
		arg2 : list
			Query IDs
		arg3 : list
			Relevance judgments
		arg4 : int
			The k value

		Returns
		-------
		float
			MRR value
		"""

		rrs = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = self._get_true_doc_IDs(query_id, qrels)
			if not true_doc_IDs:
				continue
			rr = self.queryReciprocalRank(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			rrs.append(rr)

		meanReciprocalRank = sum(rrs) / len(rrs) if rrs else 0.0

		return meanReciprocalRank


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TEST  —  python evaluation.py
# ══════════════════════════════════════════════════════════════════════════════
# Runs without util.py, cranfield data, or any external dependency.
# Uses a small hand-crafted scenario with known ground-truth answers so every
# metric can be verified manually before running against the full dataset.
#
# Scenario
# --------
# 5 documents  (IDs 1–5).  Relevant docs for the query: {1, 3, 5}  (3 total)
# System ranking for the query: [1, 2, 3, 4, 5]
#
# Hand-computed expected values at k=3:
#   top-3 retrieved : [1, 2, 3]
#   relevant in top-3: {1, 3}  → 2 hits
#
#   Precision@3  = 2/3  ≈ 0.6667
#   Recall@3     = 2/3  ≈ 0.6667   (2 of 3 relevant found)
#   F0.5@3       = (1+0.25) * P * R / (0.25*P + R)
#                = 1.25 * (2/3)*(2/3) / (0.25*(2/3) + (2/3))
#                = 1.25 * (4/9) / (1/4 * 2/3 + 2/3)
#                = 1.25 * (4/9) / (5/6)
#                ≈ 0.6667  (P==R so F0.5 == F1 here)
#   AP@3         = (1/3) * [P@1*rel1 + P@2*rel2 + P@3*rel3]
#                = (1/3) * [1*1 + 0 + (2/3)*1]
#                = (1/3) * (1 + 2/3) = (1/3)*(5/3) = 5/9 ≈ 0.5556
#   nDCG@3       = DCG@3 / IDCG@3
#                  DCG@3  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0 + 0.5 = 1.5
#                  IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) ≈ 1+0.631+0.5 = 2.131
#                  nDCG@3 ≈ 1.5/2.131 ≈ 0.7039
#   RR@3         = 1/1 = 1.0   (first relevant doc is at rank 1)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

	import math as _math

	print("=" * 60)
	print("  Evaluation — local unit test")
	print("=" * 60)

	ev = Evaluation()

	# ── Shared scenario ───────────────────────────────────────────────────
	RANKING     = [1, 2, 3, 4, 5]   # system's ranked output
	TRUE_IDS    = [1, 3, 5]          # ground-truth relevant docs
	QUERY_ID    = 1
	K           = 3
	TOLERANCE   = 1e-4               # float comparison tolerance

	# ── Test 1: queryPrecision ────────────────────────────────────────────
	print("\n[TEST 1] queryPrecision()")
	p = ev.queryPrecision(RANKING, QUERY_ID, TRUE_IDS, K)
	expected_p = 2 / 3
	print(f"  Precision@{K} = {p:.4f}  (expected {expected_p:.4f})")
	assert abs(p - expected_p) < TOLERANCE, f"FAIL — got {p}"
	print(f"  ✓  PASS")

	# ── Test 2: queryRecall ───────────────────────────────────────────────
	print("\n[TEST 2] queryRecall()")
	r = ev.queryRecall(RANKING, QUERY_ID, TRUE_IDS, K)
	expected_r = 2 / 3
	print(f"  Recall@{K} = {r:.4f}  (expected {expected_r:.4f})")
	assert abs(r - expected_r) < TOLERANCE, f"FAIL — got {r}"
	print(f"  ✓  PASS")

	# ── Test 3: queryFscore ───────────────────────────────────────────────
	print("\n[TEST 3] queryFscore()  (F0.5)")
	f = ev.queryFscore(RANKING, QUERY_ID, TRUE_IDS, K)
	beta = 0.5
	denom = (beta**2) * expected_p + expected_r
	expected_f = (1 + beta**2) * expected_p * expected_r / denom
	print(f"  F0.5@{K} = {f:.4f}  (expected {expected_f:.4f})")
	assert abs(f - expected_f) < TOLERANCE, f"FAIL — got {f}"
	print(f"  ✓  PASS")

	# ── Test 4: queryAveragePrecision ─────────────────────────────────────
	print("\n[TEST 4] queryAveragePrecision()  (AP@k)")
	ap = ev.queryAveragePrecision(RANKING, QUERY_ID, TRUE_IDS, K)
	# rank1=relevant→P@1=1, rank2=not relevant, rank3=relevant→P@3=2/3
	expected_ap = (1/3) * (1.0 + 2/3)
	print(f"  AP@{K} = {ap:.4f}  (expected {expected_ap:.4f})")
	assert abs(ap - expected_ap) < TOLERANCE, f"FAIL — got {ap}"
	print(f"  ✓  PASS")

	# ── Test 5: queryNDCG ─────────────────────────────────────────────────
	print("\n[TEST 5] queryNDCG()")
	ndcg = ev.queryNDCG(RANKING, QUERY_ID, TRUE_IDS, K)
	dcg  = 1/_math.log2(2) + 0 + 1/_math.log2(4)         # ranks 1,2,3
	idcg = 1/_math.log2(2) + 1/_math.log2(3) + 1/_math.log2(4)  # ideal
	expected_ndcg = dcg / idcg
	print(f"  nDCG@{K} = {ndcg:.4f}  (expected {expected_ndcg:.4f})")
	assert abs(ndcg - expected_ndcg) < TOLERANCE, f"FAIL — got {ndcg}"
	print(f"  ✓  PASS")

	# ── Test 6: queryReciprocalRank ───────────────────────────────────────
	print("\n[TEST 6] queryReciprocalRank()")
	rr = ev.queryReciprocalRank(RANKING, QUERY_ID, TRUE_IDS, K)
	print(f"  RR@{K} = {rr:.4f}  (expected 1.0000 — first doc is relevant)")
	assert abs(rr - 1.0) < TOLERANCE, f"FAIL — got {rr}"
	print(f"  ✓  PASS")

	# RR when first relevant doc is at rank 2
	rr2 = ev.queryReciprocalRank([2, 1, 3, 4, 5], QUERY_ID, TRUE_IDS, K)
	print(f"  RR@{K} = {rr2:.4f}  (expected 0.5000 — first relevant at rank 2)")
	assert abs(rr2 - 0.5) < TOLERANCE, f"FAIL — got {rr2}"
	print(f"  ✓  PASS")

	# RR when no relevant doc in top-k
	rr3 = ev.queryReciprocalRank([2, 4, 2, 4, 5], QUERY_ID, [99, 100], K)
	print(f"  RR@{K} = {rr3:.4f}  (expected 0.0000 — no relevant in top-k)")
	assert abs(rr3 - 0.0) < TOLERANCE, f"FAIL — got {rr3}"
	print(f"  ✓  PASS")

	# ── Test 7: mean functions with mock qrels ────────────────────────────
	print("\n[TEST 7] mean*() functions with 2-query mock qrels")

	# Two queries; qrels uses the cran_qrels.json field names
	mock_qrels = [
		{"query_num": "1", "id": 1, "position": 1},
		{"query_num": "1", "id": 3, "position": 2},
		{"query_num": "1", "id": 5, "position": 3},
		{"query_num": "2", "id": 2, "position": 1},
		{"query_num": "2", "id": 4, "position": 2},
	]
	# Query 1 ranking: [1,2,3,4,5]  relevant={1,3,5}
	# Query 2 ranking: [2,1,3,4,5]  relevant={2,4}
	mock_rankings = [
		[1, 2, 3, 4, 5],
		[2, 1, 3, 4, 5],
	]
	mock_query_ids = [1, 2]

	mp  = ev.meanPrecision(mock_rankings, mock_query_ids, mock_qrels, K)
	mr  = ev.meanRecall(mock_rankings, mock_query_ids, mock_qrels, K)
	mf  = ev.meanFscore(mock_rankings, mock_query_ids, mock_qrels, K)
	map_= ev.meanAveragePrecision(mock_rankings, mock_query_ids, mock_qrels, K)
	mnd = ev.meanNDCG(mock_rankings, mock_query_ids, mock_qrels, K)
	mrr = ev.meanReciprocalRank(mock_rankings, mock_query_ids, mock_qrels, K)

	print(f"  meanPrecision@{K} = {mp:.4f}")
	print(f"  meanRecall@{K}    = {mr:.4f}")
	print(f"  meanF0.5@{K}      = {mf:.4f}")
	print(f"  MAP@{K}           = {map_:.4f}")
	print(f"  meanNDCG@{K}      = {mnd:.4f}")
	print(f"  MRR@{K}           = {mrr:.4f}")

	# Sanity checks on mean values
	assert 0.0 <= mp  <= 1.0, f"FAIL — meanPrecision out of range: {mp}"
	assert 0.0 <= mr  <= 1.0, f"FAIL — meanRecall out of range: {mr}"
	assert 0.0 <= mf  <= 1.0, f"FAIL — meanFscore out of range: {mf}"
	assert 0.0 <= map_<= 1.0, f"FAIL — MAP out of range: {map_}"
	assert 0.0 <= mnd <= 1.0, f"FAIL — meanNDCG out of range: {mnd}"
	assert 0.0 <= mrr <= 1.0, f"FAIL — MRR out of range: {mrr}"

	# Query 2: ranking=[2,1,3], relevant={2,4}
	# top-3: [2,1,3] → 2 is relevant (rank1), 4 is not in top-3
	# P@3=1/3, R@3=1/2
	p_q2 = ev.queryPrecision([2,1,3,4,5], 2, [2,4], K)
	r_q2 = ev.queryRecall([2,1,3,4,5], 2, [2,4], K)
	assert abs(p_q2 - 1/3) < TOLERANCE, f"FAIL — P@3 for q2: {p_q2}"
	assert abs(r_q2 - 1/2) < TOLERANCE, f"FAIL — R@3 for q2: {r_q2}"
	print(f"  ✓  Per-query spot check: q2 P@3={p_q2:.4f} R@3={r_q2:.4f}")

	print(f"  ✓  All mean metrics in valid [0,1] range")

	# ── Test 8: edge cases ────────────────────────────────────────────────
	print("\n[TEST 8] Edge cases")

	# k=1 precision — only the top doc
	p_k1 = ev.queryPrecision([1, 2, 3], QUERY_ID, TRUE_IDS, 1)
	assert abs(p_k1 - 1.0) < TOLERANCE, f"FAIL — P@1 should be 1.0, got {p_k1}"
	print(f"  ✓  P@1 = {p_k1:.4f}  (top doc is relevant)")

	p_k1_miss = ev.queryPrecision([2, 1, 3], QUERY_ID, TRUE_IDS, 1)
	assert abs(p_k1_miss - 0.0) < TOLERANCE, f"FAIL — P@1 should be 0.0, got {p_k1_miss}"
	print(f"  ✓  P@1 = {p_k1_miss:.4f}  (top doc is NOT relevant)")

	# nDCG when NO relevant docs in top-k
	ndcg_miss = ev.queryNDCG([2, 4], QUERY_ID, [99], K)
	assert abs(ndcg_miss - 0.0) < TOLERANCE, f"FAIL — nDCG should be 0, got {ndcg_miss}"
	print(f"  ✓  nDCG = {ndcg_miss:.4f}  (no relevant docs in ranking)")

	# AP when no relevant docs retrieved
	ap_miss = ev.queryAveragePrecision([2, 4, 6], QUERY_ID, [99], K)
	assert abs(ap_miss - 0.0) < TOLERANCE, f"FAIL — AP should be 0, got {ap_miss}"
	print(f"  ✓  AP = {ap_miss:.4f}  (no relevant docs retrieved)")

	print("\n" + "=" * 60)
	print("  ALL TESTS PASSED")
	print("=" * 60)