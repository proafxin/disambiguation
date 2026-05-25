ENT_TYPE_IDS = {
    "PERSON": 0, "ORG": 1, "GPE": 2, "LOC": 3, "NORP": 4,
    "FAC": 5, "PRODUCT": 6, "EVENT": 7, "WORK_OF_ART": 8,
    "": 9,  # not an entity
    # DATE/TIME/MONEY/PERCENT/QUANTITY/ORDINAL/CARDINAL → fallback len(ENT_TYPE_IDS)
}

POS_IDS = {
    "PROPN": 0, "NOUN": 1, "PRON": 2, "ADJ": 3, "VERB": 4, "DET": 5,
    "ADP": 6, "AUX": 7, "ADV": 8, "SCONJ": 9, "CCONJ": 10, "PART": 11,
    "NUM": 12, "PUNCT": 13, "X": 14,
}
DEP_IDS = {
    "nsubj": 0, "obj": 1, "obl": 2, "nmod": 3, "nmod:poss": 4, "appos": 5,
    "conj": 6, "compound": 7, "flat": 8, "det": 9, "amod": 10, "root": 11,
    "nsubj:pass": 12, "obl:agent": 13, "iobj": 14, "ccomp": 15, "xcomp": 16,
    "acl": 17, "acl:relcl": 18, "advcl": 19, "expl": 20, "cop": 21,
    "advmod": 22, "mark": 23, "aux": 24, "aux:pass": 25, "case": 26,
    "cc": 27, "punct": 28, "nummod": 29,
}
GENDER_IDS = {"Masc": 0, "Fem": 1, "Neut": 2, "unknown": 3}
NUMBER_IDS = {"Sing": 0, "Plur": 1, "unknown": 2}
PRONTYPE_IDS = {"Prs": 0, "Art": 1, "Dem": 2, "Rel": 3, "Int": 4, "unknown": 5}

NUM_FEATURES = 54

FEATURE_NAMES = [
    "o_pos", "o_dep", "o_gender", "o_number", "o_is_subj",
    "cur_pos", "cur_dep", "cur_gender", "cur_number",
    "c_pos", "c_dep", "c_gender", "c_number", "c_person", "c_is_subj",
    "same_pos_current", "c_depth_to_root", "same_sentence",
    "token_distance", "sent_distance", "hop_count",
    "gender_match_origin", "number_match_origin",
    "gender_match_current", "number_match_current",
    "resolved_gender_match", "resolved_number_match",
    "dep_consistent", "pos_consistent", "is_propn_terminal",
    "both_heads_verb", "same_head_token", "cand_is_verb_arg", "origin_is_verb_arg",
    "num_cands", "num_gender_match_cands", "num_propn_cands",
    "graph_candidate_resolved", "graph_candidate_confidence", "graph_same_cluster_as_origin",
    "propn_first_occurrence_distance", "propn_doc_frequency",
    "cand_sent_propn_count", "cand_token_pos_in_sent", "origin_doc_position",
    "chain_progress", "prior_same_pos_count", "cand_in_quotes",
    "cur_is_subj", "cand_is_subj_graph_resolved", "cand_is_subj_no_graph",
    "o_ent_type", "cur_ent_type", "c_ent_type",
]

# ---------------------------------------------------------------------------
# Full per-token spaCy attribute spec for the Stage 1 graph model.
# Every categorical attribute spaCy emits is exposed as its own embedding.
# Each map holds real values only; an absent/unseen value falls back to
# len(map), so the embedding cardinality is len(map) + 1.
# ---------------------------------------------------------------------------

TAG_IDS = {tag: i for i, tag in enumerate([
    "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "CC", "CD", "DT", "EX",
    "FW", "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP",
    "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
    "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$",
    "WRB", "XX", "_SP", "``",
])}

ENT_IOB_IDS = {"O": 0, "B": 1, "I": 2}

# spaCy morphologizer feature value sets (English), one map per feature.
MORPH_IDS = {
    "Gender": {"Masc": 0, "Fem": 1, "Neut": 2},
    "Number": {"Sing": 0, "Plur": 1},
    "Person": {"1": 0, "2": 1, "3": 2},
    "PronType": {"Prs": 0, "Art": 1, "Dem": 2, "Rel": 3, "Ind": 4},
    "Case": {"Nom": 0, "Acc": 1},
    "Definite": {"Def": 0, "Ind": 1},
    "Degree": {"Pos": 0, "Cmp": 1, "Sup": 2},
    "VerbForm": {"Fin": 0, "Inf": 1, "Part": 2, "Ger": 3},
    "Tense": {"Past": 0, "Pres": 1},
    "Mood": {"Ind": 0},
    "Aspect": {"Perf": 0, "Prog": 1},
    "NumType": {"Card": 0, "Ord": 1, "Mult": 2},
    "Poss": {"Yes": 0},
    "Reflex": {"Yes": 0},
    "Polarity": {"Neg": 0},
    "VerbType": {"Mod": 0},
    "ConjType": {"Cmp": 0},
    "Foreign": {"Yes": 0},
    "PunctType": {"Brck": 0, "Comm": 1, "Dash": 2, "Peri": 3, "Quot": 4},
    "PunctSide": {"Ini": 0, "Fin": 1},
}

# Orthographic shape strings covering ~99% of corpus tokens (derived from full corpus scan).
SHAPE_IDS = {s: i for i, s in enumerate([
    "xxxx", "xxx", "xx", ".", "Xxxxx", ",", "x", "Xxx", "Xxxx", " ",
    "Xx", "X", "'x", "``", "''", "x'x", "dd", "?", ":", "xxxx-xxxx",
    "dddd", "'xx", "--", "!", "d", "-", "XX", ")", "(", "XXX",
    "ddd", "'", "XXXX", ";", "/.", "$", "Xx.", '"', "xxx-xxxx", "%",
])}

# Ordered categorical columns: (name, value->id map).
# Token-level attrs first (pos/tag/dep/ent_type/ent_iob/shape), then every morph feature.
CAT_SPEC: list[tuple[str, dict]] = [
    ("pos", POS_IDS),
    ("tag", TAG_IDS),
    ("dep", DEP_IDS),
    ("ent_type", ENT_TYPE_IDS),
    ("ent_iob", ENT_IOB_IDS),
    ("shape", SHAPE_IDS),
] + [(name, MORPH_IDS[name]) for name in MORPH_IDS]

CAT_CARDINALITIES = [len(m) + 1 for _, m in CAT_SPEC]
N_CAT = len(CAT_SPEC)

# Column indices used for in-model pairwise agreement features.
CAT_INDEX = {name: i for i, (name, _) in enumerate(CAT_SPEC)}
IDX_GENDER = CAT_INDEX["Gender"]
IDX_NUMBER = CAT_INDEX["Number"]
IDX_PERSON = CAT_INDEX["Person"]
IDX_ENT_TYPE = CAT_INDEX["ent_type"]

# Continuous/boolean per-token columns (orthographic + tree shape).
# Columns: is_alpha, is_digit, is_title, is_upper, is_lower, is_stop,
#          like_num, depth/20, n_lefts/L, n_rights/L,
#          sent_pos (ti/(L-1)), is_sent_start, is_bracket, is_quote
N_CONT_FULL = 14

# Pairwise syntactic-relation features (precomputed per nominal pair).
# Columns: same_clause, dominates, path_len_norm, arc_appos, arc_conj,
#          arc_poss, arc_relcl, same_head
# (signed linear order lives in the pair head as `gap`, so it is not repeated here)
N_PAIR_SYNT = 8
