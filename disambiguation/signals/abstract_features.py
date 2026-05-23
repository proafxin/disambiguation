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

NUM_FEATURES = 51

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
]
