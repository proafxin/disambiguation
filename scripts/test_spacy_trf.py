import spacy
import spacy.tokens
import torch
from datasets import load_from_disk

N_EXAMPLES = 5
MAX_TOKENS = 4000


def make_doc(nlp, tokens, sent_lens):
    sent_starts = []
    for sl in sent_lens:
        sent_starts.append(True)
        sent_starts.extend([False] * (sl - 1))
    return spacy.tokens.Doc(nlp.vocab, words=tokens, sent_starts=sent_starts)


def split_example_into_chunks(sents, max_tokens):
    chunks = []
    cur_tokens, cur_lens = [], []
    for sent in sents:
        if cur_tokens and len(cur_tokens) + len(sent) > max_tokens:
            chunks.append((cur_tokens, cur_lens))
            cur_tokens, cur_lens = [], []
        cur_tokens.extend(sent)
        cur_lens.append(len(sent))
    if cur_tokens:
        chunks.append((cur_tokens, cur_lens))
    return chunks


def check_dep_heads_within_sentences(doc, sent_lens):
    abs_pos = 0
    violations = 0
    for sl in sent_lens:
        sent_start = abs_pos
        sent_end = abs_pos + sl
        for i in range(sent_start, sent_end):
            head_i = doc[i].head.i
            if not (sent_start <= head_i < sent_end):
                violations += 1
        abs_pos = sent_end
    return violations


def print_sample_tokens(doc, sent_lens, n_sents=2):
    abs_pos = 0
    printed = 0
    for si, sl in enumerate(sent_lens):
        if printed >= n_sents:
            break
        print(f"  sent {si}:")
        for i in range(abs_pos, abs_pos + sl):
            tok = doc[i]
            head_rel = tok.head.i - abs_pos
            print(f"    [{i - abs_pos:2d}] {tok.text:<15} pos={tok.pos_:<6} dep={tok.dep_:<12} head_rel={head_rel:+d} ent={tok.ent_iob_}-{tok.ent_type_}")
        abs_pos += sl
        printed += 1


def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def main():
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["senter", "lemmatizer"])
    print(f"pipeline: {nlp.pipe_names}")

    ds = load_from_disk("data/preco")["train"]
    examples = [ds[i] for i in range(N_EXAMPLES)]

    all_docs = []
    all_meta = []
    for ex_idx, ex in enumerate(examples):
        sents = ex["sentences"]
        total = sum(len(s) for s in sents)
        chunks = split_example_into_chunks(sents, MAX_TOKENS)
        for chunk_tokens, chunk_lens in chunks:
            all_docs.append(make_doc(nlp, chunk_tokens, chunk_lens))
            all_meta.append((ex_idx, chunk_tokens, chunk_lens, total))

    print(f"\n{N_EXAMPLES} examples → {len(all_docs)} docs\n")

    vram_before = vram_mb()
    processed = list(nlp.pipe(all_docs, batch_size=4))
    vram_after = vram_mb()
    print(f"VRAM before: {vram_before:.0f} MB  after: {vram_after:.0f} MB\n")

    for doc, (ex_idx, chunk_tokens, chunk_lens, total_toks) in zip(processed, all_meta):
        print(f"=== example {ex_idx}  total_tokens={total_toks}  chunk_len={len(chunk_tokens)}  n_sents={len(chunk_lens)} ===")
        violations = check_dep_heads_within_sentences(doc, chunk_lens)
        print(f"  dep head cross-sentence violations: {violations}")
        print_sample_tokens(doc, chunk_lens, n_sents=2)
        print()


main()
