from datasets import load_dataset

from disambiguation.signals.unified_data import UnifiedDocument


def load_corefud_docs(split: str = "validation") -> list[UnifiedDocument]:
    ds = load_dataset("coref-data/corefud_raw", "en_gum-corefud", split=split)

    docs = []
    for doc_idx in range(len(ds)):
        sample = ds[doc_idx]

        # Build sentence list and sent_id -> sent_idx mapping
        sentences = []
        sent_id_to_idx = {}
        for si, sent in enumerate(sample["sentences"]):
            tokens = [tok["form"] for tok in sent["tokens"]]
            sentences.append(tokens)
            sent_id_to_idx[sent["sent_id"]] = si

        # Convert coref entities to clusters in our format [sent_idx, start, end_exclusive]
        clusters = []
        for entity in sample["coref_entities"]:
            if len(entity) < 2:
                continue
            cluster = []
            for mention in entity:
                sent_id = mention["sent_id"]
                if sent_id not in sent_id_to_idx:
                    continue
                si = sent_id_to_idx[sent_id]
                span = mention["span"]
                # Span format: "5" (single token) or "3-7" (range), 1-indexed
                if "-" in span:
                    parts = span.split("-")
                    start = int(parts[0]) - 1  # convert to 0-indexed
                    end = int(parts[1])  # exclusive (was inclusive 1-indexed)
                else:
                    start = int(span) - 1
                    end = start + 1
                cluster.append([si, start, end])
            if len(cluster) >= 2:
                clusters.append(cluster)

        docs.append(UnifiedDocument(
            doc_id=f"corefud_{doc_idx}",
            sentences=sentences,
            clusters=clusters,
            source="corefud",
        ))

    return docs


if __name__ == "__main__":
    docs = load_corefud_docs("validation")
    print(f"CorefUD validation: {len(docs)} docs")

    total_clusters = sum(len(d.clusters) for d in docs)
    total_mentions = sum(sum(len(c) for c in d.clusters) for d in docs)
    print(f"  Clusters: {total_clusters}")
    print(f"  Mentions: {total_mentions}")

    # Show a sample
    doc = docs[0]
    print(f"\nDoc: {doc.doc_id}, {len(doc.sentences)} sentences")
    for ci, cluster in enumerate(doc.clusters[:3]):
        print(f"  Cluster {ci} ({len(cluster)} mentions):")
        for si, st, en in cluster[:4]:
            tokens = doc.sentences[si][st:en]
            print(f"    S{si} [{st}:{en}] \"{' '.join(tokens)}\"")
