import nltk


def postprocess_summary(summary: str):
    # remove leading and trailing whitespace
    summary = summary.strip()

    # remove newlines in summary because ROUGE looks for newlines as sentence separators
    summary = summary.replace("\n", " ")

    # explicitly add newlines as sentence separators
    summary = "\n".join(nltk.sent_tokenize(summary))
    return summary



def postprocess_for_rouge(preds: list[str], labels: list[str]):
    preds = list(map(postprocess_summary, preds))
    labels = list(map(postprocess_summary, labels))

    return preds, labels
