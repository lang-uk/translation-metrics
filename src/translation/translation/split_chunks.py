import spacy
from ftfy import fix_encoding

nlp = spacy.load("en_core_web_sm")
# doc = nlp("This is a sentence. This is another sentence.")


def split_chunks(text: str, char_limit=5000):
    """Split a long text into smaller chunks for translation."""
    text = fix_encoding(text)
    text = text.strip()

    if len(text) <= char_limit:
        return [text], [""]

    # For long texts, split at newlines near 5000 characters while preserving empty lines
    chunks = []

    # Split by lines but preserve consecutive newlines
    lines = text.split("\n")
    chunk_separators = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "":
            # since we split by newline there would always be a new line at the end, handles tabs also
            chunk_separators[-1] += lines[i] + "\n"
        else:
            current_chunk = lines[i]

            if len(current_chunk) > char_limit:
                sentences = list(nlp(current_chunk).sents)

                if max([len(s.text) for s in sentences]) > char_limit:
                    chunks.append(current_chunk)
                    chunk_separators.append("\n")
                    i += 1
                    continue

                current_sentence_chunk = ""

                chunk_pos = 0
                count = 0
                for sntnc_idx, sentence in enumerate(sentences):
                    sentence_text = sentence.text.strip()
                    count += len(sentence_text)

                    chunk_pos = current_chunk.find(sentence_text)

                    current_sentence_chunk = current_chunk[:chunk_pos]

                    if len(current_sentence_chunk) + len(sentence_text) > char_limit:
                        chunks.append(current_chunk[: chunk_pos + len(sentence_text)])
                        separator = ""
                        if sntnc_idx == len(sentences) - 1:
                            separator = current_chunk[chunk_pos + len(sentence_text) :]
                        else:
                            next_sentence = sentences[sntnc_idx + 1].text.strip()
                            separator = current_chunk[
                                chunk_pos + len(sentence_text) : current_chunk.find(
                                    next_sentence
                                )
                            ]

                        chunk_separators.append(separator)

                        current_sentence_chunk = ""
                        current_chunk = current_chunk[
                            chunk_pos + len(sentence_text) + len(separator) :
                        ]

                if current_chunk != "":
                    chunks.append(current_chunk)
                    chunk_separators.append("")

                chunk_separators[-1] += "\n"

            else:
                chunks.append(current_chunk)
                chunk_separators.append("\n")
        i += 1
    if not chunks:
        return [], []

    if len(chunks) != len(chunk_separators):
        raise ValueError("Number of chunks and separators must match")

    joined_chunks = []
    joined_separators = []
    current_chunk = chunks[0]

    for i, (chunk, separator) in enumerate(zip(chunks[1:], chunk_separators[:-1])):
        # Check if adding next chunk and separator would exceed limit
        if len(current_chunk) + len(separator) + len(chunk) <= char_limit:
            current_chunk = current_chunk + separator + chunk
        else:
            joined_chunks.append(current_chunk)
            joined_separators.append(separator)
            current_chunk = chunk

    # Add final chunk and separator
    joined_chunks.append(current_chunk)
    joined_separators.append(chunk_separators[-1])

    reconstructed_text = ""
    for chunk, sep in zip(joined_chunks, joined_separators):
        # print(repr(sep), "|", repr(chunk))
        reconstructed_text += chunk + sep

    if not reconstructed_text.strip() == text.strip():
        # print difference
        import difflib

        diff = difflib.ndiff(reconstructed_text.splitlines(), text.splitlines())

        print("Differences:")
        with open("reconstructed.txt", "w") as f:
            f.write(reconstructed_text)

        with open("original.txt", "w") as f:
            f.write(text)
        # print("\n".join(diff))
        raise ValueError("Reconstructed text does not match original")

    return joined_chunks, joined_separators


# split_chunks(dataset[3249]["conversations"][message_idx]["value"])


def process_example(example):
    for message_idx, message in enumerate(example["conversations"]):
        text = message["value"]
        chunks = split_chunks(text)
        for chunk in chunks:
            if len(chunk) > 5300:
                print(chunk)
                print("====================")
                print(f"Length: {len(chunk)}")
                # calculate percentage of number characters in text
                num_chars = sum(
                    c.isdigit() or c in [",", ".", " ", "=", "/", "+", "-", "x"]
                    for c in text
                )
                total_chars = len(text)
                percentage = 0
                if total_chars > 0:
                    percentage = (num_chars / total_chars) * 100
                    print(f"Percentage of numeric characters: {percentage:.2f}%")

                print(set(chunk))
                print("====================")
                # print(text)
                if percentage > 45:
                    continue
                continue
                raise ValueError("Chunk still too long")
    return {"chunks": max([len(chunk) for chunk in chunks])}


if __name__ == "__main__":
    print("Testing split_chunks function...")
    print(split_chunks("This is a test.\n\nThis is another test."))

    exit()
    import datasets

    dataset_name = "open-thoughts/OpenThoughts2-1M"
    dataset_name = "open-thoughts/OpenThoughts3-1.2M"
    dataset_name = "open-thoughts/OpenThoughts-114k"

    dataset = datasets.load_dataset(dataset_name, split="train")

    print(dataset)

    idx = 0
    message_idx = 1
    print(dataset[idx]["conversations"][message_idx]["value"])

    dataset = dataset.map(process_example, num_proc=12)
    dataset
