#计算两份预料的相似度

def ngram_count(text_list, N=2):
    ngram_freq = {}
    total_ngram_number = 0
    for text in text_list:
        text = "".join(text)
        for i in range(len(text)-N):
            ngram = text[i: i + N]
            ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
            total_ngram_number += 1
    for ngram in ngram_freq:
        ngram_freq[ngram] /= total_ngram_number
    return ngram_freq

def get_similarity_with_ngram_freq(ngram_freq1, ngram_freq2):
    a, b, c = 0, 0, 0
    for ngram in ngram_freq1:
        a += ngram_freq1[ngram] * ngram_freq2.get(ngram, 0)

    for ngram in ngram_freq1:
        b += ngram_freq1[ngram]**2

    for ngram in ngram_freq2:
        c += ngram_freq2[ngram]**2

    return a/(b**0.5 * c**0.5)


