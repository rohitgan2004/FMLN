from collections import defaultdict

def group_anagrams(strings):

    anagrams = defaultdict(list)

    for word in strings:

        char_count = [0] * 26
        for char in word:
            char_count[ord(char) - ord('a')] += 1

        key = tuple(char_count)
        anagrams[key].append(word)
    
    return list(anagrams.values())

strings = ["abc", "cafe", "face", "cab",  "goo"]
result = group_anagrams(strings)
print(result)