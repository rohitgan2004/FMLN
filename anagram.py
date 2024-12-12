from collections import defaultdict

def group_anagrams(strings):

	anagrams = defaultdict(list)

	for word in strings:

		key = ''.join(sorted(word))
		anagrams[key].append(word)

	return list(anagrams.values())

strings = ["abc", "cafe", "face", "cab",  "goo"]
result = group_anagrams(strings)
print(result)