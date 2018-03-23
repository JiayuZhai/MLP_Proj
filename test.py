from Train import plot_attention
import numpy as np
arr = np.random.randn(20,1)
print(arr)
inputs = (np.random.randn(20,1)*100+10000).astype(int)

V_dict = dict()
i = 1
with open('Vac.txt', 'r') as f:
	V = f.readlines()
	# print(V)
	for w in V:
		V_dict[w.strip()] = i
		# f.write(w + '\n')
		i = i+1
words = []
seq_len = 10
# print()
for x in range(seq_len):
	for word, index in V_dict.items():
		if index == inputs[x][0]:
			words.append(word)


outputs = "Positive"
plot_attention(arr,words,outputs,seq_len,"test")