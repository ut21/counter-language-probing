from paren_2 import *
import pickle
from torch.utils.data import DataLoader, TensorDataset

with open('model_mask_paren_4_dec_maxlen_100_probable.pkl', 'rb') as f:
    model = pickle.load(f)

tokeniser = SimpleTokenizer("()")

output, internal = model(tokeniser.tokenize("()").to(device), return_states=True)
print(internal.shape)

def generate_parentheses(n):
    def backtrack(s='', left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)

    result = []
    backtrack()
    return result

sizes = [4, 8]
paren = []
for size in sizes:
    paren.extend(generate_parentheses(size))
    print("size", size, "done")

