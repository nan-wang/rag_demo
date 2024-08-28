import json

# load the json file `qa_pairs.json`
with open("qa_pairs.json", "r") as f:
    qa_pairs = json.load(f)

docs = [d["documents"] for d in qa_pairs]
import random
predict_doc_ids = [i for i in range(len(docs))]
random.shuffle(predict_doc_ids)
pred = [docs[i] for i in predict_doc_ids]
partial_pred = [docs[i] for i in predict_doc_ids]
partial_pred[0] = docs[0]

from langchain.evaluation import ExactMatchStringEvaluator

evaluator = ExactMatchStringEvaluator()

# for p, r in zip(pred, docs):
#     print(f"pred: {repr(p)}")
#     print(f"ref: {repr(r)}")
#     print('-' * 10)
#
# score = evaluator.evaluate_strings(prediction=pred, reference=docs)
# print(f"random pred: {score}")

for p, r in zip(partial_pred, docs):
    print(f"prd: {repr(p)}")
    print(f"ref: {repr(r)}")
    score = evaluator.evaluate_strings(prediction=p, reference=r)
    print(f"partial random pred: {score}")
    print("-" * 10)



# score = evaluator.evaluate_strings(prediction=docs, reference=docs)
# print(f"perfect pred: {score}")
