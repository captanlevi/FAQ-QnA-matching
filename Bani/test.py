from common.utils import *
from common.FAQ import *
from sentence_transformers import  SentenceTransformer

from common.generation import IdentityProducer


from common.generation import GeneratonManager
q2L = load_dict("babybonus/question_to_label.pkl")
a2L = load_dict("babybonus/answer_to_label.pkl")



l2A = dict()
for a,l in a2L.items():
    l2A[l] = a


questions = []
answers = []

limit = 1000
for q,l in q2L.items():
    questions.append(q)
    answers.append(l2A[l])
    if(limit < 0):
        break
    limit -= 1


Q,A = processRaw(questions,answers)


faq = FAQ(name = "babyBonus", questions= Q, answers= A)
model  = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")
#faq._paraphrase(Identitygenerator())

faq.buildFAQ(model ,GeneratonManager([IdentityProducer]))
print(len(faq.FAQ))
#faq.save('./')
print(faq.isUsable())


"""

#generationManager = GeneratonManager(producers= [SymSubGenerator(dummyEN("lite")),FPMGenerator()])

#print(generationManager.generate(questions[:10]))

"""