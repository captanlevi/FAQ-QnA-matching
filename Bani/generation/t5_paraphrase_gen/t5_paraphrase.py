from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from .abbreviation_helper \
    import get_abbreviation_dict, remove_abbreviation_expansion, reinstate_abbreviation_expansion
from ..rajat_work.qgen.generator.base import BaseGenerator


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class T5Generator(BaseGenerator):
    """ Generate questions using a T5 Model Trained on Paraphrase Dataset -
    QQP (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
    """

    def __init__(self,
                 model_path="ramsrigouthamg/t5_paraphraser",
                 top_p=0.98, num_return=11, max_len=128, top_k=120, is_early_stopping=True,
                 can_model_path="paraphrase-distilroberta-base-v1"):
        """
        Load model with model path and initialize parameters for generation
        :param str model_path: Points to the directory where the model's config.json and pytorch_model.bin is stored.
                               Can also point to a path where it loads a model from the web.
        :param float top_p: only the most probable tokens with probabilities that add up to top_p or higher are kept for
                            generation.
        :param int num_return: Number of sentences to return from the generate() method
        :param int max_len: Maximum length that will be used in the generate() method
        :param int top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering in the generate()
                          method
        :param bool is_early_stopping: Flag that will be used to determine whether to stop the beam search
        """
        super().__init__("T5 Model - Paraphrase Generation")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)

        # Parameters for generation
        self.top_p = top_p
        self.num_return = num_return
        self.max_len = max_len
        self.top_k = top_k
        self.is_early_stopping = is_early_stopping

        # Load for candidate paraphrase selection
        self.can_model = SentenceTransformer(can_model_path)
        self.original_sentences = []

    def _load_model(self, model_path):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model = model.to(self.device)
        return model

    @staticmethod
    def _preprocess(sentence, abbrevs):
        sentence = remove_abbreviation_expansion(sentence, abbrevs)
        return sentence

    @staticmethod
    def _post_process(sentence, abbrevs):
        sentence = reinstate_abbreviation_expansion(sentence, abbrevs)
        return sentence

    def generate(self, sentence):
        """
        Generate paraphrases for a given sentence

        :param str sentence: Original sentence used for generating its paraphrases

        """
        text = "paraphrase: " + sentence + "</s>"
        encoding = self.tokenizer.encode_plus(text, padding='max_length', return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=self.max_len,
            top_k=self.top_k,
            top_p=self.top_p,
            early_stopping=self.is_early_stopping,
            num_return_sequences=self.num_return
        )

        sentences_generated = []
        # sentence = re.sub(r'[^\w\s]', '', sentence)
        for output in outputs:
            sent = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in sentences_generated:
                sentences_generated.append(sent)

        return sentences_generated

    def batch_generate(self, sentences):
        # For use in candidate_selection method
        self.original_sentences = sentences

        results = dict()
        for sentence in tqdm(sentences):
            sentences_generated = self.generate_special(sentence)
            results[sentence] = sentences_generated
        return results

    def generate_special(self, sentence):
        abbrevs = get_abbreviation_dict(sentence)
        new_abbrevs = abbrevs
        # new_abbrevs = {key + "_1": value for key, value in abbrevs.items()}
        # 1. Pre Processing
        sentence_ready = self._preprocess(sentence, new_abbrevs)

        # 2. Generation
        sentences_generated = self.generate(sentence_ready)

        # 3. Post Processing
        sentence_return = []
        for sentence_gen in sentences_generated:
            sentence_return.append(self._post_process(sentence_gen, new_abbrevs))

        # 4. Candidate Paraphrase Selection
        candidate_paraphrases = self._candidate_selection(sentence, sentence_return)

        return candidate_paraphrases
        # return sentence_return

    def _candidate_selection(self, original, generated_paraphrases,
                             lower_bound=4.0, position_choices=[1]):

        # Step 1: Filter by Similarity Scores
        similarity_scores = [self._similarity_score(original, paraphrase) for paraphrase in generated_paraphrases]

        candidate_paraphrases = []
        candidate_paraphrases_score = []
        for paraphrase, score in zip(generated_paraphrases, similarity_scores):
            if original.lower() == paraphrase.lower() and paraphrase not in candidate_paraphrases:
                print(f"Removing original.lower() == paraphrase.lower() for original:{original}")
                continue

            score_float = float(score)
            if score_float >= lower_bound:
                candidate_paraphrases.append(paraphrase)
                candidate_paraphrases_score.append(score_float)

        # Step 2: Filter by the position of original in top-5 most similar questions when compared to paraphrase
        corpus = self.original_sentences
        corpus_embeddings = self.can_model.encode(corpus, convert_to_tensor=True)
        positions = []
        for paraphrase in candidate_paraphrases:
            query = [paraphrase]
            top_k = min(5, len(corpus))
            query_embedding = self.can_model.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            corpus_matches = [corpus[idx] for idx in top_results[1]]
            positions.append(self.check_position(corpus_matches, original))

        final_can_paraphrases = []
        final_can_paraphrases_score = []
        for paraphrase, position, score in zip(candidate_paraphrases, positions,candidate_paraphrases_score):
            position = int(position)
            if position in position_choices:
                final_can_paraphrases.append(paraphrase)
                # For later sorting via score
                final_can_paraphrases_score.append(score)

        # Sorting via score
        paraphrase_and_score = []
        for paraphrase, score in zip(final_can_paraphrases, final_can_paraphrases_score):
            paraphrase_and_score.append((paraphrase, score))

        paraphrase_and_score.sort(key=lambda x: x[1])

        selected_paraphrases = []
        for paraphrase_score_tuple in paraphrase_and_score:
            selected_paraphrases.append(paraphrase_score_tuple[0])

        return selected_paraphrases

    def _similarity_score(self, original, paraphrase):
        sentences1 = [original]
        sentences2 = [paraphrase]

        # Compute embedding for both lists
        embeddings1 = self.can_model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.can_model.encode(sentences2, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        score = cosine_scores[0][0] * 5
        return round(float(score), 2)

    @staticmethod
    def check_position(corpus_matches, original):
        # Return 1,2,3,4,5 for their respective position of the query_original_q in corpus_matches
        # Return -1 if not found
        for ind, match in enumerate(corpus_matches):
            if match.lower() == original.lower():
                return ind + 1
        return -1

    def debug_generate(self, sentence, is_preprocess=True, is_postprocess=True,
                       is_select=True, lower_bound=4.0, position_choices=[1]):
        sentence_ready = sentence

        print(f"Generating for: {sentence_ready}")
        if is_preprocess:
            abbrevs = get_abbreviation_dict(sentence)
            # new_abbrevs = {key + "_1": value for key, value in abbrevs.items()}
            print(f"is_preprocess=True: \nabbrevs:{abbrevs},\nnew_abbrevs:{abbrevs}\n")
            sentence_ready = self._preprocess(sentence, abbrevs)
            print(f"After pre-processing, it becomes: {sentence_ready}")

        print(f"Generating for: {sentence_ready}")
        sentences_generated = self.generate(sentence_ready)

        sentence_return = []
        if is_postprocess:
            for sentence_gen in sentences_generated:
                processed_sentence = self._post_process(sentence_gen, abbrevs)
                sentence_return.append(processed_sentence)
                print(f"Before: {sentence_gen}\nAfter: {processed_sentence}")
                print("\n")
        else:
            sentence_return = sentences_generated

        res = []
        if is_select:
            if not self.original_sentences:
                self.original_sentences = [sentence]
            res = self._candidate_selection(sentence, sentence_return, lower_bound, position_choices)
        else:
            res = sentence_return
        return res