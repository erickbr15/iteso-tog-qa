# Extracted code from Hands-on Question Answering Systems with BERT: Applications in Neural Networks and Natural Language Processing

from flask import Flask, request
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch
import json

def get_answer_using_bert(question, reference_text):
    # Load fine-tuned model for QA
    bert_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Load Vocab as well
    bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    # Apply bert_tokenizer on input text
    input_ids = bert_tokenizer.encode(question, reference_text)
    input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

    # Search index of first [SEP] token
    sep_location = input_ids.index(bert_tokenizer.sep_token_id)
    first_seg_len, second_seg_len = sep_location + 1, len(input_ids) - (sep_location + 1)
    seg_embedding = [0] * first_seg_len + [1] * second_seg_len

    # Run our example on model
    model_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([seg_embedding]))
    ans_start_loc, ans_end_loc = torch.argmax(model_scores[0]), torch.argmax(model_scores[1])
    result = ' '.join(input_tokens[ans_start_loc:ans_end_loc + 1])

    # Return final result
    result = result.replace(' ##', '')
    return result

app=Flask(__name__)

@app.route ("/questionAnswering", methods=['POST'])
def questionAnswering():
    try:
        json_data = request.get_json(force=True)
        query = json_data['query']
        context_list = json_data['context_list']
        result = []
        for val in context_list:
            context = val['context']
            context = context.replace("\n"," ")
            answer_json_final = dict()
            answer = get_answer_using_bert(context, query)
            answer_json_final['answer'] = answer
            answer_json_final['id'] = val['id']
            answer_json_final['question'] = query
            result.append(answer_json_final)

        result={'results':result}
        result = json.dumps(result)
        return result

    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__" :
    app.run(debug=True,port="5001")