from flask import Flask, request
import json
from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

def odqa_deeppavlov(questions):
    odqa = build_model(configs.odqa.en_odqa_infer_wiki, download = True)
    results = odqa(questions)
    return results

app=Flask(__name__)
@route ("/opendomainquestionAnswering", methods=['POST'])
def opendomainquestionAnswering():
    try:
        json_data = request.get_json(force=True)
        questions = json_data['questions']
        answers_list = odqa_deeppavlov(questions)
        index = 0
        result = []
        for answer in answers_list:
            qa_dict = dict()
            qa_dict['answer']=answer
            qa_dict['question']=questions[index]
            index = index+1
            result.append(qa_dict)
        results = {'results':result}
        results = json.dumps(results)
        return results
    except Exception as e:
        return {"Error": str(e)}
if __name__ == "__main__" :
    app.run(debug=True,port="5000")