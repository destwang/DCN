
import os
import sys

def convert_from_myformat_to_sighan(input_path, output_path, pred_path, orig_path=None, spellgcn=False):
  with open(pred_path, "w") as labels_writer:
    with open(input_path, "r") as org_file, open(orig_path, "r") as id_f:
      with open(output_path, "r") as test_file:
        test_file = test_file.readlines()
        org_file = org_file.readlines()
        print(len(test_file), len(org_file))
        assert len(test_file) == len(org_file)
        for k, (pred, inp, sid) in enumerate(zip(test_file, org_file, id_f)):
          if spellgcn:
            _, atl = inp.strip().split("\t")
            atl = atl.split(" ")[1:]
            pred = pred.split(" ")[1:len(atl)+1]
          else:
            atl, _, _= inp.strip().split("\t")[:3]
            atl = atl.split(" ")
            pred = pred.split(" ")[:len(atl)]
          output_list = [sid.split()[0]]
          for i, (pt, at) in enumerate(zip(pred[:], atl[:])):
            if at == "[SEP]" or at == '[PAD]':
              break
            # Post preprocess with unsupervised methods, 
      #because unsup BERT always predict punchuation at 1st pos
            if i == 0:
              if pt == "。" or pt == "，":
                continue
            if pt.startswith("##"):
                pt = pt.lstrip("##")   
            if at.startswith("##"):
                at = at.lstrip("##")   
            if pt != at:
              output_list.append(str(i+1))
              output_list.append(pt)
              
          if len(output_list) == 1:
            output_list.append("0")
          labels_writer.write(", ".join(output_list) + "\n") 



def eval_spell(truth_path, pred_path, with_error=True):
  #Compute F1-score
  detect_TP, detect_FP, detect_FN = 0, 0, 0
  correct_TP, correct_FP, correct_FN = 0, 0, 0
  detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
  dc_TP, dc_FP, dc_FN = 0, 0, 0
  for idx, (pred, actual) in enumerate(zip(open(pred_path, "r", encoding='utf-8'), 
    open(truth_path, "r", encoding='utf-8') if with_error else
    open(truth_path, "r", encoding='utf-8'))):
    pred_tokens = pred.strip().split(" ")
    actual_tokens = actual.strip().split(" ")
    #assert pred_tokens[0] == actual_tokens[0]
    pred_tokens = pred_tokens[1:]
    actual_tokens = actual_tokens[1:]
    detect_actual_tokens = [int(actual_token.strip(",")) \
  for i,actual_token in enumerate(actual_tokens) if i%2 ==0]
    correct_actual_tokens = [actual_token.strip(",") \
  for i,actual_token in enumerate(actual_tokens) if i%2 ==1]
    detect_pred_tokens = [int(pred_token.strip(",")) \
  for i,pred_token in enumerate(pred_tokens) if i%2 ==0]
    _correct_pred_tokens = [pred_token.strip(",") \
  for i,pred_token in enumerate(pred_tokens) if i%2 ==1]

    # Postpreprocess for ACL2019 csc paper which only deal with last detect positions in test data.
    # If we wanna follow the ACL2019 csc paper, we should take the detect_pred_tokens to:

    
    max_detect_pred_tokens = detect_pred_tokens
    
    correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
    correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)
      
    if detect_pred_tokens[0] !=  0:
      sent_P += 1
      if sorted(correct_pred_zip) == sorted(correct_actual_zip):
        correct_sent_TP += 1
    if detect_actual_tokens[0] != 0:
      if sorted(detect_actual_tokens) == sorted(detect_pred_tokens): 
        detect_sent_TP += 1
      sent_N += 1

  

    if detect_actual_tokens[0]!=0:
      detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens)) 
      detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens)) 
    detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens)) 
    
    correct_pred_tokens = []
    #Only check the correct postion's tokens
    for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
      if dpt in detect_actual_tokens:
        correct_pred_tokens.append((dpt,cpt))

    correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens,correct_actual_tokens))) 
    correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens,correct_actual_tokens)))
    correct_FN += len(set(zip(detect_actual_tokens,correct_actual_tokens)) - set(correct_pred_tokens)) 

    # Caluate the correction level which depend on predictive detection of BERT
    dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
    dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
    dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens)) 
    dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens)) 
    dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens)) 
  
  detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP)
  detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN)
  detect_F1 = 2. * detect_precision * detect_recall/ ((detect_precision + detect_recall) + 1e-8)

  correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP)
  correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN)
  correct_F1 = 2. * correct_precision * correct_recall/ ((correct_precision + correct_recall) + 1e-8)

  dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8)
  dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8)
  dc_F1 = 2. * dc_precision * dc_recall/ (dc_precision + dc_recall + 1e-8)
  if with_error:
    #Token-level metrics
    print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" %(detect_precision, detect_recall, detect_F1))
    print("correct_precision=%f, correct_recall=%f, correct_Fscore=%f" %(correct_precision, correct_recall, correct_F1))  
    print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" %(dc_precision, dc_recall, dc_F1))
  
  detect_sent_precision = detect_sent_TP * 1.0 / (sent_P)
  detect_sent_recall = detect_sent_TP * 1.0 / (sent_N)
  detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall/ ((detect_sent_precision + detect_sent_recall) + 1e-8)

  correct_sent_precision = correct_sent_TP * 1.0 / (sent_P)
  correct_sent_recall = correct_sent_TP * 1.0 / (sent_N)
  correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall/ ((correct_sent_precision + correct_sent_recall) + 1e-8)

  if not with_error:
    #Sentence-level metrics
    print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" %(detect_sent_precision, detect_sent_recall, detect_sent_F1))
    print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" %(correct_sent_precision, correct_sent_recall, correct_sent_F1))  


if __name__ == '__main__':
  output_path = sys.argv[1]
  data_path = sys.argv[2]
  input_path = os.path.join(data_path, "test_format.txt")
  pred_path = os.path.join(os.path.dirname(output_path), 'pred_result.txt')
  orig_input_path = os.path.join(data_path, "TestInput.txt")
  orig_truth_path = os.path.join(data_path, "TestTruth.txt")
  convert_from_myformat_to_sighan(input_path, output_path, pred_path, orig_truth_path, spellgcn=False)
  eval_spell(orig_truth_path, pred_path, with_error=False)
