# encoding:UTF-8

import codecs
import csv
import json
import pprint
import sys
import time

#import exact_match.mm
#import fuzzy_match.fuzzy_match
#from clfier import sentence_clfier
#from clfier.sentence_clfier import tokenizer
from es_match import search_index
import pypinyin
import jieba
import jieba.posseg





def get_common_word(filename):
    """
    从文件中获取common word词表
    :param filename: common word文件，一个json文件
    :return: 一个包含common word词表的list，每个词是unicode形式
    """
    json_file = open(filename, 'r')
    common_word_list = json.load(json_file)
    return common_word_list['data']




def en_candidate(segs, common_words):
    s=[]
    for i in segs:  
        if i.word not in common_words and not i.flag == u't' and not i.flag == u'm':
            s.append(i.word)
    name_index=0
    en_sets=set([])
    
    '''
    如果分词结果有一个字的词，那就与其上一个的词合并
    '''
    for sname in s:        
        if len(sname)==1 and name_index>0:
            en_sets.add(s[name_index-1]+sname)
            if s[name_index-1] in en_sets:
                en_sets.remove(s[name_index-1])
        else:
            en_sets.add(sname)
        name_index+=1
    return en_sets

    
def entity_identify(sentence):
    #print sentence_clfier.sentence_clfier(sentence)
    stime=time.clock()
    entity_name_file = 'data/name-idlist-dict-all.json'
    common_words_file = 'data/merge_split2.json'

    jieba.load_userdict('data/words.txt')
    
    questions = [sentence]
    result_json={}
    #result_json[u'type'] = sentence_clfier.sentence_clfier(sentence)               
    
    
    for question in questions:
        #精确匹配
        #question_hanzi_list = list(question)
        #hanzi_entity_info = hanzi_bseg.entity_identify(question_hanzi_list)
        #question_pinyin_list = hanzi_list2pinyin(question_hanzi_list)
        #pinyin_entity_info = pinyin_bseg.entity_identify(question_pinyin_list)
        #hanzi_entity_result = exact_entity_extract(hanzi_entity_info)
        #pinyin_entity_result = exact_entity_extract(pinyin_entity_info)
                       
        
        seg = jieba.posseg.cut(question)
        en_candis=en_candidate(seg, common_words)
        
        fuzzy_entity_result=[]
        for name in en_candis:
            es_results=es_match.search_index(name)
            for es_result in es_results:
                fuzzy_entity_result.append(es_result)                
        
        
        result_json[u'entity']=fuzzy_entity_result 
        
    
        
    return result_json    

    
if __name__ == "__main__":
    stime=time.clock()
    result=entity_identify(u'感冒，发骚，咳嗽吃什么药？')
    dstr=json.dumps(result,ensure_ascii=False,indent=4);
    dstr=unicode.encode(dstr,'utf-8');
    with open('qa_result.json','wb') as f:
        f.write(dstr)               
    etime=time.clock()
    print "read: %f s" % (etime - stime)
