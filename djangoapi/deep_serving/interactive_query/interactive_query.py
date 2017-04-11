# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import sys
import traceback
from StringIO import StringIO
import pycurl
from ..utils.utils import config
from ..simple_qa.simple_query import *

reload(sys)
sys.setdefaultencoding('utf-8')

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file_dir = os.path.join(app_dir, 'config', 'config.conf')

interactive_query_params = config(filename=config_file_dir, section='interactive_query')
symptom_disease_dir = interactive_query_params['symptom_disease_dir']
symptom_medication_dir = interactive_query_params['symptom_medication_dir']

# symptom_disease_dir
did_sid_all = json.load(open(symptom_disease_dir + 'disease-symptomlist-dict.json'))
sid_name_all = json.load(open(symptom_disease_dir + 'symptom-id-name-dict.json'))
did_name_dict = json.load(open(symptom_disease_dir + 'disease-id-name-dict.json'))
dep_name_dict = json.load(open(symptom_disease_dir + 'department-id-name-dict.json'))
did_deplist_dict = json.load(open(symptom_disease_dir + 'disease-departmentlist-dict.json'))

id_degree_dict = json.load(open(symptom_disease_dir + 'id-degree-dict.json'))

# symptom_medication_dir
# = json.load(open(symptom_disease_dir + 'json'))
symptom_name_id_dict = json.load(open(symptom_disease_dir + 'symptom_name_id_dict.json'))
sid_midlist_dict = json.load(open(symptom_disease_dir + 'sid_didlist_dict.json'))
symptom_id_name_dict = json.load(open(symptom_disease_dir + 'symptom_id_name_dict.json'))
medicine_id_name_dict = json.load(open(symptom_disease_dir + 'drug_id_name_dict.json'))
taboo_id_name_dict = json.load(open(symptom_disease_dir + 'taboo_id_name_dict.json'))
mid_tidlist_dict  = json.load(open(symptom_disease_dir + 'did_tidlist_dict.json'))
mid_sidlist_dict  = json.load(open(symptom_disease_dir + 'did_sidlist_dict.json'))


def get_id_name_list(ret):
    idname_list = []
    if 'empty' not in ret[0]:
        for line in ret:
            line = line.strip()
            url,name = line.split()
            name = name.replace('"','')
            xid = url.split('/')[-1].replace('>','')
            idname_list.append({ 'Id':xid,'Name':name })
    return idname_list

def get_type_list(typ):
    dummy_node = RDF_node(None, None)
    query = dummy_node.prefix
    query += '''SELECT DISTINCT ?n1 ?p1    WHERE {
        ?n1 pro:type ?t1 FILTER (regex(?t1,"'''+typ+'''")).
        ?n1 pro:name ?p1 
        }
    '''.replace('\n',' ')
    ret = call_api_rdf3x(query)
    typ_list = get_id_name_list(ret)
    return typ_list

def get_fid_to_nodetype2list(fid,nodetype1,nodetype2):
    dummy_node = RDF_node(None, None)
    query = dummy_node.prefix
    xquery = ''
    # { ?n2 ?r1 dis:'''+dis+'''. ?n2 pro:type ?t2 FILTER (regex(?r1,"Dis")) } 
    query += '''SELECT DISTINCT ?n2 ?p2  WHERE {{
        { '''+nodetype1[:3]+''':'''+fid+''' ?r1 ?n2. ?n2 pro:type ?t2 FILTER (regex(?t2,"'''+nodetype2+'''")) } UNION 
        { ?n2 ?r2 '''+nodetype1[:3]+''':'''+fid+'''. ?n2 pro:type ?t2 FILTER (regex(?t2,"'''+nodetype2+'''")) } 
        }.
        { ?n2 pro:name ?p2}
        }
    '''
    # print query.replace('PREFIX','\nPREFIX').replace('SELECT','\nSELECT')
    query = query.replace('\n','').replace('    ','')
    ret = call_api_rdf3x(query)
    idname_list = get_id_name_list(ret)
    return idname_list

def get_fids_to_nodetype2list(fids,nodetype1,nodetype2):
    dummy_node = RDF_node(None, None)
    query = dummy_node.prefix
    query += '''SELECT DISTINCT ?n0 ?p0  WHERE {\n'''
    for index,fid in enumerate(fids):
        index = index + 1
        query1 = nodetype1[:3]+ ':'+fid+' ?r1 ?n0. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")).\n'
        # query1 = 'sym:'+sid+' ?r ?n0 FILTER (regex(?r,"Dis")).'
        query2 = '?n0 ?r2 '+nodetype1[:3]+':'+fid+'. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")).\n'
        query += '{{'+query1+' } UNION { '+query2+'}}.'
    query += '{ ?n0 pro:name ?p0}}'
    query = query.replace('PREFIX','\nPREFIX')
    query = query.replace('\n',' ')
    ret = call_api_rdf3x(query)
    idname_list = get_id_name_list(ret)
    return idname_list

def get_dis_sym_dict(ret):
    dis_sym_dict = { }
    sidname_dict = { }
    if len(ret) and 'empty' not in ret[0]:
        for line in ret:
            line = line.strip()
            disurl,symurl,symname = line.split()
            symname = symname.replace('"','')
            did = disurl.split('/')[-1].replace('>','')
            sid = symurl.split('/')[-1].replace('>','')
            # idname_list.append({ 'Id':xid,'Name':name })
            if did not in dis_sym_dict:
                dis_sym_dict[did] = []
            dis_sym_dict[did].append(sid)
            sidname_dict[sid] = symname
    return dis_sym_dict, sidname_dict


def get_fids_to_nodetype2all(fids,nodetype1,nodetype2):
    dummy_node = RDF_node(None, None)
    queryprefix = dummy_node.prefix
    queryprefix += '''SELECT DISTINCT ?n ?n0 ?p0  WHERE {\n'''
    fid = fids[0]
    
    query1 = '?n ?r1 ?n0. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")) FILTER (regex(?n,"'+fid+'"))\n'
    query2 = '?n0 ?r2 ?n. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")) FILTER (regex(?n,"'+fid+'"))\n'
    query = '{{'+query1+' } UNION { '+query2+'}'

    for index,fid in enumerate(fids[:-1]):
        query1 = '?n ?r1 ?n0. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")) FILTER (regex(?n,"'+fid+'"))\n'
        query2 = '?n0 ?r2 ?n. ?n0 pro:type ?t0 FILTER (regex(?t0,"'+nodetype2+'")) FILTER (regex(?n,"'+fid+'"))\n'
        query += 'UNION {'+query1+' } UNION { '+query2+'}'

    query += '} .'

    query += '{ ?n0 pro:name ?p0}}'
    query = queryprefix + query
    # print query
    query = query.replace('PREFIX','\nPREFIX')
    query = query.replace('\n',' ')
    ret = call_api_rdf3x(query)
    # print ret
    dis_sym_dict, sidname_dict = get_dis_sym_dict(ret)
    # dis_sym_dict = get_id_name_list(ret)
    return dis_sym_dict, sidname_dict


def get_bodypart(request):  # get body list back
    if request.method == "GET":
        json_out = {}
        try:
            # input_dict = json.loads(request.GET["q"])
            # json_out['Results'] = get_type_list(input_dict['Type'])
            json_out['Results'] = get_type_list('sbody')
            json_out["Return"] = 0
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(json.dumps(json_out), content_type="application/json")

def get_symptom_id(request):  # get symptom id
    if request.method == "GET":
        json_out = {}
        try:
            input_dict = json.loads(request.GET["q"])
            sname = input_dict['Name']
            query_size = 200
            es = Elasticsearch()
            res = es.search( index='medknowledge', doc_type='search', 
                    body={'size':query_size, 'query': { "query_string" : { 'fields': ['Name','Ename','Oname'], "query" :  sname } } })
            answers = res['hits']['hits']
            results = []
            for i,answer in enumerate(answers):
                d = answer['_source']
                try:
                    xid = d['Sid']
                    xname = d['Name']
                    if len(xid):
                        results.append({ 'Id':xid,'Name':xname })
                except:
                    continue
            json_out["Return"] = 0
            json_out["Results"] = results[:20]
        except:
            traceback.print_exc()
            json_out["Return"] = 1
        return HttpResponse(json.dumps(json_out), content_type="application/json")


# ycc
def get_disease_symptom(posDis, yes_sids, not_sids,
                        unknown_sids):  # get symptom of disease
    json_out = {}
    nodetype1 = 'disease'
    nodetype2 = 'symptom'
    dids = [d['Id'] for d in posDis]
    dis_sym_dict = {
        did: did_sid_all[did]
        for did in dids if did in did_sid_all
    }
    sidall = set([
        sid for sids in dis_sym_dict.values() 
        for sid in sids if sid in sid_name_all
    ])
    sidname_dict = {sid: sid_name_all[sid].strip('"') for sid in sidall}
    sidall = set(sidname_dict)
    sidall = sidall - yes_sids - not_sids - unknown_sids
    sidall = sorted(sidall)
    numdis = len(posDis)
    numsym = len(sidall)
    matrix = np.zeros([numdis, numsym], dtype=int)
    rowvis = np.ones(numdis, dtype=bool)
    for i, dis in enumerate(posDis):
        did = dis['Id']
        if did in dis_sym_dict:
            sidset = set(dis_sym_dict[did])
        else:
            sidset = set()
        for j, sid in enumerate(sidall):
            if sid in sidset:
                matrix[i, j] = 1
                if sid in not_sids:
                    rowvis[i] = False
    matrix = matrix[rowvis]
    colsum = np.sum(matrix, axis=0)
    colnum = np.ones(np.size(colsum), dtype=int) * matrix.shape[0]
    coldiff = abs(2 * colsum - colnum)
    sidindex = {sid: index for index, sid in enumerate(sidall)}
    sidall = sorted(sidall, key=lambda sid: coldiff[sidindex[sid]])
    sidname_list = [
        {
            'Id': sid,
            'Name': sidname_dict[sid]
        } for sid in sidall
        if sid not in unknown_sids and colsum[sidindex[sid]] != matrix.shape[0]
    ][:10]
    didname_dict = {}
    for i, dis in enumerate(posDis):
        if rowvis[i]:
            didname_dict[dis['Id']] = dis['Name']
    return sidname_list, didname_dict


# ycc
def get_dis_list(didlist, prefix=''):
    did_list_new = []
    for did in didlist:
        d = {}
        name = did_name_dict[did]
        deps = did_deplist_dict[did]
        d['Id'] = did
        d['Name'] = prefix + name
        deplist = []
        for dep in did_deplist_dict[did]:
            deplist.append({'Id': dep, 'Name': dep_name_dict[dep]})
        d['department'] = [dtemp['Id'] for dtemp in deplist]
        did_list_new.append(d)
    return did_list_new


# ycc
def get_dep_dis_list(didlist):

    didlist = [d['Id'] for d in didlist]
    didlist = didlist[:50]
    dep_did_dict = {}
    for did in didlist:
        for dep in did_deplist_dict[did]:
            try:
                dep_did_dict[dep].append(did)
            except:
                dep_did_dict[dep] = [did]
    dep_list = sorted(
        dep_did_dict.keys(), key=lambda s: len(dep_did_dict[s]), reverse=True)
    dep_dict_list = []
    for dep in dep_list:
        d = {'Id': dep, 'Name': dep_name_dict[dep]}
        d['disease'] = dep_did_dict[dep]
        dep_dict_list.append(d)
    return dep_dict_list


# if __name__ == "__main__":
#     pass
