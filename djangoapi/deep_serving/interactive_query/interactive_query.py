# -*- coding: utf-8 -*-
import json
import os
import sys
import traceback
from StringIO import StringIO
import pycurl
from ..utils.utils import config

reload(sys)
sys.setdefaultencoding('utf-8')

PREFIX_BASE = "<http://xianjiaotong.edu/"
PREFIX_PRO = PREFIX_BASE + "property/>"
PREFIX_DIS = PREFIX_BASE + "disease/>"
PREFIX_LAB = PREFIX_BASE + "lab/>"
PREFIX_SYM = PREFIX_BASE + "symptom/>"
PREFIX_MED = PREFIX_BASE + "medicine/>"
PREFIX_DC = PREFIX_BASE + "dclass/>"
PREFIX_MC = PREFIX_BASE + "mclass/>"
PREFIX_SC = PREFIX_BASE + "sclass/>"
PREFIX_SB = PREFIX_BASE + "sbody/>"
PREFIX_LC = PREFIX_BASE + "lclass/>"
PREFIX_OP = PREFIX_BASE + "operation/>"
PREFIX_DP = PREFIX_BASE + "department/>"
PREFIX_BP = PREFIX_BASE + "bodypart/>"

ABBRAVIATION_PRO = "pro:"
ABBRAVIATION_DIS = "dis:"
ABBRAVIATION_LAB = "lab:"
ABBRAVIATION_SYM = "sym:"
ABBRAVIATION_MED = "med:"
ABBRAVIATION_DC = "dc:"
ABBRAVIATION_MC = "mc:"
ABBRAVIATION_SC = "sc:"
ABBRAVIATION_SB = "sb:"
ABBRAVIATION_LC = "lc:"
ABBRAVIATION_OP = "op:"
ABBRAVIATION_DP = "dp:"
ABBRAVIATION_BP = "bp:"

prefix_str = "PREFIX " + ABBRAVIATION_PRO + PREFIX_PRO
prefix_str += " PREFIX " + ABBRAVIATION_DIS + PREFIX_DIS
prefix_str += " PREFIX " + ABBRAVIATION_LAB + PREFIX_LAB
prefix_str += " PREFIX " + ABBRAVIATION_SYM + PREFIX_SYM
prefix_str += " PREFIX " + ABBRAVIATION_MED + PREFIX_MED
prefix_str += " PREFIX " + ABBRAVIATION_DC + PREFIX_DC
prefix_str += " PREFIX " + ABBRAVIATION_MC + PREFIX_MC
prefix_str += " PREFIX " + ABBRAVIATION_SC + PREFIX_SC
prefix_str += " PREFIX " + ABBRAVIATION_SB + PREFIX_SB
prefix_str += " PREFIX " + ABBRAVIATION_LC + PREFIX_LC
prefix_str += " PREFIX " + ABBRAVIATION_OP + PREFIX_OP
prefix_str += " PREFIX " + ABBRAVIATION_DP + PREFIX_DP
prefix_str += " PREFIX " + ABBRAVIATION_BP + PREFIX_BP

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file_dir = os.path.join(app_dir, 'config', 'config.conf')

# simple_qa_params = config(filename=config_file_dir, section='simple_qa')
# RDF3X_API_DIR = simple_qa_params['RDF3X_API_DIR']
# KG_DAT_DIR = simple_qa_params['KG_DAT_DIR']
# KG_DATABASE = simple_qa_params['KG_DATABASE']
# REQ_FILE_DIR = simple_qa_params['REQ_FILE_DIR']


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
