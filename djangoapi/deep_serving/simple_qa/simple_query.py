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
params = config(filename=config_file_dir, section='simple_qa')

RDF3X_API_DIR = params['RDF3X_API_DIR']
KG_DAT_DIR = params['KG_DAT_DIR']
KG_DATABASE = params['KG_DATABASE']
REQ_FILE_DIR = params['REQ_FILE_DIR']

sub_dict = json.load(open(params['sub_dict']))
relations = json.load(open(params['relations']))
nodes_type = {
    "disease": "dis:",
    "symptom": "sym:",
    "lab": "lab:",
    "medication": "med:",
    "medicine": "med:",
    "department": "dp:",
    "operation": "op:",
    "bodypart": "bp:"
}
q_template = json.load(open(params['q_template']))
obj_ref = json.load(open(params['obj_ref']))

root_token = params['root_token']


def get_cat(id):
    if id[0] == 'd':
        return 'disease'
    elif id[0] == 's':
        return 'symptom'
    elif id[0] == 'l':
        return 'lab'
    elif id[0] == 'm':
        return 'medicine'
    elif id[0] == 'o':
        return 'operation'


def get_nodes_list(content_list):
    list_content = content_list
    ret_list = []

    if (list_content[0] == "<empty result>\n"):
        return ret_list
    else:
        for var in list_content:
            ret_list.append(var.strip('\n').strip('"'))

    return ret_list


def get_degree(node):
    query_str = prefix_str
    query_str += 'SELECT DISTINCT ?p WHERE {{ ' + \
        node + \
        ' ?r ?p FILTER (!regex (?r, "property"))} UNION {?p ?r ' + node + '}}'
    edges = call_api_rdf3x(query_str)
    return len(edges)


def get_nodes_degree_list(content_list):
    list_content = content_list
    ret_dict = {}

    if (list_content[0] == "<empty result>\n"):
        return []
    else:
        for var in list_content:
            n, p = var.split()
            ret_dict[p.strip('\n').strip('"')] = get_degree(n)
    return ret_dict


def call_api_rdf3x(request):
    print >> sys.stderr, 'call api rdf3x'
    print >> sys.stderr, request
    req_content = request

    tmp_file_dir = REQ_FILE_DIR
    tmp_file = tmp_file_dir + "input"
    file_obj = open(tmp_file, "w")
    file_obj.write(req_content)
    file_obj.close()

    api_dir = RDF3X_API_DIR
    dat_dir = KG_DAT_DIR
    dbname = dat_dir + KG_DATABASE
    command_line = api_dir + "rdf3xquery " + dbname + " " + tmp_file
    # print(command_line)

    api_req = os.popen(command_line)
    out_list = api_req.readlines()

    return out_list


def ops_api(url):
    storage = StringIO()
    try:
        nurl = url
        c = pycurl.Curl()
        c.setopt(pycurl.URL, nurl)
        c.setopt(pycurl.HTTPHEADER, [
            'Content-Type: application/json', 'token:{:s}'.format(root_token)
        ])
        c.setopt(pycurl.CONNECTTIMEOUT, 3)
        c.setopt(c.WRITEFUNCTION, storage.write)
        c.perform()
        c.close()
    except:
        return 2
    response = storage.getvalue()
    res = json.loads(response)
    return res


def get_med_content(ids, table, ckeys):
    url = 'http://59.110.52.133:9999/medknowledge/op/?q={{"Table":"{table_type}","Id":"{xid}"}}'
    meds = {}
    for id in ids:
        nurl = url.format(table_type=table, xid=id)
        res = ops_api(nurl)
        if res['Return'] == 0:
            med_name = res['Results']['Name']
            if type(ckeys) is not list:
                meds[med_name] = ckeys + ': '
                for var in res['Results']['Content'][ckeys]:
                    meds[med_name] += var[1]
                meds[med_name] += '\n'
            else:
                meds[med_name] = ''
                for ckey in ckeys:
                    meds[med_name] += ckey + ': '
                    for var in res['Results']['Content'][ckey]:
                        if len(var) > 1:
                            meds[med_name] += var[1]
                        else:
                            continue
                    meds[med_name] += '\n'
    return meds


def get_sub_types(sub_names):
    """
    返回实体所属类别及对应id
    """
    sub_types = {}
    for name in sub_names:
        sub_id = sub_dict[name][0]
        sub_type = get_cat(sub_id)
        if sub_type not in sub_types:
            sub_types[sub_type] = []
        sub_types[sub_type].append(sub_id)
    return sub_types


def get_tep(sub_types, obj_type):
    """
    返回对应类型模板
    """
    tp = ''
    if len(sub_types.keys()) == 1:
        sub_type = sub_types.keys()[0]
        tp = q_template[sub_type][obj_type][0]
    elif obj_type == 'disease':
        tp = u'相关的疾病'
    elif obj_type == 'med':
        tp = u'相关的药品'
    elif obj_type == 'symptom':
        tp = u'相关的症状'
    return str(tp)


def query_multi_sub(sub_names, obj_type):
    """
    增加多种类型主语{d,s} query {d,m,s}
    """
    sub_types_ids = get_sub_types(sub_names)
    query_str = prefix_str

    if obj_type == '2':
        if len(sub_names) == 1 and 'medicine' in sub_types_ids:
            sub_id = sub_types_ids.values()
            return get_med_content(sub_id, 'medication',
                                   [u'药物剂型', u'适应证', u'用法用量'])
        elif len(sub_names) >= 2 and 'medicine' in sub_types_ids:
            sub_ids = sub_types_ids['medicine']
            return get_med_content(sub_ids, 'medication', [u'适应证', u'禁忌证'])
        else:
            return None
    elif obj_type == '11':
        if len(sub_names) >= 1 and 'medicine' in sub_types_ids:
            sub_ids = sub_types_ids['medicine']
            return get_med_content(sub_ids, 'medication', [u'用法用量'])  # type 11
        else:
            return None
    elif obj_type in [3, 4, 5, 6]:
        print >> sys.stderr, sub_types_ids.keys()
        query_str = prefix_str
        query_str += ' SELECT DISTINCT ?n ?p WHERE { '
        for idx, sub_type in enumerate(sub_types_ids.keys()):
            sub_ids = sub_types_ids[sub_type]  # 同一类型可能会有多个node
            sub_values = ''
            filter_values = ''
            sub_r = str(sub_type[0]) + 'r'
            query_relations = relations[sub_type][obj_ref[str(obj_type)]]
            for iidx, sub_id in enumerate(sub_ids):  # 多个实体id
                if iidx == len(sub_ids) - 1:
                    sub_values += str(nodes_type[sub_type]) + \
                        str(sub_id) + ' ?' + sub_r + ' ?n '
                else:
                    sub_values += str(nodes_type[sub_type]) + \
                        str(sub_id) + ' ?' + sub_r + ' ?n. '
            for iidx, query_rel in enumerate(query_relations):
                if iidx == len(query_relations) - 1:
                    filter_values += 'regex (?' + \
                        sub_r + ', "' + query_rel + '")))'
                else:
                    filter_values += 'regex (?' + sub_r + \
                        ', "' + query_rel + '") || '
            if idx == len(sub_types_ids.keys()) - 1:
                query_str += sub_values + ' FILTER ((' + filter_values
            else:
                query_str += sub_values + ' FILTER ((' + filter_values + '.'
        query_str += ' FILTER (regex (?n, "' + obj_ref[str(obj_type)] + \
            '")).' + ' ?n pro:name ?p}'
        t_string = ''
        tep = get_tep(sub_types_ids, obj_ref[str(obj_type)])
        for name in sub_names:
            t_string += name + ' '

        t_string += tep.encode('utf-8')
        return [t_string, query_str]


def simple_qa(entities, label):
    json_out = {}
    out_anws = {}
    try:
        rdf_strings = query_multi_sub(entities, label)
        if rdf_strings is None:  # question is unsupported
            json_out['return'] = 1
            json_out['content'] = [u'意图识别不明确,句子分类可能存在错误']
        elif type(rdf_strings) is list:
            rdf_anw = call_api_rdf3x(rdf_strings[-1])
            anws_dict = get_nodes_degree_list(rdf_anw)
            out_anws.update(anws_dict)
            json_out['return'] = 0
            json_out['content'] = [rdf_strings[0]]
            if out_anws:
                temp = [
                    var[0]
                    for var in sorted(
                        out_anws.items(), key=lambda d: d[1], reverse=True)
                ]
                temp_s = ''
                for node in temp:
                    temp_s += node + ', '
                json_out['content'].append(temp_s)
            else:
                json_out['return'] = 1
                json_out['content'].append(u'暂无记录')
        elif type(rdf_strings) is dict:  # 结果由调用医疗百科api返回
            json_out['return'] = 0
            json_out['content'] = []
            if len(rdf_strings.values()) == 0:
                json_out['return'] = 1
                json_out['content'] = [u'医疗知识百科暂无记录']
            else:
                for k, v in rdf_strings.items():
                    json_out['content'].append(k)
                    json_out['content'].append(v)
    except Exception, err:
        json_out["exception"] = traceback.print_exc()
        json_out["return"] = 1
    return json_out


# ycc
def get_disease_symptom(posDis,yes_sids,not_sids,unknown_sids):  # get symptom of disease
        json_out = {}
        nodetype1 = 'disease'
        nodetype2 = 'symptom'
        # dis_sym_dict, sidname_dict = get_fids_to_nodetype2all([d['Id'] for d in posDis],nodetype1,nodetype2)
        ### read file
        dids = [d['Id'] for d in posDis]
        with open('/var/www/djangoapi/file/symptom_disease/disease-symptomlist-dict.json','r') as f:
            did_sid_all = json.loads(f.read())
        dis_sym_dict = { did:did_sid_all[did] for did in dids if did in did_sid_all }
        with open('/var/www/djangoapi/file/symptom_disease/symptom-id-name-dict.json','r') as f:
            sid_name_all = json.loads(f.read())
        sidall = set([sid for sids in dis_sym_dict.values() for sid in sids if sid in sid_name_all])
        sidname_dict = { sid:sid_name_all[sid].strip('"') for sid in sidall }
        ### 
        sidall = set(sidname_dict)
        sidall = sidall - yes_sids - not_sids - unknown_sids
        sidall = sorted(sidall)
        numdis = len(posDis)
        numsym = len(sidall)
        matrix = np.zeros([numdis,numsym],dtype=int)
        rowvis = np.ones(numdis,dtype=bool)
        for i,dis in enumerate(posDis):
            did = dis['Id']
            if did in dis_sym_dict:
                sidset = set(dis_sym_dict[did])
            else:
                sidset = set()
            for j,sid in enumerate(sidall):
                if sid in sidset:
                    matrix[i,j] = 1
                    if sid in not_sids:
                        rowvis[i] = False
        # print rowvis.shape
        # print matrix.shape
        matrix = matrix[rowvis]
        # print matrix.shape
        colsum = np.sum(matrix,axis=0) 
        colnum = np.ones(np.size(colsum),dtype=int)*matrix.shape[0]
        coldiff = abs(2*colsum - colnum)
        sidindex = { sid:index for index,sid in enumerate(sidall) }
        sidall = sorted(sidall,key = lambda sid: coldiff[sidindex[sid]])
        sidname_list = [{ 'Id':sid,'Name':sidname_dict[sid]} for sid in sidall if sid not in unknown_sids and colsum[sidindex[sid]] != matrix.shape[0]][:10]
        didname_dict= { }
        for i,dis in enumerate(posDis):
            if rowvis[i]:
                didname_dict[dis['Id']] = dis['Name']
        '''
        sidsel = [sid for sid in sidall if colsum[sidindex[sid]] > 0 and colsum[sidindex[sid]] < matrix.shape[0]]
        ddd = OrderedDict()
        for sid in sidall:
            ddd[sid] = colsum[sidindex[sid]]
        return sidall,sidsel,ddd
        '''
        return sidname_list,didname_dict
        return dis_sym_dict,sidname_dict


# ycc
def get_dis_list(didlist,prefix=''):
    def myreadjson(fname):
        with open(fname,'r') as f:
            return json.loads(f.read())
    did_name_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/disease-id-name-dict.json')
    dep_name_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/department-id-name-dict.json')
    did_deplist_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/disease-departmentlist-dict.json')
    did_list_new = []
    for did in didlist:
        d = { }
        name = did_name_dict[did]
        deps = did_deplist_dict[did]
        d['Id'] = did
        d['Name'] = prefix + name
        deplist = []
        for dep in did_deplist_dict[did]:
            deplist.append({ 'Id':dep,'Name':dep_name_dict[dep] })
        # d['department'] = deplist
        d['department'] = [dtemp['Id'] for dtemp in deplist]
        did_list_new.append(d)
    return did_list_new
        
# ycc
def get_dep_dis_list(didlist):
    def myreadjson(fname):
        with open(fname,'r') as f:
            return json.loads(f.read())
    did_name_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/disease-id-name-dict.json')
    dep_name_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/department-id-name-dict.json')
    did_deplist_dict  = myreadjson('/var/www/djangoapi/file/symptom_disease/disease-departmentlist-dict.json')
    didlist = [d['Id'] for d in didlist]
    didlist = didlist[:50]
    dep_did_dict = { }
    for did in didlist:
        for dep in did_deplist_dict[did]:
            try:
                dep_did_dict[dep].append(did)
            except:
                dep_did_dict[dep] = [did]
    dep_list = sorted(dep_did_dict.keys(),key = lambda s:len(dep_did_dict[s]),reverse=True)
    dep_dict_list = []
    for dep in dep_list:
        d = { 'Id':dep,'Name':dep_name_dict[dep] }
        # d['disease'] = [{ 'Id':did,'Name':did_name_dict[did] } for did in dep_did_dict[dep]]
        d['disease'] = dep_did_dict[dep]
        dep_dict_list.append(d)
    return dep_dict_list

            



# if __name__ == "__main__":
#     s = simple_qa([u'发烧'], 3)
#     json.dump(s, open('./res.json', 'w'), indent=4, ensure_ascii=False)
