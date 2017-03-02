# encoding:UTF-8

import json
import time

from core import entity_with_clfier

if __name__ == "__main__":
    stime = time.clock()
    result = entity_with_clfier.entity_identify(u'感冒吃什么药？')
    dstr = json.dumps(result, ensure_ascii=False, indent=4)
    dstr = unicode.encode(dstr, 'utf-8')
    with open('qa_result.json', 'wb') as f:
        f.write(dstr)
    etime = time.clock()
    print "read: %f s" % (etime - stime)
