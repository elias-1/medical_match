1. 处理word2vec数据
```Shell
cat chunyu_qa11.txt |egrep  -a '^q[0-9]*?\:'|sed 's/（[男|女]，.*）//g' | sed 's/^q[0-9]*\://g' > chunyu_question.txt
cat aliyun_local_test2.txt | awk '{print $6}'|sed s/,//g| sort -n
```

2. Elasticsearch
 - drop index
    ```
    curl -XDELETE 'localhost:9200/entity-index?pretty'

    ```
 - get index
    ```Shell
    curl -XGET 'localhost:9200/entity-index?pretty'

    ```