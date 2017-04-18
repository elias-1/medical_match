
# Welcome to [Drcubic Lab](http://www.drcubic.com/)

@(Djangoapi)[DeepQA|Django|Tensorflow]

## DeepQA
A Django based restful api for deep qa.

### Prerequisites
#### tensorflow serving
> Refer to [tensorflow serving](https://github.com/tensorflow/serving)

> Refer to [bazel](https://bazel.build/versions/master/docs/install-ubuntu.html)
```
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev

virtualenv serving-env
source serving-env/bin/activate
pip install numpy

pip install grpcio

git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving

cd tensorflow
./configure
cd ..

bazel build tensorflow_serving/...

bazel build //tensorflow_serving/example/ner:ner_serving
bazel-bin/tensorflow_serving/example/ner_serving /tmp/ner_output

bazel build //tensorflow_serving/example/ner:sentence_clfier_serving
bazel-bin/tensorflow_serving/example/sentence_clfier_serving /tmp/clfier_output


bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=ner --model_base_path=/tmp/ner_output/

bazel build //tensorflow_serving/example/ner:ner_client
bazel-bin/tensorflow_serving/example/ner/ner_client --server=localhost:9000
```


#### elasticsearch
> Refer to [elasticsearch](https://www.elastic.co/downloads/elasticsearch)

#### Install postgresql
---------------
##### Create the file **/etc/apt/sources.list.d/pgdg.list**, and add a line for the repository
```Shell
deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main
```

##### Import the repository signing key, and update the package lists
    ```Shell
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | \
      sudo apt-key add -
    sudo apt-get update
    ```

##### Install the db
    ```Shell
    sudo apt-get install postgresql
    ```

##### Add system user
    ```Shell
    sudo adduser dbuser
    ```

##### conifg the db
    ```Shell
    sudo su - postgres
    psql
    \password postgres
    CREATE USER dbuser WITH PASSWORD 'password';
    CREATE DATABASE kgdata OWNER dbuser;
    GRANT ALL PRIVILEGES ON DATABASE kgdata to dbuser;
    \q
    ```

##### Login
    ```Shell
    psql -U dbuser -d kgdata -h 127.0.0.1 -p 5432
    ```

##### Database administrative login by Unix domain socket
     ```Shell
     sudo vim /etc/postgresql/9.5/main/pg_hba.conf
     ```
---------------

### Files for Installation
1. Files for tensorflow serving
  - chars_vec_100.txt
  - chars_vec_50.txt
  - transition.npy
  - words_vec_100.txt

    ```Shell
    $ cd qa/data
    $ tree  -L 1
    .
    |-- chars_vec_100.txt
    |-- chars_vec_50.txt
    |-- kg_data.nt
    |-- qadata
    |-- symptom_disease
    |-- symptom_drug
    |-- transition.npy
    `-- words_vec_100.txt
    ```

2. Files for sparql-based knowledge graph
    ```Shell
     $ tree qadata/
    qadata/
    |-- input
    |-- merge-relation.json
    |-- merge-relation.json.bak
    |-- name-idlist-dict-all.json
    |-- obj_ref.json
    |-- obj_type.json
    |-- obj_type.json.bak
    |-- q_template.json
    `-- words.txt

    0 directories, 9 files
    ```

3. Files for sql-based knowledge graph
    ```Shell
    $ ll kg_data.nt
    -rwxrw-r-- 1 elias elias 244992648 Apr 12 17:27 kg_data.nt*
    ```

4. Files for interactive query
    ```Shell
     $ tree symptom_d*
    symptom_disease
    |-- department-id-name-dict.json
    |-- disease-departmentlist-dict.json
    |-- disease-id-name-dict.json
    |-- disease-symptomlist-dict.json
    |-- id-degree-dict.json
    `-- symptom-id-name-dict.json
    symptom_drug
    |-- did_sidlist_dict.json
    |-- did_tidlist_dict.json
    |-- disease-symptomlist-dict.json
    |-- drug_id_name_dict.json
    |-- drug_name_id_dict.json
    |-- sid_didlist_dict.json
    |-- symptom_id_name_dict.json
    |-- symptom_name_id_dict.json
    |-- taboo_id_name_dict.json
    `-- taboo_name_id_dict.json

    0 directories, 16 files
    ```
### Installation
1. Insert data into Elasticsearch for fuzzy matching
    ```Shell
    $ cd qa/es_match
    $ python es_input.py
    ```

2. Install [uwsgi](https://uwsgi-docs.readthedocs.io/en/latest/WSGIquickstart.html)
    > A sample uwsgi config can be found at [uwsgi config](config/uwsgi.ini)

3. Install [nginx](https://www.nginx.com/resources/wiki/start/topics/tutorials/install/)
   > A sample nginx config can be found at [uwsgi config](config/nginx.conf)
   ```
   sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/qa.key -out ssl/qa.crt
   ```

4. Run migrate
    ```Shell
    $ python manage.py makemigrations
    $ python manage.py migrate
     ```

4. Run the server for debuging
    ```Shell
    python manage.py runserver 202.117.16.221:9999 --settings=djangoapi.settings.local
     ```
5. Run the server for production env
    ```Shell
    $ uwsgi --ini config/uwsgi.ini -d config/uwsgi.log
    $ sudo service nginx start
     ```

### After Installation

1. Insert data into Postgresql for sql-based knowledge graph

   **Note** This need run django api, Make sure the api url in send_kg_data.py is right
    ```Shell
    $ cd ../postgresql_kg/
    $ python send_kg_data.py  ../data/kg_data.nt
    ```