cat chunyu_qa11.txt |egrep  -a '^q[0-9]*?\:'|sed 's/（[男|女]，.*）//g' | sed 's/^q[0-9]*\://g' > chunyu_question.txt
