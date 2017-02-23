cat chunyu_qa11.txt |egrep '^q[0-9]*?\:'|sed 's/（[男|女]，.*）//g' > chunyu_question.txt
