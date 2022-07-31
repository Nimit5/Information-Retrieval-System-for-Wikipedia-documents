run:
	pip install numpy
	pip install pandas
	pip install nltk
	pip install unidecode
	sh Ques1.sh
	cd 21111045-ir-systems && sh Ques4.sh $(ARGS)
	
clean:
	rm -rf BM25_Qrel.txt
	rm -rf Boolean_Qrel.txt
	rm -rf TfIdf_Qrel.txt
	

