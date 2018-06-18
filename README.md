# 텍스트 데이터 확장 연구

딥러닝과 같은 복잡한 모델의 경우 과적합 문제를 피하기 위해서 다양한 테크닉들을 사용한다. 그 중에서 학습 데이터를 늘리는 것이 가장 근본적인 해결책이 될 수 있다. 보통 수작업으로 학습 데이터를 구축하는데 많은 비용과 시간이 소요되는 단점이 있다. 따라서, 자동으로 학습 데이터를 확장할 수 있는 연구가 필요하다. 본 연구에서는 개체명 인식 문제에 사용되는 텍스트 학습 데이터를 자동으로 확장할 수 있는 알고리즘을 개발한다. 

![](assets/figure.PNG)

연속적인 값을 가지는 이미지, 음성 데이터와 다르게 이산적인 값을 가지는 텍스트 데이터는 휴리스틱(로직) 기반으로 알고리즘을 설계한다. 규칙들과 언어 자원들을 사용하여 (1)교체, (2)삽입, (3)삭제, (4)이동으로 크게 4가지 알고리즘을 개발하였다. 개체명 인식을 위한 데이터 확장 연구이지만, 특정 모듈만 사용하여 문서 분류 문제에서도 사용될 수 있다 (참고: doc_class폴더에서 python main.py --doc 실행).

본 연구의 내용은 [문서](https://1drv.ms/w/s!AllPqyV9kKUrkULe4nbvFU8Oj8j3)와 [파워포인트](https://1drv.ms/p/s!AllPqyV9kKUrkUOft37gKbgNnOIz)에 정리되어 있다.

## Prerequisites
* [spaCy](https://spacy.io/)
* [NLTK](https://www.nltk.org/)
* [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.html)
* [sense2vec](https://github.com/explosion/sense2vec)
* [pdfminer](https://github.com/euske/pdfminer) (Python PDF Parser)
* [NeuroNER](https://github.com/Franck-Dernoncourt/NeuroNER) (Neural Model for NER)
* [GloVe](https://nlp.stanford.edu/projects/glove/) (pretrained word embedding; glove.6B.100d.txt)
* [Oxford Collocation Dictionary](http://www.freecollocation.com/)

## Model
* NeuroNER; a program for named entity recognition based on nueral networks [[github](https://github.com/Franck-Dernoncourt/NeuroNER)] [[web-site](http://neuroner.com/)]
* Window Model for NER [[cs224d](http://cs224d.stanford.edu/assignment2/index.html)] (참고: window_rnn_ner 폴더에서 python root.py 10(#exe) 실행)

## Dataset
* [CoNLL 2003 shared task (NER) data](https://cogcomp.org/page/resource_view/81)

## Usage

0. **Prepare resources** <br>
* Download download `Oxford Collocation Dictionary` as pdf file in the pdfminer/myDir/
* Generate *friend_list* and *voca* for insert module by executing pdfminer.six/convert_pdf.ipynb and locate them in resources/for_insert/


1. **Generate Augmented Data**
```
~$ python src-aug/main.py
```

2. **Execute Model** (before doing, change data in data/conll2003/en/.. into augmented data generated via 1.)
```
~$ python src-neuroner/root.py 10
```


## Contribution
* 이산적 특징을 가지는 텍스트 학습 데이터를 자동으로 늘릴 수 있는 연구
* wordnet과 word2vec을 사용한 교체 알고리즘 설계
* 연어 사전을 이용한 삽입 알고리즘 설계
* 구문 분석기를 이용한 이동, 삭제 알고리즘 설계


## Summary
* 개체명 인식 데이터셋 정리 [[문서](https://1drv.ms/w/s!AllPqyV9kKUrkUUHOqncvjE1ZMbW)]
* 연구 아이디어 및 관련연구 정리 [[문서](https://1drv.ms/p/s!AllPqyV9kKUrkUt29nfhcEL-XRI_)]
* 연구 노트 정리 [[문서](https://1drv.ms/w/s!AllPqyV9kKUrkW9YbbtvBB_UpHVd)]
* 삽입 모듈에 사용되는 oxford collocation dictionary를 문서화하기 위하여 [pdfminer](https://github.com/euske/pdfminer) 사용 (참고: insert.ipynb)


## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST), Korea <br>
2017.08 ~ 10
