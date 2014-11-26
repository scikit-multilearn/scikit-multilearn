Datasets: loading data sets for scikit-multilearn
=================================================


Loading from ARFF
-----------------
The class ``ArffHandler`` allows loading data from ``WEKA``, ``MULAN`` or ``MEKA`` provided data sets in `ARFF` format. The module depends on ``liac-arff`1 and apart from the file path it takes two arguments:

- labelcount: integer, the number of labels in the data set
- endian: enum{"big", "little"}: defiens whether to look for labels at the beginning of attributes in the ARFF file ("big" endian) such as the files used by MEKA or at the end ("little") endian such as the MULAN files.


Scikit-multilearn data set format
---------------------------------
Following ``scikit-learn``'s approach we assume multilabel data to be divided into two sets:

- X: the array-like of vector-likes, i.e. the array of row vectors that consist of input features (same length, i.e. feature/attribute count), 
ex. a two-object set with each row being a small 1px x 1px image with rgb channels (3 int8 describing red,blue,green colors per pixel): ``[[128,10,10,20,30,128], [10,155,30,10,155,10]]``

- y: the array-like of vector-likes, i.e. the array of binary label vectors of the same length (i.e. the label count) ex, for 3 labels: ``[[1,0,1], [0,1,0]]``


The ``scikit-multilearn`` provided data sets are produced using ``ArffHandler`` and ``Dataset`` clasess and contain a dictionary with two keys: ``X``, ``y``, containing a data set in the format described above. The data sets are ``pickle`` dumps compressed using the ``bz2`` module. They can be loaded using the ``Dataset`` class.