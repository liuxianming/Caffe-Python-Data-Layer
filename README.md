<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. Python Data Layer</a>
<ul>
<li><a href="#sec-1-1">1.1. TO-DO</a></li>
<li><a href="#sec-1-2">1.2. Fileds and Keywords in 'param_str':</a>
<ul>
<li><a href="#sec-1-2-1">1.2.1. TripletDataLayer:</a></li>
</ul>
</li>
<li><a href="#sec-1-3">1.3. How to deal with multi-labels</a></li>
</ul>
</li>
</ul>
</div>
</div>


# Python Data Layer<a id="sec-1" name="sec-1"></a>

This is implemenation of python data layer based on python\_layer in caffe.

## TO-DO<a id="sec-1-1" name="sec-1-1"></a>

1.  [-] Add siamese layer, triplet sampling layer implementations<code>[50%]</code>
    1.  [ ] Siamese layer
    2.  [X] Triplet Layer
2.  [ ] Add free sampling functions / interface to sampling functions in python data layer
3.  [X] Write documents and instructions on how to use these layers (what's the input, what's the output)
4.  [X] List all the options in param\_str for python layer

## Fileds and Keywords in 'param\_str':<a id="sec-1-2" name="sec-1-2"></a>

-   batch\_size: default 256
-   resize: default -1, don't do any resizing
-   mean\_file: default NA, filename / path of image mean file, or in formate of [mean\_r, mean\_g, mean\_b]
-   source\_type: type of input source, could be "CSV", "LMDB", "BCF". Default = "CSV"
-   Input Related (used by DataManager):
    -   source: source of input file name
    -   BCF MODE: bcf is compressed binary file format used in Adobe Research Lab
        -   bcf\_mode: FILE or MEM, read BCF into memory or open file in cache, default FILE
        -   labels: the file name of label files, in numpy binary file format, each row should be labels for one sample
    -   CSV MODE: in this mode, the input is a csv file, separator could be space, tab, or comma. The first column is image / sample file name, and the rest columns are labels. If there are multiple columns labels, it will read all labels and concate as a string
        -   root: root dir relative to the file name in filename column, by default None
    -   LMDB MODE: read compressed data from LMDB, will use caffe.io.caffe\_pb2.Datum to decode data
        -   labels: path to Label LMDB. If exists, will read labels from label LMDB, otherwise, will use datum.label from data LMDB as labels

### TripletDataLayer:<a id="sec-1-2-1" name="sec-1-2-1"></a>

-   prefetch: if using a prefetch process or not, default = False
-   type: the type of sampling (not case sensitive), including:
    -   **RANDOM**: random sampling
    -   **RANDOM\_MULTILABEL**: randomly sampling with assumption of multilabel. A margin (similarity of positive pair - similarity of negative pair) will also be provided as label
    -   **HARD\_MULTILABEL**: hard negative sampling based on multiple labels. It will pick several negative samples and find one with smallest similarity with the anchor image based on their labels.
    -   **HARD**: hard negative sampling based on pre-calculated similarity graph. The graph is in the format of adjacant matrix.

These options are used for hard negative sampling:
-   k: how many negative smaples to choose as candidates to find the hardest negative one.
-   m: similarity graph filename, in format of python dict (or json, or CSV)
-   n: number of iterations to run randomly sampling before hard negative sampling, to ensure the network well initialized in the beginning.

## How to deal with multi-labels<a id="sec-1-3" name="sec-1-3"></a>

The way to deal with multiple labe is to encode all labels for one sample into a string separated by ":", for example, sample with labels 17, 24, 35 will be encoded into string "17:24:35".

## Usage Example:
```
layer {
  name: "input"
  type: "Python"
  top: "Python1"
  top: "Python2"
  top: "Python3"
  python_param {
    module: "TripletDataLayer"
    layer: "TripletDataLayer"
    param_str: "{\'source_type\': \'CSV\', \'root\': \'../data/ALISC/train_image\', \'batch_size\': 32, \'source\': \'../data/ALISC/train_cat.csv\', \'prefetch\': True, \'type\': \'RANDOM\', \'resize\': [256, 256], \'compressed\': Tru
  }
}
```
