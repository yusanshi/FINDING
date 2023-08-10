# FINDING

The repository contains the code for the paper **Federated News Recommendation with Fine-grained Interpolation and Dynamic Clustering**.


## Get started

### Download the package

```bash
git clone https://github.com/yusanshi/FINDING
cd FINDING
export WORKING_DIRECTORY=`pwd`
```

### Download and process the datasets

Download and unzip GloVe pre-trained word embedding.

```bash
cd $WORKING_DIRECTORY
mkdir -p data/raw/glove && cd "$_"
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

Download punkt for NLTK.
```bash
python -c "import nltk; nltk.download('punkt')"
```

Download and process MIND-small dataset.

```bash
cd $WORKING_DIRECTORY
mkdir -p data/raw/mind-small && cd "$_"
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip \
 https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test # MIND Small has no test set :)
rm MINDsmall_*.zip

cd $WORKING_DIRECTORY
python -m fednewsrec.data_preprocess.mind \
 --source_dir=./data/raw/mind-small \
 --target_dir=./data/mind-small \
 --glove_path=./data/raw/glove/glove.840B.300d.txt
```

Download and process Adressa 1week dataset.
```bash
cd $WORKING_DIRECTORY
mkdir -p data/raw/adressa-1week && cd "$_"
wget https://reclab.idi.ntnu.no/dataset/one_week.tar.gz
tar -xzvf one_week.tar.gz
mv one_week/* .
rm -rf one_week*

cd $WORKING_DIRECTORY
python -m fednewsrec.data_preprocess.adressa \
 --source_dir=./data/raw/adressa-1week \
 --target_dir=./data/adressa-1week
```

Run.

```bash
python -m fednewsrec.train --dataset adressa-1week --model FindingNRMS
```

For more options, see `python -m fednewsrec.train -h`.

### Homomorphic Encryption

By passing `--homomorphic_encryption True`, you can enable the homomorphic encryption (HE). However we recommend not using it since: 
- it makes no difference to the model performance, but increases the computational cost.
- it needs some additional configuration, which is shown below.

The following assumes that you're using a Linux-based OS.

1. Go to `homomorphic-encryption/lib_ntl`, compile and install NTL library by the instructions from <https://libntl.org/doc/tour-unix.html>. You should install the header file to `/usr/include/NTL` and build the static library file as `lib_ntl/src/ntl.a`.
2. Go to `homomorphic-encryption/lib_fhesi`, compile FHE-SI library: `make`, create the static library file: `cd build && ar rvs libfhesi.a *.o`.
3. Build the executables.
    ```bash
    cd $WORKING_DIRECTORY/homomorphic-encryption
    
    mkdir -p bin
    
    for target in client server1 server2
    do
        mkdir -p $target/build && cd $_
        cmake .. && make
        cp ./$target ../../bin/
        cd -
    done
    ```
