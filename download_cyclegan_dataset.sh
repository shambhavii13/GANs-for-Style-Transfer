FILE=$1

if [[ $FILE != "apple2orange" && $FILE != "vangogh2photo"  ]]; then
    echo "Available datasets are: apple2orange, vangogh2photo"
    exit 1
fi


echo "Specified [$FILE]"
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
