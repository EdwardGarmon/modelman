#!bin/bash/
declare -a files=("MasterTokenizer.pkl" "jerry_model.h5" "elaine_model.h5" "kramer_model.h5" "george_model.h5")
declare burl="https://nothingforevermodels.s3.us-east-2.amazonaws.com/"

for i in "${files[@]}"
do
    curl "$burl$i" --output "./model/$i"
done

gunicorn app:app
