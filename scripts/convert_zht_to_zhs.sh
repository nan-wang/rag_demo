#!/bin/zsh

# Convert Traditional Chinese to Simplified Chinese using opencc
# loop through all files in the directory `data` and convert them into Simplified Chinese and save them in the directory `data_zh`
# replace `雪梨` with `悉尼` in the converted files
# convert the file names to simplified chinese
for file in data/*.txt; do
    opencc -c t2s -i $file -o data_zh/$(basename $file)
done
