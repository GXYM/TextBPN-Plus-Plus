#!/bin/bash
root_dir="./gt_e2e/"
fn="/*.txt"
out_fn="gt_e2e.txt"

for element in `ls $root_dir`
    do
    echo $root_dir$element$fn
    `ls $root_dir$element$fn >>$out_fn`
done

