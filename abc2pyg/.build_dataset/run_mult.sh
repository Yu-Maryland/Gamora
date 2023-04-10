#input definitions
if [ "$#" -eq 2 ]; then
    s=$1
    copy=$2
fi
if [ "$#" -eq 3 ]; then
    s=$1
    s2=$2
    copy=$3
fi

#copies of designs
double=""
START=1
for i in $(eval echo "{$START..$copy}")
do
    double=$double"double;"
done

if [ "$#" -gt 3 ]; then
    echo "Error: maximum number of inputs is 4\n"
    exit
fi

# mult
if [ "$#" -eq 3 ]; then
    echo "./abc -c \"gen -N $s -m mult$s.blif;strash;&get;&get; &edgelist -F mult$s.el -c mult$s-class_map.json -f mult$s-feats.csv\"" | bash
    if [ "$copy" -gt 0 ]; then
        echo "./abc -c \"gen -N $s2 -m mult$s2.blif;$double;strash;&get;&get; &edgelist -F mult$s2.el -c mult$s2-class_map.json -f mult$s2-feats.csv\""
        echo "./abc -c \"gen -N $s2 -m mult$s2.blif;$double;strash;&get;&get; &edgelist -F mult$s2.el -c mult$s2-class_map.json -f mult$s2-feats.csv\"" | bash
    else
        echo "./abc -c \"gen -N $s2 -m mult$s2.blif;strash;&get;&get; &edgelist -F mult$s2.el -c mult$s2-class_map.json -f mult$s2-feats.csv\""
        echo "./abc -c \"gen -N $s2 -m mult$s2.blif;strash;&get;&get; &edgelist -F mult$s2.el -c mult$s2-class_map.json -f mult$s2-feats.csv\"" | bash
    fi
fi

mkdir mult"$s"_raw mult"$s2"_raw; mv mult"$s".el  mult"$s"-class_map.json mult"$s"-feats.csv ./mult"$s"_raw; mv mult"$s2".el  mult"$s2"-class_map.json mult"$s2"-feats.csv ./mult"$s2"_raw
