#!/bin/sh

#python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True -k 5 -f 0

for k in 5 10 20
do
    for f in 2 3 4 5
    do
        python3 cmain.py -m stamp_cikm -d cikm16 -e 30 -r True -k $k -f $f

        python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True -k $k -f $f

        python3 cmain.py -m stamp_rsc -d rsc15_4 -e 30 -r True -k $k -f $f
    done
done

#stamp k=5
#python3 cmain.py -m stamp_cikm -d cikm16 -e 30 -r True -k 5 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True -k 5 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_4 -e 30 -r True -k 5 -f 5
#
##stamp k=10
#python3 cmain.py -m stamp_cikm -d cikm16 -e 30 -r True -k 10 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True -k 10 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_4 -e 30 -r True -k 10 -f 5
#
##stamp k=20
#python3 cmain.py -m stamp_cikm -d cikm16 -e 30 -r True -k 20 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True -k 20 -f 5
#
#python3 cmain.py -m stamp_rsc -d rsc15_4 -e 30 -r True -k 20 -f 5


