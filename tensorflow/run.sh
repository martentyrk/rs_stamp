#!/bin/sh

#stamp
python3 cmain.py -m stamp_cikm -d cikm16 -n -e 30 -r True

python3 cmain.py -m stamp_rsc -d rsc15_64 -n -e 30 -r True

python3 cmain.py -m stamp_rsc -d rsc15_4 -n -e 30 -r True


