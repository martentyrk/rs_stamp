#!/bin/sh

#stamp
python3 cmain.py -m stamp_cikm -d cikm16 -e 30 -r True

python3 cmain.py -m stamp_rsc -d rsc15_64 -e 30 -r True

python3 cmain.py -m stamp_rsc -d rsc15_4 -e 30 -r True


