#!/bin/bash
# auth : Omar benabdejlil
echo -en "[+] - compiling each file ! .."
sleep 2 

for i in $(ls | grep -E "*.c" ); do nvcc $i|cut -d"." -f1 ./bin/; done
echo -e "[!] - finishing , bye !"

sleep 2 

