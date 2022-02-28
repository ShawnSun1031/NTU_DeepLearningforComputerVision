wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ny9crJorqiCOaQgrKbkFzbaSIGYvukFP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ny9crJorqiCOaQgrKbkFzbaSIGYvukFP" -O hw1_2_model.pth && rm -rf /tmp/cookies.txt
python3 hw1_2.py $1 $2



