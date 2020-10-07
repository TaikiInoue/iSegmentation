echo -n CITYSPACES_USERNAME:
read CITYSPACES_USERNAME
echo -n CITYSPACES_PASSWORD:
read CITYSPACES_PASSWORD
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username='${CITYSPACES_USERNAME}'&password='${CITYSPACES_PASSWORD}'&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P data/cityscape
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P data/cityscape
unzip data/cityscape/gtFine_trainvaltest.zip
unzip data/cityscape/leftImg8bit_trainvaltest.zip
