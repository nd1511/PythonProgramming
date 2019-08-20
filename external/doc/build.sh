rm -rf _build/html
make -e SPHINXOPTS="-D language='ja'" html
mv _build/html _build/ja

make html
mv _build/ja _build/html
