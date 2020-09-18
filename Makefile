download_cityscape:
		sh data/cityspace/download_cityspace.sh

prepare_cityscape:
		python data/cityspace/prepare_cityspace.py

run_cityscape:
		python iseg/run.py yamls/cityscape.yaml
