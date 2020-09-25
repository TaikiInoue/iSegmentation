docker_build:
		docker build -t iseg:latest -f docker/Dockerfile .

docker_run:
		docker run --runtime nvidia -it --rm \
			--network host \
			--workdir /app/iSegmentation \
			--volume ${PWD}:/app/iSegmentation
			--name iseg \
			--hostname iseg \
			iseg:latest /bin/bash

download_cityscape:
		sh data/cityscape/download_cityscape.sh

prepare_cityscape:
		python data/cityscape/prepare_cityscape_dataset.py

run_cityscape:
		python iseg/run.py yamls/cityscape.yaml
