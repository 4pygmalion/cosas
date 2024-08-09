IMAGE_NAME := cosas:test

.PHONY: build 
build:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: SOURCE_PATH is not set. Please provide a valid path using 'make build MODEL_PATH=path/to/source'"; \
		exit 1; \
	fi
	cp ${MODEL_PATH} ./model.pth
	sudo docker build -t $(IMAGE_NAME) .

.PHONY: run
run:
	sudo docker run \
	--gpus all \
	-v /home/heon/dev/cosas/task2/input/input \
	-v /home/heon/dev/cosas/task2/output:/output \
	cosas:test