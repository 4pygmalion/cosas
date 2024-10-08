IMAGE_NAME := cosas:test

.PHONY: build 
build:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: SOURCE_PATH is not set. Please provide a valid path using 'make build MODEL_PATH=path/to/checkpoint.pth'"; \
		exit 1; \
	fi
	cp ${MODEL_PATH} ./model.pth
	docker build --no-cache -t $(IMAGE_NAME) .

.PHONY: test_run
test_run:
	docker run \
	--gpus all \
	-v $(shell pwd)/task2/input/domain1:/input \
	-v $(shell pwd)/task2/output:/output \
	cosas:test


.PHONY: save
save:
	docker save cosas:test -o cosas.test.tar.gz