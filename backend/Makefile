mlflow_up:
	docker compose -f ./docker/mlflow/compose.yml up -d

mlflow_down:
	docker compose -f ./docker/mlflow/compose.yml down

serving_up:
	set MODEL_NAME=${model_name} && set MODEL_ALIAS=${model_alias} && set DEVICE=cpu && docker compose -f ./docker/model_serving/compose.yml up -d

serving_down:
	set MODEL_NAME="" && set MODEL_ALIAS="" && set DEVICE="" && docker compose -f ./docker/model_serving/compose.yml down

all_up:
	make mlflow_up
	make model_name=resnet_18 model_alias=Production port=5000 serving_up

all_down:
	make mlflow_down
	make serving_down

